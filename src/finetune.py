import argparse
import os
import peft
import torch
import warnings

import datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)
from transformers.tokenization_utils_base import logger as tokenization_logger

from configs.finetune_config import FinetuneConfig
from prompt.prompter import Prompter

warnings.filterwarnings(
    "ignore",
    message=".*GPTNeoXTokenizerFast.*",
    category=UserWarning,
    module="transformers.tokenization_utils_base",
)

# Arg definitions
parser = argparse.ArgumentParser(description="finetune llm model using lora")
parser.add_argument(
    "--conf_path",
    default="/home/user0/work/data/config/default.yaml",
    help="conf file for finetune using lora")

args = parser.parse_args()
torch.cuda.empty_cache()


tokenization_logger.setLevel("ERROR")


class TokenizerHelper:
    def __init__(
        self, prompter, tokenizer, train_on_inputs, cutoff_len, add_eos_token=True
    ):
        self.prompter = prompter
        self.tokenizer = tokenizer
        self.train_on_inputs = train_on_inputs
        self.add_eos_token = add_eos_token
        self.cutoff_len = cutoff_len

    def tokenize(self, prompt):
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.cutoff_len,
            # Set padding to 'max_length' instead of False for GPTNeoXTokenizerFast???
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.cutoff_len
            and self.add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        return result

    def generate_and_tokenize_prompt(self, data_point):
        full_prompt = self.prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = self.tokenize(full_prompt)

        if not self.train_on_inputs:
            user_prompt = self.prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = self.tokenize(user_prompt)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if self.add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["input_ids"][
                user_prompt_len:
            ]  # could be sped up, probably
        else:
            tokenized_full_prompt["labels"] = tokenized_full_prompt["input_ids"]

        return tokenized_full_prompt


def load_model(conf: FinetuneConfig, device_map: any) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
    model = AutoModelForCausalLM.from_pretrained(
        conf.base_model,
        trust_remote_code=True,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        quantization_config=quantization_config,
        cache_dir=conf.model_cache_dir,
    )

    # 8-bit training
    # had to turn int8 training off for some reason. could it be the titan rtx?
    # turned it on and kinda working now, but wtf?

    # model = peft.prepare_model_for_int8_training(model)


    tokenizer = AutoTokenizer.from_pretrained(
        conf.base_model,
        use_fast=False,
    )

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    return model, tokenizer


def apply_lora(model:AutoModelForCausalLM, conf: FinetuneConfig) -> AutoModelForCausalLM:
    #
    # LoRA
    #
    config = peft.LoraConfig(
        r=conf.lora_r,
        lora_alpha=conf.lora_alpha,
        target_modules=conf.lora_target_modules,
        lora_dropout=conf.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = peft.get_peft_model(model, config)
    return model


def load_dataset(conf: FinetuneConfig):
    if conf.data_path.endswith(".json") or conf.data_path.endswith(".jsonl"):
        data = datasets.load_dataset("json", data_files=conf.data_path)
    else:
        data = datasets.load_dataset(conf.data_path)
    
    return data


def main():
    conf: FinetuneConfig = FinetuneConfig.load(args.conf_path)
    conf.print()


    device_map = "auto"
    gradient_accumulation_steps = conf.batch_size // conf.micro_batch_size

    # use DistributedDataParallel if needed
    ddp = conf.num_of_gpus > 1
    if ddp:
        device_map = {"": conf.local_rank}
        gradient_accumulation_steps = gradient_accumulation_steps // conf.num_of_gpus
    print(f"device map: {device_map}")

    model, tokenizer = load_model(conf, device_map)
    model = apply_lora(model, conf)
    if conf.resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            conf.resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                conf.resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            conf.resume_from_checkpoint = False  # So the trainer won't try loading its state
        # The two files above have a different name depending on how they were saved,
        # but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            peft.set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    # Be more transparent about the % of trainable params.
    model.print_trainable_parameters()

    data = load_dataset(conf)
    tokenizer_helper = TokenizerHelper(
        Prompter(conf.prompt_template_name, conf.prompt_template_dir_path),
        tokenizer,
        conf.train_on_inputs,
        conf.cutoff_len,
        conf.add_eos_token
    )

    if conf.val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=conf.val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"]
            .shuffle()
            .map(tokenizer_helper.generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"]
            .shuffle()
            .map(tokenizer_helper.generate_and_tokenize_prompt)
        )
    else:
        train_data = (
            data["train"].shuffle().map(tokenizer_helper.generate_and_tokenize_prompt)
        )
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism
        # when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    use_wandb = False
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=TrainingArguments(
            per_device_train_batch_size=conf.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=conf.num_epochs,
            learning_rate=conf.learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if conf.val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if conf.val_set_size > 0 else None,
            save_steps=200,
            output_dir=conf.output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if conf.val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=conf.group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=conf.wandb_run_name if use_wandb else None,
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: peft.get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    if torch.__version__ >= "2":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=conf.resume_from_checkpoint)

    model.save_pretrained(conf.output_dir)

    print("\n If there's a warning about missing keys above, please disregard :)")


if __name__ == "__main__":
    main()
