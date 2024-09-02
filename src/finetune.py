import argparse
import os
import peft
import torch
import warnings

import bitsandbytes as bnb
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPTNeoXTokenizerFast,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)
from transformers.tokenization_utils_base import logger as tokenization_logger

from configs.finetune_config import FinetuneConfig
from utils.datasets import load_dataset
from utils.deepspeed import get_train_ds_config
from utils.prompter import Prompter

warnings.filterwarnings(
    "ignore",
    message=".*GPTNeoXTokenizerFast.*",
    category=UserWarning,
    module="transformers.tokenization_utils_base",
)

# Arg definitions
parser = argparse.ArgumentParser(description="finetune llm model using lora")
parser.add_argument(
    "--local_rank",
    default="",
    help="dummy")
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
    n_gpus = torch.cuda.device_count()
    max_memory = f'{conf.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}

    compute_dtype = torch.bfloat16
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        load_in_8bit=False,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4", # {'fp4', 'nf4'}
    )
    model = AutoModelForCausalLM.from_pretrained(
        conf.base_model,
        # load_in_4bit=True,
        # load_in_8bit=False,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir=conf.model_cache_dir,
        attn_implementation="eager",
    )

    model.config.torch_dtype=torch.bfloat16
    model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            conf.base_model,
            use_fast=False,
            cache_dir=conf.model_cache_dir,
        )
    except ValueError as e:
        model_config = AutoConfig.from_pretrained(conf.base_model)
        if model_config.model_type == "gpt_neox":
            tokenizer = GPTNeoXTokenizerFast.from_pretrained(
                conf.base_model,
                use_fast=False,
                cache_dir=conf.model_cache_dir,
            )
        else:
            raise

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    return model, tokenizer


def find_all_linear_names(model: AutoModelForCausalLM) -> list:
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')

    return list(lora_module_names)


def apply_lora(model:AutoModelForCausalLM, conf: FinetuneConfig) -> AutoModelForCausalLM:
    #
    # LoRA
    #
    modules:list = find_all_linear_names(model) \
                        if not conf.lora_target_modules else conf.lora_target_modules
    config = peft.LoraConfig(
        r=conf.lora_r,
        lora_alpha=conf.lora_alpha,
        target_modules=modules,
        lora_dropout=conf.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = peft.get_peft_model(model, config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    for name, module in model.named_modules():
        if isinstance(module, peft.tuners.lora.LoraLayer):
            module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    return model


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    trainable_params /= 2
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable: {100 * trainable_params / all_param}")


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
    else:
        conf.resume_from_checkpoint = False

    # Be more transparent about the % of trainable params.
    print_trainable_parameters(model)

    data = load_dataset(conf)
    tokenizer_helper = TokenizerHelper(
        Prompter(conf.prompt_template_name, conf.prompt_template_dir_path),
        tokenizer,
        conf.train_on_inputs,
        conf.cutoff_len,
        conf.add_eos_token
    )
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

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

    ds_config = get_train_ds_config(offload=True, conf=conf) if conf.deepspeed_enabled else None
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
            logging_steps=10,
            optim="paged_adamw_32bit",
            bf16=True,
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
            deepspeed=ds_config,
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
