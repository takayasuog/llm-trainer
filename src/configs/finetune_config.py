import dataclasses
from configs.config import Config

@dataclasses.dataclass
class FinetuneConfig(Config):
    # model/data params
    base_model: str                 # the only required argument
    data_path: str
    output_dir: str
    model_cache_dir: str 
    # wandb params
    wandb_run_name: str
    # training hyperparams
    local_rank: int
    num_of_gpus: int
    batch_size: int
    micro_batch_size: int
    num_epochs: int
    learning_rate: float
    cutoff_len: int
    val_set_size: int
    # lora hyperparams
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list[str]
    train_on_inputs: bool           # if False, masks out inputs in loss
    add_eos_token: bool
    group_by_length: bool           # faster, but produces an odd training loss curve
    resume_from_checkpoint: str     # either training checkpoint or final adapter
    prompt_template_name: str       # Prompt template to use, default to Alpaca
    prompt_template_dir_path: str   # Dir path for prompt template files