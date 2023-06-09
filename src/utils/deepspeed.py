from configs.finetune_config import FinetuneConfig


def get_train_ds_config(offload, 
                        conf: FinetuneConfig):

    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": conf.stage,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/home/user0/.ds_offload_cache",
            "pin_memory": True,
            "pipeline_read": True,
            "buffer_count": 10,
            "fast_init": True
        },
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/home/user0/.ds_offload_cache",
            "pin_memory": True,
            "buffer_count": 10,
            "buffer_size": 1e8,
            "max_in_cpu": 1e9
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 5e8,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": "auto",
        "stage3_max_reuse_distance": "auto",
        "stage3_gather_16bit_weights_on_model_save": True,
        "memory_efficient_linear": False
    }
    return {
        "train_batch_size": conf.batch_size,
        "train_micro_batch_size_per_gpu": conf.micro_batch_size,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "fp16": {
            "enabled": True,
            "loss_scale_window": 100
        },
        "aio": {
            "block_size": 262144,
            "queue_depth": 32,
            "thread_count": 1,
            "single_submit": False,
            "overlap_events": True
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": conf.learning_rate,
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto"
            }
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": conf.enable_hybrid_engine,
            "max_out_tokens": conf.max_out_tokens,
            "inference_tp_size": conf.inference_tp_size,
            "release_inference_cache": conf.release_inference_cache,
            "pin_parameters": conf.pin_parameters,
            "tp_gather_partition_size": conf.tp_gather_partition_size,
        }
    }


def get_eval_ds_config(offload, conf: FinetuneConfig):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": conf.stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {
            "device": device
        },
        "memory_efficient_linear": False
    }
    return {
        "train_batch_size": conf.batch_size,
        "train_micro_batch_size_per_gpu": conf.micro_batch_size,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "fp16": {
            "enabled": True
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }
