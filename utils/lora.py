from peft import (
    LoraConfig,
    get_peft_model
)

def lora_config_setup(args):
    """
    LoRA config_setup
    """
    lora_config = LoraConfig(
        r=args.lora_rank,   # lora rank
        lora_alpha=args.lora_alpha,   
        target_modules=[  
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attn layer
            "gate_proj", "up_proj", "down_proj"     # MLP layer
        ], 
        lora_dropout=args.lora_dropout,
        task_type=args.lora_tasktype  # 为了清晰起见显式指定
    )
    return lora_config

def get_lora_model(model, lora_config):
    """
    get lora model and print trainable parameters
    """
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model