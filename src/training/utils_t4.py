"""
Tesla T4 GPU Stability & Anti-Crash Guard Module
------------------------------------------------
Provides critical fixes for QLoRA fine-tuning on Tesla T4:
1. Prevents PyTorch GradientScaler crashes due to bfloat16 buffers on T4.
2. Force-casts PEFT adapter layers to float32.
3. Sets up BitsAndBytesConfig for 4-bit NF4 double quantization.
"""

import torch
import logging
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

logger = logging.getLogger(__name__)


def get_bnb_4bit_config() -> BitsAndBytesConfig:
    """Returns standard 4-bit NF4 double quantization configuration."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )


def apply_t4_anti_crash_patch(model):
    """
    Force-casts all bfloat16 parameters and buffers in the base model to float32 or float16.
    Ensures model.config.torch_dtype is float32 so PEFT creates adapters in float32.
    """
    logger.info("Applying Tesla T4 anti-crash patches...")
    
    # 1. Prepare model for kbit training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    # 2. Force config torch_dtype to float32 for adapter safety
    if hasattr(model, "config"):
        model.config.torch_dtype = torch.float32
        model.config.use_cache = False
    
    # 3. Cast any rogue bfloat16 buffers to float32
    for name, module in model.named_modules():
        for buf_name, buf in module.named_buffers():
            if buf.dtype == torch.bfloat16:
                setattr(module, buf_name, buf.to(torch.float32))
                logger.debug(f"Cast buffer {name}.{buf_name} from bfloat16 -> float32")
                
    logger.info("T4 anti-crash patch applied successfully.")
    return model


def get_sme_lora_model(model, r=16, alpha=32, dropout=0.05):
    """Wraps model with LoRA targeting all linear attention and MLP projections."""
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model
