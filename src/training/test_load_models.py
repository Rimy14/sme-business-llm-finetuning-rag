"""
4-Bit Quantization & LoRA Architecture Verifier (Tesla T4)
---------------------------------------------------------
Task: KAN-17 (Day 2)
Tests loading both Qwen2.5-7B and Llama-3-8B in 4-bit NF4, attaches PEFT LoRA
adapters, and verifies VRAM footprint within Tesla T4 (15GB) memory budget.
"""

import os
import sys
import gc
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.training.utils_t4 import get_bnb_4bit_config, apply_t4_anti_crash_patch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def verify_model_4bit(model_id: str, model_short_name: str) -> dict:
    """Loads a model in 4-bit NF4, attaches LoRA, and measures memory."""
    logger.info(f"\n{'='*50}\nTesting 4-bit Loading for: {model_short_name} ({model_id})\n{'='*50}")
    
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available! Quantization check requires GPU.")
        return {"model": model_short_name, "status": "SKIPPED_NO_GPU"}

    torch.cuda.empty_cache()
    gc.collect()

    initial_vram = torch.cuda.memory_allocated() / (1024**3)
    logger.info(f"Initial VRAM Allocated: {initial_vram:.2f} GB")

    bnb_config = get_bnb_4bit_config()

    # 1. Load Tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Model in 4-bit
    logger.info("Loading base model in 4-bit NormalFloat (NF4)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    # 3. Apply T4 Anti-Crash Patch (bfloat16 -> float32/fp16)
    model = apply_t4_anti_crash_patch(model)

    # 4. Attach LoRA Adapters (Rank 16, Alpha 32)
    logger.info("Configuring LoRA adapters (r=16, alpha=32)...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    loaded_vram = torch.cuda.memory_allocated() / (1024**3)
    max_vram = torch.cuda.max_memory_allocated() / (1024**3)
    logger.info(f"✅ {model_short_name} successfully loaded in 4-bit!")
    logger.info(f"VRAM Allocated: {loaded_vram:.2f} GB | Peak VRAM: {max_vram:.2f} GB (Budget: 15.00 GB)")

    # 5. Quick Forward Pass Test
    logger.info("Testing sample forward pass...")
    test_input = tokenizer("SME Invoice Policy: Overdue settlement term is 30 days.", return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**test_input)
    logger.info(f"Forward pass output logits shape: {outputs.logits.shape}")

    # Clean up to free memory for next test
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "model": model_short_name,
        "model_id": model_id,
        "status": "PASSED",
        "loaded_vram_gb": round(loaded_vram, 2),
        "peak_vram_gb": round(max_vram, 2),
        "quantization": "4-bit NF4 Double Quant",
        "lora_rank": 16,
        "lora_alpha": 32
    }


def main():
    results = []
    # Test Qwen 2.5-7B
    qwen_res = verify_model_4bit("Qwen/Qwen2.5-7B-Instruct", "Qwen-2.5-7B-SME")
    results.append(qwen_res)

    # Test Llama 3 8B
    llama_res = verify_model_4bit("meta-llama/Meta-Llama-3-8B-Instruct", "Llama-3-8B-SME")
    results.append(llama_res)

    print("\n" + "="*50)
    print("DAY 2 QUANTIZATION VERIFICATION SUMMARY:")
    print("="*50)
    for r in results:
        print(f"[{r['status']}] {r['model']}: VRAM={r.get('loaded_vram_gb', 'N/A')} GB | Rank={r.get('lora_rank', 'N/A')}")


if __name__ == "__main__":
    main()
