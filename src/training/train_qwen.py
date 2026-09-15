"""
Qwen 2.5 7B Instruct SFTTrainer Pipeline (Tesla T4 Optimized)
------------------------------------------------------------
Task: KAN-21 (Day 3)
Domain: SME Daily Business (SME-Daily-Business)
Model: Qwen/Qwen2.5-7B-Instruct
Quantization: 4-bit NF4 Double Quantization
PEFT: LoRA (r=16, alpha=32, dropout=0.05 on all linear layers)
"""

import os
import sys
import gc
import json
import yaml
import torch
import logging
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from trl import SFTTrainer
import wandb

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.training.utils_t4 import get_bnb_4bit_config, apply_t4_anti_crash_patch, get_sme_lora_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def format_chatml(example: dict, tokenizer) -> str:
    """Formats an SME sample into Qwen ChatML template."""
    instruction = example.get("instruction", "").strip()
    context = example.get("context", "").strip()
    response = example.get("response", "").strip()

    system_prompt = "You are an expert AI assistant specializing in SME daily business operations, invoicing, bookkeeping, inventory management, customer support, and HR policies."

    user_content = instruction
    if context:
        user_content = f"Context:\n{context}\n\nQuestion:\n{instruction}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": response}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def run_training(config_path: str, max_train_samples: int = None, max_val_samples: int = None, use_wandb: bool = False):
    """Executes SFTTrainer fine-tuning for Qwen2.5-7B on SME dataset."""
    logger.info(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_id = cfg.get("model_name_or_path", "Qwen/Qwen2.5-7B-Instruct")
    train_data_path = cfg.get("dataset_name", "data/processed/train_v1.json")
    val_data_path = cfg.get("val_dataset_name", "data/processed/val_v1.json")
    output_dir = cfg.get("output_dir", "models/v1/qwen_sme_v1")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Initialize W&B (Optional)
    if use_wandb:
        wandb.init(
            project="sme-daily-business-llm",
            name="qwen-2.5-7b-sme-v1",
            config=cfg
        )

    # 2. Load Datasets
    logger.info(f"Loading training data from {train_data_path}...")
    with open(train_data_path, "r", encoding="utf-8") as f:
        train_raw = json.load(f)
    with open(val_data_path, "r", encoding="utf-8") as f:
        val_raw = json.load(f)

    if max_train_samples and max_train_samples < len(train_raw):
        train_raw = train_raw[:max_train_samples]
    if max_val_samples and max_val_samples < len(val_raw):
        val_raw = val_raw[:max_val_samples]

    logger.info(f"Training on {len(train_raw)} samples | Validating on {len(val_raw)} samples")

    # 3. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4. Format Prompts
    train_texts = [format_chatml(x, tokenizer) for x in train_raw]
    val_texts = [format_chatml(x, tokenizer) for x in val_raw]

    train_ds = Dataset.from_dict({"text": train_texts})
    val_ds = Dataset.from_dict({"text": val_texts})

    # 5. Load 4-Bit Base Model
    logger.info(f"Loading {model_id} in 4-bit NF4...")
    bnb_config = get_bnb_4bit_config()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    # 6. Apply T4 Anti-Crash Patch & LoRA
    model = apply_t4_anti_crash_patch(model)
    peft_cfg = cfg.get("peft", {})
    model = get_sme_lora_model(
        model,
        r=peft_cfg.get("r", 16),
        alpha=peft_cfg.get("lora_alpha", 32),
        dropout=peft_cfg.get("lora_dropout", 0.05)
    )

    # 7. Training Arguments
    t_cfg = cfg.get("training", {})
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=t_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=t_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=t_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=t_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(t_cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=t_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=t_cfg.get("warmup_ratio", 0.05),
        weight_decay=t_cfg.get("weight_decay", 0.01),
        fp16=True,
        bf16=False,
        logging_steps=t_cfg.get("logging_steps", 10),
        eval_strategy="steps",
        eval_steps=t_cfg.get("eval_steps", 50),
        save_strategy="steps",
        save_steps=t_cfg.get("save_steps", 100),
        save_total_limit=t_cfg.get("save_total_limit", 2),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if use_wandb else "none",
        seed=42
    )

    # 8. SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=t_cfg.get("max_seq_length", 512),
        tokenizer=tokenizer,
        args=training_args
    )

    logger.info("Starting SFTTrainer fine-tuning...")
    trainer.train()

    # 9. Save Final Model & Adapters
    logger.info(f"Saving final fine-tuned adapter to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training summary
    metrics = trainer.state.log_history
    with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("✅ Qwen 2.5-7B SME Training Completed Successfully!")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    cfg_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs", "qwen_sme_config.yaml")
    run_training(cfg_file)
