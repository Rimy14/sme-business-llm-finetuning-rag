"""
SME Daily Business Data Cleaning & 80/10/10 Formatter
----------------------------------------------------
Task: KAN-17 (Day 2)
Cleans raw domain records from data/raw/sme_raw_dataset.jsonl:
1. Strips corrupted/empty records and deduplicates.
2. Normalizes whitespace, quotes, and placeholders.
3. Standardizes into canonical schema: {"instruction": "...", "context": "...", "response": "..."}
4. Deterministic 80/10/10 train/val/test split saved to data/processed/
"""

import os
import re
import json
import random
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")
RAW_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw", "sme_raw_dataset.jsonl")


def clean_text(text: str) -> str:
    """Normalizes whitespace and standardizes template placeholders."""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'[\r\t]+', ' ', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


def process_dataset(raw_path: str, output_dir: str, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42) -> Tuple[int, int, int]:
    """Cleans and splits raw SME dataset into 80/10/10 train/val/test sets."""
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset not found at: {raw_path}. Run Day 1 collection first!")

    logger.info(f"Loading raw records from: {raw_path}")
    raw_records = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    raw_records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    logger.info(f"Total raw records loaded: {len(raw_records)}")

    # Clean and standardize
    cleaned_records = []
    seen_instructions = set()

    for item in raw_records:
        inst = clean_text(item.get("instruction", ""))
        resp = clean_text(item.get("response", ""))
        ctx = clean_text(item.get("context", ""))
        category = item.get("category", "general_sme")

        if len(inst) < 5 or len(resp) < 5:
            continue

        # Deduplication key
        dedup_key = inst.lower()
        if dedup_key in seen_instructions:
            continue
        seen_instructions.add(dedup_key)

        cleaned_records.append({
            "instruction": inst,
            "context": ctx,
            "response": resp,
            "category": category
        })

    logger.info(f"Total valid unique records after cleaning: {len(cleaned_records)}")

    # Deterministic Shuffle
    random.seed(seed)
    random.shuffle(cleaned_records)

    # 80 / 10 / 10 Split
    total = len(cleaned_records)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    
    train_data = cleaned_records[:n_train]
    val_data = cleaned_records[n_train:n_train + n_val]
    test_data = cleaned_records[n_train + n_val:]

    train_file = os.path.join(output_dir, "train_v1.json")
    val_file = os.path.join(output_dir, "val_v1.json")
    test_file = os.path.join(output_dir, "test_v1.json")

    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open(val_file, "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Split completed successfully:")
    logger.info(f"   - Train (80%): {len(train_data):,} samples -> {train_file}")
    logger.info(f"   - Val   (10%): {len(val_data):,} samples -> {val_file}")
    logger.info(f"   - Test  (10%): {len(test_data):,} samples -> {test_file}")

    return len(train_data), len(val_data), len(test_data)


if __name__ == "__main__":
    process_dataset(RAW_FILE, PROCESSED_DIR)
