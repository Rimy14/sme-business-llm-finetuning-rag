"""
SME Daily Business Scaled Dataset Collector
-------------------------------------------
Fetches ~11,000 domain-specific raw records for SME daily operations:
1. Customer support, billing & invoicing (7,500 samples from Bitext)
2. Financial & Accounting Q&A (3,000 samples from Financial QA 10K)
3. Operational SME daily business workflows (500 samples of SOPs & templates)

Saves normalized raw records to data/raw/sme_raw_dataset.jsonl
"""

import os
import json
import logging
from typing import List, Dict, Any
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")
OUTPUT_FILE = os.path.join(RAW_DATA_DIR, "sme_raw_dataset.jsonl")


def collect_bitext_customer_support(limit: int = 7500) -> List[Dict[str, Any]]:
    """Collects customer support, billing, invoice inquiries from Bitext dataset."""
    logger.info(f"Fetching {limit} customer support & billing samples from Hugging Face...")
    collected = []
    try:
        ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
        for i, item in enumerate(ds):
            if i >= limit:
                break
            instruction = item.get("instruction", "").strip()
            response = item.get("response", "").strip()
            category = item.get("intent", item.get("category", "customer_support"))
            if instruction and response:
                collected.append({
                    "source": "bitext_customer_support",
                    "category": category,
                    "instruction": instruction,
                    "context": "",
                    "response": response
                })
        logger.info(f"Successfully collected {len(collected)} samples from Bitext customer support.")
    except Exception as e:
        logger.warning(f"Could not download Bitext dataset ({e}). Generating fallback seed.")
        collected.extend(generate_curated_sme_seed(limit=limit))
    return collected


def collect_financial_qa(limit: int = 3000) -> List[Dict[str, Any]]:
    """Collects financial, bookkeeping, and cash flow Q&A."""
    logger.info(f"Fetching {limit} financial and accounting Q&A samples from Hugging Face...")
    collected = []
    try:
        ds = load_dataset("virattt/financial-qa-10K", split="train")
        for i, item in enumerate(ds):
            if i >= limit:
                break
            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()
            context = item.get("context", "").strip()
            if question and answer:
                collected.append({
                    "source": "financial_qa_10k",
                    "category": "financial_accounting",
                    "instruction": question,
                    "context": context,
                    "response": answer
                })
        logger.info(f"Successfully collected {len(collected)} samples from Financial QA.")
    except Exception as e:
        logger.warning(f"Could not download Financial QA dataset ({e}). Generating fallback financial seed.")
        collected.extend(generate_curated_sme_seed(limit=limit))
    return collected


def generate_curated_sme_seed(limit: int = 500) -> List[Dict[str, Any]]:
    """Generates rich SME operational SOP templates."""
    sample_templates = [
        ("How should an SME draft an invoice overdue notice for client {client}?",
         "Subject: Polite Reminder: Overdue Payment for Invoice #{inv_num}\n\nDear {client} Accounts Team,\n\nWe hope this note finds you well. Our records indicate that Invoice #{inv_num} for ${amt} was due on {date}. If the payment has already been processed, please disregard this notice. Otherwise, please provide an estimated settlement date.\n\nThank you for your ongoing partnership.\n\nBest regards,\nFinance & Accounts Department",
         "billing_invoicing"),
        ("What is the standard procedure for a warehouse inventory stock cycle count?",
         "1. Freeze all inventory movements in the target warehouse zone.\n2. Print system stock inventory sheets.\n3. Conduct physical dual-counting (two staff members independently count).\n4. Log any discrepancy exceeding 1.5% and conduct an immediate recount.\n5. Post approved stock adjustment journal in the ERP system with manager sign-off.",
         "inventory_management"),
        ("How do I calculate Gross Profit Margin vs Net Profit Margin for my business?",
         "Gross Profit Margin = ((Revenue - Cost of Goods Sold) / Revenue) * 100.\nNet Profit Margin = ((Revenue - All Operating Expenses, Taxes, and Interest) / Revenue) * 100.\nGross margin evaluates production efficiency, while Net margin assesses comprehensive enterprise profitability.",
         "financial_accounting"),
        ("Draft a purchase order follow-up email to a supplier regarding delayed raw materials.",
         "Subject: Urgent: Delivery Status Inquiry for Purchase Order PO-{po_num}\n\nDear Supplier Team,\n\nPO-{po_num} was scheduled for delivery by {date}. We have not yet received tracking details or dispatch confirmation. As our production lines rely on this shipment, please provide tracking information and ETA immediately.\n\nSincerely,\nProcurement Lead",
         "procurement_vendor"),
        ("What are the mandatory payroll deduction guidelines for SME employees?",
         "1. Statutory Income Tax Withholding (TDS / PAYE based on tax brackets).\n2. Social Security / Pension / Provident Fund mandatory employee and employer shares.\n3. State disability / health insurance contributions.\n4. Documented pre-tax employee deductions (health savings, voluntary retirement top-ups).",
         "hr_payroll"),
        ("How should an SME handle customer warranty claims for defective products?",
         "1. Validate proof of purchase, serial number, and warranty period eligibility.\n2. Request photo/video evidence or inspect physical item at service hub.\n3. Issue Return Merchandise Authorization (RMA) ticket within 24 hours.\n4. Provide replacement or refund within 5 business days per SLA.",
         "customer_support"),
        ("Draft a standard nondisclosure agreement (NDA) clause for vendor pricing confidentiality.",
         "The Vendor and Buyer agree that all commercial terms, discount tiers, price schedules, and technical specifications exchanged shall remain strictly confidential and shall not be disclosed to third parties without prior written consent for a period of two (2) years.",
         "legal_compliance")
    ]
    seed_data = []
    for i in range(limit):
        tpl = sample_templates[i % len(sample_templates)]
        inv_num = 1000 + i
        po_num = 5000 + i
        amt = (i + 1) * 175
        seed_data.append({
            "source": "sme_curated_ops",
            "category": tpl[2],
            "instruction": tpl[0].format(client=f"Client-{i+1}", inv_num=inv_num, po_num=po_num, amt=amt, date="2026-09-30"),
            "context": "",
            "response": tpl[1].format(client=f"Client-{i+1}", inv_num=inv_num, po_num=po_num, amt=amt, date="2026-09-30")
        })
    return seed_data


def main():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    all_data = []

    # 1. Customer Support & Billing (7,500)
    all_data.extend(collect_bitext_customer_support(limit=7500))

    # 2. Financial & Bookkeeping QA (3,000)
    all_data.extend(collect_financial_qa(limit=3000))

    # 3. SME Operational SOPs (500)
    all_data.extend(generate_curated_sme_seed(limit=500))

    logger.info(f"Total raw records gathered: {len(all_data)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in all_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Successfully saved scaled raw dataset ({len(all_data)} records) to: {OUTPUT_FILE}")
    print(f"\n[DONE] Dataset Collection Complete! Total records: {len(all_data)}")


if __name__ == "__main__":
    main()
