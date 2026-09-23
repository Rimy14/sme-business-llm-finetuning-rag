"""
RAG-Aware Dataset Builder for SME Daily Business (Day 7 - KAN-39)
Injects realistic SME enterprise context (SOPs, accounting standards, supply chain guidelines)
into instruction-response pairs to create train_v3.json, val_v3.json, and test_v3.json.
Also exports raw enterprise knowledge documents to data/rag_docs/ for Day 9 ChromaDB integration.
"""

import os
import json
import random

# ── 1. Enterprise SME SOPs & Policies (for RAG Knowledge Base) ───────────────────────────
ENTERPRISE_RAG_DOCS = {
    "cash_flow_and_working_capital_sop.md": """# SME Standard Operating Procedure: Cash Flow & Working Capital Management
**Document ID:** SOP-FIN-001 | **Version:** 3.2 | **Applies To:** All Financial & Accounting Staff

## 1. Cash Conversion Cycle (CCC) Policy
- **Target CCC:** Maintain an operating Cash Conversion Cycle between 30 and 45 days.
- **Days Sales Outstanding (DSO):** Invoices must carry payment terms of Net 30. Early settlement discount of 2/10 Net 30 is authorized for clients with order volumes exceeding $10,000.
- **Days Payable Outstanding (DPO):** All supplier payments must be scheduled exactly on their due date (Net 45 or Net 60) via automated batch ACH to maximize working capital liquidity.
- **Emergency Reserve Threshold:** A mandatory liquid reserve equal to at least 90 days (3 months) of fixed operational expenditure (OpEx) must be maintained in an insured sweep account.

## 2. Short-Term Financing & Invoicing
- Invoices aged past 60 days without dispute resolution may be submitted for non-recourse invoice factoring at fee rates not exceeding 2.5% per 30-day tranche.
- Revolving Credit Lines (LOC) should only be tapped for seasonal inventory builds where gross margin exceeds the annualized borrowing cost by at least 15%.
""",

    "inventory_control_and_procurement_sop.md": """# SME Standard Operating Procedure: Inventory Control & Procurement
**Document ID:** SOP-OPS-004 | **Version:** 2.8 | **Applies To:** Warehouse, Supply Chain & Purchasing

## 1. Reorder Point (ROP) & Safety Stock Formulas
- **Reorder Point (ROP):** $\\text{ROP} = (\\text{Daily Demand} \\times \\text{Lead Time in Days}) + \\text{Safety Stock}$
- **Safety Stock Buffer:** For critical class-A items, safety stock must cover a minimum of 14 days of average demand. For class-B and class-C items, a 7-day safety buffer is mandated.
- **Economic Order Quantity (EOQ):** Purchase orders must balance holding cost (calculated at 18% annual inventory value) against fixed purchase order processing costs.

## 2. Supplier Price Hike Escalation & Dual Sourcing
- When a sole vendor issues a price hike exceeding 5%, procurement must immediately initiate RFQs across qualified secondary suppliers.
- Minimum 25% of annual raw material volume must be allocated to an alternative domestic supplier to insulate against overseas freight bottlenecks.
""",

    "tax_compliance_and_payroll_guidelines.md": """# SME Standard Operating Procedure: Tax Compliance & Payroll Operations
**Document ID:** SOP-TAX-007 | **Version:** 4.1 | **Applies To:** HR, Payroll & Finance

## 1. Worker Classification Standards (W-2 vs 1099)
- **Behavioral Control:** If the SME dictates specific working hours, mandatory software tools, and direct supervision, the worker MUST be classified as a W-2 Employee.
- **Independent Contractors (1099):** Permitted only when the worker retains independence in project execution, provides their own equipment, and maintains an independent business entity (LLC/EIN).
- **Penalties:** Misclassification incurs retroactive employer payroll taxes (7.65% FICA), state unemployment insurance back-pay, and mandatory statutory interest.

## 2. Tax Depreciations & Section 179 Deductions
- Qualifying capital machinery and technological infrastructure purchased for business use can be fully expensed in Year 1 under IRS Section 179 up to allowable federal caps.
""",

    "pricing_and_unit_economics_policy.md": """# SME Standard Operating Procedure: Pricing Strategy & Margin Control
**Document ID:** SOP-STRAT-002 | **Version:** 2.1 | **Applies To:** Sales, Product & Commercial Management

## 1. Margin & Break-Even Standards
- **Gross Margin Threshold:** No product line may be priced with a gross margin below 35% without explicit CFO sign-off.
- **Break-Even Unit Formula:** $\\text{Break-Even Units} = \\frac{\\text{Fixed Operating Costs}}{\\text{Selling Price} - \\text{Variable Cost per Unit}}$.
- **Discounting Limits:** Sales representatives are authorized to offer maximum 10% volume discounts for orders exceeding 500 units. Discounts between 11% and 20% require Director approval.

## 2. Customer Lifetime Value (LTV) to CAC Ratio
- Commercial acquisition campaigns must target a Minimum LTV:CAC ratio of 3.0x over a 24-month horizon.
"""
}

# ── 2. RAG-Aware Training Pair Templates ────────────────────────────────────────────────
RAG_CONTEXT_TEMPLATES = [
    {
        "context_header": "Enterprise Policy Excerpt — SOP-FIN-001 (Cash Flow & Working Capital):",
        "context_body": "According to SOP-FIN-001 Section 1, standard customer payment terms are Net 30, with an early-payment discount of 2/10 Net 30 authorized for orders exceeding $10,000. Supplier payables must be paid on Net 45/60 terms. The company requires a minimum liquid emergency buffer equal to 90 days (3 months) of fixed operational expenses.",
        "pairs": [
            ("What payment terms should our sales team offer a new client ordering $15,000 in goods, and how can we encourage fast payment?",
             "Based on **SOP-FIN-001 Section 1**, standard payment terms are **Net 30**. Since the client's order of $15,000 exceeds the $10,000 threshold, you are authorized to offer an early-settlement incentive of **2/10 Net 30** (a 2% discount if paid within 10 days, otherwise full balance due in 30 days) to accelerate cash collection."),
            ("What is our company's mandatory cash reserve policy for fixed monthly operating costs?",
             "According to **SOP-FIN-001 Section 1**, the company mandates maintaining a liquid emergency cash reserve equal to at least **90 days (3 months) of fixed operational expenditure (OpEx)** in an insured sweep account to protect against operational shortfalls.")
        ]
    },
    {
        "context_header": "Enterprise Policy Excerpt — SOP-OPS-004 (Inventory Control & Procurement):",
        "context_body": "Under SOP-OPS-004 Section 1, Reorder Point is calculated as (Daily Demand × Lead Time) + Safety Stock. Critical Class-A inventory items require a minimum 14-day safety stock buffer, while Class-B/C items require a 7-day buffer. Section 2 dictates that at least 25% of annual material volume must be allocated to a qualified secondary domestic supplier.",
        "pairs": [
            ("How should we calculate the Reorder Point for a critical Class-A component with a daily usage of 20 units and a supplier lead time of 10 days?",
             "Under **SOP-OPS-004 Section 1**:\n1. **Lead Time Demand:** 20 units/day × 10 days = 200 units.\n2. **Safety Stock Buffer:** For critical Class-A items, policy mandates 14 days of safety stock: 14 days × 20 units/day = 280 units.\n3. **Reorder Point (ROP):** 200 + 280 = **480 units**.\n\nA purchase order must be issued immediately when stock reaches 480 units."),
            ("What is our supply chain dual-sourcing rule when working with foreign component vendors?",
             "According to **SOP-OPS-004 Section 2**, at least **25% of annual raw material volume must be allocated to an alternative qualified domestic supplier** to protect the company against international logistics delays and single-source dependencies.")
        ]
    },
    {
        "context_header": "Enterprise Policy Excerpt — SOP-TAX-007 (Worker Classification & Compliance):",
        "context_body": "SOP-TAX-007 mandates that workers subject to direct supervision, set working hours, and mandatory company tools must be classified as W-2 employees. 1099 contractor status is only permitted for individuals operating an independent business entity with control over their own schedules and tools.",
        "pairs": [
            ("Can we hire a full-time customer support specialist on a 1099 contractor basis if we require them to work 9-5 using our company laptop?",
             "No. Under **SOP-TAX-007 Section 1**, requiring set working hours (9-5), mandatory company equipment, and direct operational control classifies the role as a **W-2 Employee**. Misclassifying them as a 1099 contractor violates company compliance and creates tax penalty liabilities."),
            ("What criteria must be met to legitimately hire a specialist as an independent contractor under company policy?",
             "According to **SOP-TAX-007**, an independent contractor (1099) must:\n1. Retain independent control over their execution methods and schedule.\n2. Provide their own tools and equipment.\n3. Operate under an independent legal business entity (LLC or EIN).")
        ]
    },
    {
        "context_header": "Enterprise Policy Excerpt — SOP-STRAT-002 (Pricing & Margin Control):",
        "context_body": "SOP-STRAT-002 Section 1 establishes a strict minimum gross margin floor of 35% across all product lines. Sales representatives are authorized to grant discounts up to 10% on orders of 500+ units. Discounts between 11% and 20% require Director approval, and anything above 20% requires CFO sign-off.",
        "pairs": [
            ("A sales rep wants to offer a 15% discount on an order of 800 units. Who needs to approve this?",
             "Under **SOP-STRAT-002 Section 1**, discounts between 11% and 20% **require Director approval**. The sales rep cannot approve this unilaterally, but can proceed once written Director sign-off is obtained."),
            ("What is our company's gross margin policy for launching a new product line?",
             "According to **SOP-STRAT-002 Section 1**, no product line may be priced with a **gross margin below 35%** without explicit written CFO approval.")
        ]
    }
]


def generate_rag_aware_dataset(n_samples=600):
    """Generate structured RAG-aware instruction-context-response samples."""
    samples = []
    random.seed(42)

    for i in range(n_samples):
        grp = random.choice(RAG_CONTEXT_TEMPLATES)
        q, a = random.choice(grp["pairs"])
        full_context = f"{grp['context_header']}\n{grp['context_body']}"
        samples.append({
            "instruction": q,
            "context": full_context,
            "response": a,
            "is_rag_aware": True
        })
    return samples


def build_train_v3(project_root):
    """Combine train_v2.json with RAG-aware context pairs to produce train_v3.json."""
    processed_dir = os.path.join(project_root, "data", "processed")
    rag_docs_dir = os.path.join(project_root, "data", "rag_docs")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(rag_docs_dir, exist_ok=True)

    # 1. Save raw RAG Documents
    for filename, content in ENTERPRISE_RAG_DOCS.items():
        doc_path = os.path.join(rag_docs_dir, filename)
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Saved RAG SOP: {doc_path}")

    # 2. Load train_v2.json if present, or fallback to train_v1.json
    v2_path = os.path.join(processed_dir, "train_v2.json")
    v1_path = os.path.join(processed_dir, "train_v1.json")
    base_data = []

    if os.path.exists(v2_path):
        with open(v2_path, "r", encoding="utf-8") as f:
            base_data = json.load(f)
        print(f"Loaded train_v2.json: {len(base_data):,} samples")
    elif os.path.exists(v1_path):
        with open(v1_path, "r", encoding="utf-8") as f:
            base_data = json.load(f)
        print(f"Loaded train_v1.json: {len(base_data):,} samples")
    else:
        print("ℹ️ Existing training files not found locally (will be combined on Colab).")

    # 3. Ensure ALL base samples have a valid context field
    for item in base_data:
        if not item.get("context"):
            item["context"] = "General SME Business Knowledge Base & Accounting Guidelines."

    # 4. Generate RAG-aware context samples
    rag_samples = generate_rag_aware_dataset(600)
    print(f"Generated {len(rag_samples):,} RAG-aware enterprise context samples.")

    # 5. Create train_v3.json
    train_v3 = base_data + rag_samples
    v3_path = os.path.join(processed_dir, "train_v3.json")
    with open(v3_path, "w", encoding="utf-8") as f:
        json.dump(train_v3, f, indent=2)
    print(f"✅ Saved train_v3.json: {v3_path} ({len(train_v3):,} total samples)")

    # 6. Update/save val_v3.json
    val_v1_path = os.path.join(processed_dir, "val_v1.json")
    val_v3_path = os.path.join(processed_dir, "val_v3.json")
    if os.path.exists(val_v1_path):
        with open(val_v1_path, "r", encoding="utf-8") as f:
            val_data = json.load(f)
        for item in val_data:
            if not item.get("context"):
                item["context"] = "General SME Business Operations & Financial Management Guidelines."
        with open(val_v3_path, "w", encoding="utf-8") as f:
            json.dump(val_data, f, indent=2)
        print(f"✅ Saved val_v3.json: {val_v3_path} ({len(val_data):,} samples)")

    return len(train_v3), len(rag_samples)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    build_train_v3(base_dir)
