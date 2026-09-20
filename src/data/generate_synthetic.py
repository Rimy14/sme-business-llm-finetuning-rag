"""
Synthetic Data Generator for SME Daily Business (Day 5/6 - KAN-31, KAN-35)
Generates high-quality synthetic domain Q&A pairs across key SME operational verticals:
1. Cash Flow & Working Capital
2. Inventory & Supply Chain
3. Tax, Compliance & Auditing
4. Invoicing, Credit Terms & Factoring
5. HR, Payroll & Employee Retention
6. Sales, Pricing Strategy & Unit Economics
"""

import os
import json
import random

SME_SYNTHETIC_TEMPLATES = [
    # Cash Flow & Working Capital
    {
        "category": "cash_flow",
        "templates": [
            ("How can a small retail business with ${revenue:,} in monthly sales improve its cash conversion cycle when supplier payment terms are {pay_terms} days and receivables take {rec_terms} days?",
             "To improve the cash conversion cycle (CCC):\n1. **Shorten Receivables ({rec_terms} days → 30 days):** Introduce early-payment discounts (e.g., 2/10 net 30), mandate upfront deposits for bulk orders, and automate digital invoicing.\n2. **Negotiate Payables ({pay_terms} days → 60 days):** Request extended credit terms with key suppliers based on order consistency.\n3. **Optimize Inventory Velocity:** Implement just-in-time restocking on fast-moving SKUs to prevent capital lockup in unsold inventory.\nBy narrowing the gap between receivables and payables, the business frees up immediate working capital without taking on high-interest short-term debt."),
            ("What is the ideal emergency cash buffer for a service-based SME with fixed monthly operational expenses of ${opex:,}?",
             "For a service-based SME with monthly OpEx of ${opex:,}, the recommended cash buffer is **3 to 6 months of operating expenses**:\n- **Minimum Buffer (3 months):** ${buf_3:,}\n- **Optimal Buffer (6 months):** ${buf_6:,}\n\n**Action Plan:**\n1. Maintain these reserves in high-yield, liquid business savings accounts.\n2. Establish an undrawn revolving Line of Credit (LOC) to handle seasonal cash flow dips without dipping into payroll.")
        ]
    },
    # Inventory & Supply Chain
    {
        "category": "inventory_management",
        "templates": [
            ("How should an SME calculate the Reorder Point (ROP) for an item with daily demand of {daily_demand} units, a supplier lead time of {lead_time} days, and desired safety stock of {safety_stock} units?",
             "The Reorder Point (ROP) is calculated as:\n\n$$\\text{{ROP}} = (\\text{{Daily Demand}} \\times \\text{{Lead Time in Days}}) + \\text{{Safety Stock}}$$\n\n**Calculation:**\n- Lead time demand = {daily_demand} × {lead_time} = {lead_demand} units\n- Safety stock = {safety_stock} units\n- **ROP = {lead_demand} + {safety_stock} = {rop} units**\n\nWhen stock drops to **{rop} units**, generate a purchase order immediately to avoid stockouts while minimizing holding costs."),
            ("What strategies can an SME manufacturer use when a primary raw material supplier raises prices by {price_hike}%?",
             "When a supplier increases prices by {price_hike}%:\n1. **Dual Sourcing:** Qualify a secondary vendor to stimulate pricing competition.\n2. **Volume Tiering & Long-Term Contracts:** Lock in 6-12 month fixed price commitments in exchange for guaranteed volume.\n3. **Value Engineering:** Audit product bill-of-materials (BOM) to eliminate non-essential specifications.\n4. **Selective Price Pass-Through:** Adjust end-customer pricing selectively on low price-elasticity products while absorbing increases on highly competitive SKUs.")
        ]
    },
    # Tax, Compliance & Payroll
    {
        "category": "compliance_payroll",
        "templates": [
            ("What are the key compliance requirements an SME must follow when classifying workers as Independent Contractors (1099) vs Employees (W-2)?",
             "Worker classification hinges on three primary IRS common law rules:\n1. **Behavioral Control:** Does the business control what the worker does and how they do it? (If yes → W-2 Employee).\n2. **Financial Control:** Does the worker have unreimbursed business expenses, significant investment in tools, or freedom to work for multiple clients? (If yes → 1099 Contractor).\n3. **Relationship Type:** Are there written contracts with employee benefits like paid leave or insurance? (If yes → W-2 Employee).\n\n*Risk Warning:* Misclassifying workers can trigger severe penalties, back payroll taxes, and interest liabilities."),
            ("How should an SME with {num_emp} staff members structure an automated monthly payroll process to ensure zero tax penalties?",
             "To ensure zero payroll penalties:\n1. **Automate with Certified Payroll Software:** Use automated platforms to handle federal, state, and local tax withholdings automatically.\n2. **Direct Deposit & Paystub Delivery:** Schedule payroll 3 business days prior to disbursement date to account for ACH clearing.\n3. **Quarterly Tax Filing & Reconciliation:** Schedule automated 941 quarterly payroll tax filings and reconcile year-end W-2/W-3 forms.\n4. **Audit Overtime Calculations:** Enforce digital time tracking to maintain complete Fair Labor Standards Act (FLSA) compliance.")
        ]
    },
    # Pricing, Margins & Unit Economics
    {
        "category": "unit_economics",
        "templates": [
            ("A boutique manufacturing SME produces goods with a variable cost of ${var_cost} per unit and total fixed monthly costs of ${fixed_cost:,}. If the target selling price is ${price}, what is the monthly break-even unit volume?",
             "Break-even volume calculation:\n\n$$\\text{{Contribution Margin per Unit}} = \\text{{Selling Price}} - \\text{{Variable Cost}} = \\${price} - \\${var_cost} = \\${margin}$$\n$$\\text{{Break-Even Volume}} = \\frac{{\\text{{Fixed Costs}}}}{{\\text{{Contribution Margin}}}} = \\frac{{\\${fixed_cost:,}}}{{\\${margin}}} = {be_units:,.0f} \\text{{ units}}$$\n\n**Insight:** The business must sell at least **{be_units:,.0f} units** each month to cover overhead costs before generating operating profit."),
            ("How should an SME business evaluate whether to accept a client demanding a {discount}% volume discount on an order of {order_size} units at regular price ${price}?",
             "Evaluate using marginal contribution analysis:\n1. **Discounted Price:** ${disc_price:.2f} per unit (down from ${price}).\n2. **Total Revenue:** ${tot_rev:,.2f}.\n3. **Variable Cost Coverage:** Ensure the discounted price comfortably exceeds direct unit material & labor costs.\n4. **Capacity Utilization:** Only accept if production line has excess capacity that would otherwise go unutilized without displacing higher-margin standard orders.")
        ]
    }
]


def generate_synthetic_dataset(num_samples=550):
    """Generate diverse, realistic synthetic SME daily business Q&A pairs."""
    generated = []
    random.seed(42)

    categories = [
        "Cash Flow Management",
        "Working Capital & Financing",
        "Supply Chain & Procurement",
        "Inventory Optimization",
        "Payroll & HR Compliance",
        "Business Tax & Deductions",
        "Pricing & Unit Economics",
        "Customer Retention & Invoicing",
        "Vendor Contract Negotiation",
        "SME Digital Transformation"
    ]

    for i in range(num_samples):
        cat = random.choice(categories)
        revenue = random.randint(25, 300) * 1000
        pay_terms = random.choice([15, 30, 45])
        rec_terms = random.choice([45, 60, 90])
        opex = random.randint(10, 80) * 1000
        daily_demand = random.randint(10, 150)
        lead_time = random.randint(5, 30)
        safety_stock = random.randint(20, 100)
        price_hike = random.choice([8, 12, 15, 20, 25])
        num_emp = random.randint(5, 50)
        var_cost = round(random.uniform(15.0, 85.0), 2)
        price = round(var_cost * random.uniform(1.6, 2.8), 2)
        margin = round(price - var_cost, 2)
        fixed_cost = random.randint(15, 120) * 1000
        be_units = fixed_cost / margin if margin > 0 else 1000
        discount = random.choice([10, 15, 20, 25])
        order_size = random.randint(200, 2000)
        disc_price = price * (1 - discount / 100.0)
        tot_rev = disc_price * order_size
        lead_demand = daily_demand * lead_time
        rop = lead_demand + safety_stock

        context_val = {
            "revenue": revenue,
            "pay_terms": pay_terms,
            "rec_terms": rec_terms,
            "opex": opex,
            "buf_3": opex * 3,
            "buf_6": opex * 6,
            "daily_demand": daily_demand,
            "lead_time": lead_time,
            "safety_stock": safety_stock,
            "lead_demand": lead_demand,
            "rop": rop,
            "price_hike": price_hike,
            "num_emp": num_emp,
            "var_cost": var_cost,
            "price": price,
            "margin": margin,
            "fixed_cost": fixed_cost,
            "be_units": be_units,
            "discount": discount,
            "order_size": order_size,
            "disc_price": disc_price,
            "tot_rev": tot_rev
        }

        # Pick template
        group = random.choice(SME_SYNTHETIC_TEMPLATES)
        q_tmpl, a_tmpl = random.choice(group["templates"])

        question = q_tmpl.format(**context_val)
        answer = a_tmpl.format(**context_val)

        generated.append({
            "instruction": question,
            "context": f"SME Business Scenario: Category - {cat}",
            "response": answer,
            "category": cat
        })

    return generated


def build_30_benchmark_questions():
    """Build the standard 30-question SME evaluation benchmark for Day 6 self-testing."""
    questions = [
        # Finance & Cash Flow (1-5)
        {"id": 1, "domain": "Finance", "question": "What is the formula for Working Capital and why is it critical for an SME?"},
        {"id": 2, "domain": "Finance", "question": "How does invoice factoring differ from a traditional bank line of credit?"},
        {"id": 3, "domain": "Finance", "question": "What is the Cash Conversion Cycle (CCC) and how can an SME reduce it?"},
        {"id": 4, "domain": "Finance", "question": "How should an SME calculate its debt-service coverage ratio (DSCR) before applying for a loan?"},
        {"id": 5, "domain": "Finance", "question": "What is the difference between cash-basis accounting and accrual accounting for small businesses?"},

        # Operations & Supply Chain (6-10)
        {"id": 6, "domain": "Operations", "question": "How do you calculate Economic Order Quantity (EOQ) for inventory management?"},
        {"id": 7, "domain": "Operations", "question": "What is a Reorder Point (ROP) and how is safety stock factored in?"},
        {"id": 8, "domain": "Operations", "question": "How can an SME manage supply chain risk when relying on a single overseas vendor?"},
        {"id": 9, "domain": "Operations", "question": "What are the standard operating procedures (SOPs) for warehouse receiving and inspection?"},
        {"id": 10, "domain": "Operations", "question": "How can an SME reduce inventory carrying costs without risking stockouts?"},

        # Compliance, Tax & Payroll (11-15)
        {"id": 11, "domain": "Compliance", "question": "What is the IRS common-law standard for distinguishing 1099 contractors from W-2 employees?"},
        {"id": 12, "domain": "Compliance", "question": "What are allowable business expense deductions under Section 179 for equipment purchases?"},
        {"id": 13, "domain": "Compliance", "question": "How often must an employer deposit federal payroll taxes (Form 941)?"},
        {"id": 14, "domain": "Compliance", "question": "What are the mandatory record retention periods for SME accounting and tax records?"},
        {"id": 15, "domain": "Compliance", "question": "What steps must an SME take to maintain compliance with sales tax nexus across multiple states?"},

        # Sales, Pricing & Strategy (16-20)
        {"id": 16, "domain": "Strategy", "question": "How do you compute the Break-Even Point in both units and revenue dollars?"},
        {"id": 17, "domain": "Strategy", "question": "What is value-based pricing and how does it compare to cost-plus pricing for an SME?"},
        {"id": 18, "domain": "Strategy", "question": "How should an SME calculate Customer Acquisition Cost (CAC) and Customer Lifetime Value (LTV)?"},
        {"id": 19, "domain": "Strategy", "question": "What strategies can an SME use to handle a customer requesting a 20% discount on standard pricing?"},
        {"id": 20, "domain": "Strategy", "question": "How can an SME calculate its Gross Margin versus Net Operating Margin?"},

        # HR & Team Management (21-25)
        {"id": 21, "domain": "HR", "question": "What non-monetary incentives can an SME offer to improve key employee retention?"},
        {"id": 22, "domain": "HR", "question": "How should an SME handle non-exempt employee overtime tracking under the Fair Labor Standards Act (FLSA)?"},
        {"id": 23, "domain": "HR", "question": "What is the recommended onboarding checklist for a new small business hire during their first 30 days?"},
        {"id": 24, "domain": "HR", "question": "How should an SME conduct a formal performance improvement plan (PIP) for an underperforming employee?"},
        {"id": 25, "domain": "HR", "question": "What are the essential policies that must be included in an SME Employee Handbook?"},

        # Crisis Management & Digital Modernization (26-30)
        {"id": 26, "domain": "Modernization", "question": "What cybersecurity best practices should a 20-person SME implement on a limited budget?"},
        {"id": 27, "domain": "Modernization", "question": "How can an SME choose between off-the-shelf ERP software versus custom business automation tools?"},
        {"id": 28, "domain": "Crisis", "question": "What immediate cash-preservation steps should an SME take during an unexpected 40% revenue downturn?"},
        {"id": 29, "domain": "Crisis", "question": "How should an SME resolve a contract dispute with a critical supplier without immediately going to litigation?"},
        {"id": 30, "domain": "Modernization", "question": "How can an SME migrate from paper-based invoicing to automated electronic payment workflows?"}
    ]
    return questions


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_dir = os.path.join(base_dir, "data", "processed")
    synthetic_dir = os.path.join(base_dir, "data", "synthetic")
    eval_dir = os.path.join(base_dir, "evaluation", "day6")

    os.makedirs(synthetic_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    print("Generating 550 synthetic SME Q&A pairs...")
    synthetic_data = generate_synthetic_dataset(550)
    synth_path = os.path.join(synthetic_dir, "sme_synthetic_qa_500.json")
    with open(synth_path, "w", encoding="utf-8") as f:
        json.dump(synthetic_data, f, indent=2)
    print(f"✅ Saved synthetic dataset: {synth_path} ({len(synthetic_data)} samples)")

    # Combine with train_v1.json if it exists
    train_v1_path = os.path.join(processed_dir, "train_v1.json")
    train_v2_path = os.path.join(processed_dir, "train_v2.json")

    base_samples = []
    if os.path.exists(train_v1_path):
        with open(train_v1_path, "r", encoding="utf-8") as f:
            base_samples = json.load(f)
        print(f"Loaded train_v1.json: {len(base_samples):,} samples")
    else:
        print("ℹ️ train_v1.json not found locally (will be generated/combined on Colab).")

    combined_v2 = base_samples + synthetic_data
    with open(train_v2_path, "w", encoding="utf-8") as f:
        json.dump(combined_v2, f, indent=2)
    print(f"✅ Saved train_v2.json: {train_v2_path} ({len(combined_v2):,} samples)")

    # 30 Benchmark Questions
    bench_30 = build_30_benchmark_questions()
    bench_path = os.path.join(eval_dir, "benchmark_30_questions.json")
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(bench_30, f, indent=2)
    print(f"✅ Saved 30-Question Benchmark: {bench_path} ({len(bench_30)} questions)")


if __name__ == "__main__":
    main()
