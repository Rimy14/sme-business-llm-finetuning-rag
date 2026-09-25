"""
SME ChromaDB Vector Store & Embedding Indexer (Day 9 - KAN-48)
Handles enterprise SOP document loading, semantic chunking, and ChromaDB vector indexing.
"""

import os
import re
import json
from typing import List, Dict, Any, Optional

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


ENTERPRISE_SOP_FALLBACKS = {
    "cash_flow_and_working_capital_sop.md": """# SME Standard Operating Procedure: Cash Flow & Working Capital Management
**Document ID:** SOP-FIN-001 | **Category:** Finance | **Version:** 3.2

## Section 1: Cash Conversion Cycle (CCC) Policy
- Target CCC: Maintain an operating Cash Conversion Cycle between 30 and 45 days.
- Days Sales Outstanding (DSO): Invoices must carry standard payment terms of Net 30. An early settlement discount of 2/10 Net 30 is authorized for clients with order volumes exceeding $10,000.
- Days Payable Outstanding (DPO): All supplier payments must be scheduled exactly on their due date (Net 45 or Net 60) via automated batch ACH to maximize working capital liquidity.
- Emergency Reserve Threshold: A mandatory liquid cash reserve equal to at least 90 days (3 months) of fixed operational expenditure (OpEx) must be maintained in an insured sweep account.

## Section 2: Short-Term Financing & Factoring Rules
- Invoices aged past 60 days without dispute resolution may be submitted for non-recourse invoice factoring at fee rates not exceeding 2.5% per 30-day tranche.
- Revolving Credit Lines (LOC) should only be tapped for seasonal inventory builds where gross margin exceeds the annualized borrowing cost by at least 15%.
""",

    "inventory_control_and_procurement_sop.md": """# SME Standard Operating Procedure: Inventory Control & Procurement
**Document ID:** SOP-OPS-004 | **Category:** Operations | **Version:** 2.8

## Section 1: Reorder Point (ROP) & Safety Stock
- Reorder Point Formula: ROP = (Daily Demand x Lead Time in Days) + Safety Stock.
- Safety Stock Buffer: For critical Class-A inventory items, safety stock must cover a minimum of 14 days of average demand. For Class-B and Class-C items, a 7-day safety buffer is mandated.
- Economic Order Quantity (EOQ): Purchase orders must balance holding cost (calculated at 18% annual inventory value) against fixed purchase order processing costs.

## Section 2: Dual Sourcing & Supplier Escalation
- When a sole vendor issues a price hike exceeding 5%, procurement must immediately initiate RFQs across qualified secondary suppliers.
- Minimum 25% of annual raw material volume must be allocated to an alternative domestic supplier to insulate against overseas freight bottlenecks.
""",

    "tax_compliance_and_payroll_guidelines.md": """# SME Standard Operating Procedure: Tax Compliance & Payroll Operations
**Document ID:** SOP-TAX-007 | **Category:** Tax & HR | **Version:** 4.1

## Section 1: Worker Classification Standards (W-2 vs 1099)
- Behavioral & Operational Control: If the SME dictates specific working hours, mandatory software tools, and direct supervision, the worker MUST be classified as a W-2 Employee.
- Independent Contractors (1099): Permitted only when the worker retains independence in project execution, provides their own equipment, and maintains an independent legal business entity (LLC/EIN).
- Penalties: Misclassification incurs retroactive employer payroll taxes (7.65% FICA), state unemployment insurance back-pay, and mandatory statutory interest penalties.

## Section 2: Section 179 Capital Deductions & Payroll Filings
- Qualifying capital equipment and business software purchased can be fully expensed in Year 1 under IRS Section 179 up to allowable federal caps.
- Quarterly 941 federal payroll returns must be reconciled and submitted within 30 days of quarter end.
""",

    "pricing_and_unit_economics_policy.md": """# SME Standard Operating Procedure: Pricing Strategy & Margin Control
**Document ID:** SOP-STRAT-002 | **Category:** Strategy | **Version:** 2.1

## Section 1: Margin & Discounting Governance
- Gross Margin Floor: No product line may be priced with a gross margin below 35% without explicit written CFO sign-off.
- Break-Even Formula: Break-Even Units = Fixed Operating Costs / (Selling Price - Variable Cost per Unit).
- Discount Authority: Sales representatives are authorized to offer maximum 10% volume discounts for orders exceeding 500 units. Discounts between 11% and 20% require Director approval. Any discount above 20% requires CFO authorization.

## Section 2: Unit Economics & CAC Ratios
- Commercial acquisition campaigns must achieve a Minimum Customer Lifetime Value to Customer Acquisition Cost (LTV:CAC) ratio of 3.0x over a 24-month horizon.
"""
}


def chunk_document(
    text: str,
    doc_name: str,
    chunk_size: int = 400,
    chunk_overlap: int = 60
) -> List[Dict[str, Any]]:
    """
    Split markdown text into semantic chunks with metadata.
    Splits along markdown headers ('## Section') and paragraphs first.
    """
    chunks = []
    # Extract Document ID and Category if present
    doc_id_match = re.search(r"\*\*Document ID:\*\*\s*([^\s\|]+)", text)
    doc_id = doc_id_match.group(1) if doc_id_match else doc_name.replace(".md", "").upper()

    cat_match = re.search(r"\*\*Category:\*\*\s*([^\s\|]+)", text)
    category = cat_match.group(1) if cat_match else "General SME"

    # Split by section headers
    sections = re.split(r"(?=##\s+)", text)

    chunk_idx = 0
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue

        # Extract section title
        sec_title_match = re.match(r"##\s+([^\n]+)", sec)
        sec_title = sec_title_match.group(1) if sec_title_match else "General Policy"

        # Split section into sub-paragraphs if longer than chunk_size
        paragraphs = [p.strip() for p in sec.split("\n\n") if p.strip()]
        curr_text = ""

        for p in paragraphs:
            if len(curr_text) + len(p) + 2 <= chunk_size:
                curr_text = f"{curr_text}\n\n{p}".strip() if curr_text else p
            else:
                if curr_text:
                    chunks.append({
                        "id": f"{doc_id}_chk_{chunk_idx}",
                        "text": curr_text,
                        "metadata": {
                            "document_id": doc_id,
                            "filename": doc_name,
                            "category": category,
                            "section_title": sec_title,
                            "chunk_index": chunk_idx,
                            "char_count": len(curr_text)
                        }
                    })
                    chunk_idx += 1
                curr_text = p

        if curr_text:
            chunks.append({
                "id": f"{doc_id}_chk_{chunk_idx}",
                "text": curr_text,
                "metadata": {
                    "document_id": doc_id,
                    "filename": doc_name,
                    "category": category,
                    "section_title": sec_title,
                    "chunk_index": chunk_idx,
                    "char_count": len(curr_text)
                }
            })
            chunk_idx += 1

    return chunks


class SMEVectorStore:
    """ChromaDB Vector Store for SME SOP Knowledge Retrieval."""

    def __init__(
        self,
        persist_dir: str = "data/chroma_db",
        collection_name: str = "sme_sop_knowledge_base",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu"
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.device = device
        os.makedirs(self.persist_dir, exist_ok=True)

        print(f"📦 Initializing ChromaDB vector store at: {self.persist_dir}")
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        print(f"🤖 Loading Embedding Model: {embedding_model_name} (on {device})...")
        self.embedder = SentenceTransformer(embedding_model_name, device=device)
        print("✅ Vector Store & Embedder ready.")

    def index_documents_from_dir(
        self,
        docs_dir: str,
        chunk_size: int = 400,
        chunk_overlap: int = 60
    ) -> int:
        """Load, chunk, embed, and index all markdown documents in docs_dir."""
        os.makedirs(docs_dir, exist_ok=True)

        # Populate fallback docs if empty
        existing_files = [f for f in os.listdir(docs_dir) if f.endswith(".md")]
        if not existing_files:
            print("ℹ️ docs_dir is empty. Writing enterprise SOP templates...")
            for fname, content in ENTERPRISE_SOP_FALLBACKS.items():
                with open(os.path.join(docs_dir, fname), "w", encoding="utf-8") as f:
                    f.write(content)
            existing_files = list(ENTERPRISE_SOP_FALLBACKS.keys())

        all_chunks = []
        for fname in existing_files:
            fpath = os.path.join(docs_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            doc_chunks = chunk_document(content, fname, chunk_size, chunk_overlap)
            all_chunks.extend(doc_chunks)

        print(f"📄 Created {len(all_chunks)} semantic chunks across {len(existing_files)} SOP documents.")

        # Compute embeddings
        texts = [chk["text"] for chk in all_chunks]
        ids   = [chk["id"] for chk in all_chunks]
        metas = [chk["metadata"] for chk in all_chunks]

        embeddings = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

        # Upsert into ChromaDB
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metas
        )
        print(f"✅ Successfully indexed {len(all_chunks)} chunks into ChromaDB collection '{self.collection_name}'.")
        return len(all_chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve top_k relevant SOP passages for a user query."""
        query_embedding = self.embedder.encode([query], normalize_embeddings=True).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        if results and results.get("documents") and len(results["documents"][0]) > 0:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]

            for doc, meta, dist in zip(docs, metas, dists):
                # Cosine distance to similarity: similarity = 1 - cosine_distance
                sim = round(1.0 - float(dist), 4)
                if score_threshold is not None and sim < score_threshold:
                    continue
                retrieved.append({
                    "text": doc,
                    "metadata": meta,
                    "similarity": sim
                })

        return retrieved

    def format_retrieved_context(self, retrieved: List[Dict[str, Any]]) -> str:
        """Format retrieved passages into a structured context string for LLM prompting."""
        if not retrieved:
            return ""

        context_parts = []
        for i, item in enumerate(retrieved, 1):
            doc_id = item["metadata"].get("document_id", "SOP")
            sec = item["metadata"].get("section_title", "Policy")
            text = item["text"]
            context_parts.append(f"[{i}] [{doc_id} — {sec}]:\n{text}")

        return "\n\n".join(context_parts)
