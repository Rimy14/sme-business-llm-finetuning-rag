"""
Automated 200-Query RAG Evaluation Engine (Day 9 - KAN-48)
Benchmarks fine-tuned SME models with vs without ChromaDB retrieval.
Calculates ROUGE-1/2/L, BLEU-4, BERTScore, Retrieval Accuracy, and Latency.
"""

import os
import gc
import json
import time
from typing import List, Dict, Any
from tqdm import tqdm
import torch

from rouge_score import rouge_scorer
import sacrebleu
from bert_score import score as bert_score_fn

from .vector_store import SMEVectorStore
from .rag_pipeline import SMERAGPipeline


def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate ROUGE-1, ROUGE-2, ROUGE-L, BLEU-4, and BERTScore."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1 = r2 = rl = 0.0

    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        r1 += s["rouge1"].fmeasure
        r2 += s["rouge2"].fmeasure
        rl += s["rougeL"].fmeasure

    n = len(predictions) if len(predictions) > 0 else 1
    bleu = sacrebleu.corpus_bleu(predictions, [references]).score

    # BERTScore on CPU to guarantee 0 OOM errors
    _, _, F = bert_score_fn(
        predictions,
        references,
        lang="en",
        model_type="distilbert-base-uncased",
        device="cpu",
        verbose=False
    )

    return {
        "rouge1": round(r1 / n * 100, 2),
        "rouge2": round(r2 / n * 100, 2),
        "rougeL": round(rl / n * 100, 2),
        "bleu4": round(bleu, 2),
        "bertscore": round(F.mean().item() * 100, 2)
    }


def evaluate_rag_pipeline(
    pipeline: SMERAGPipeline,
    test_samples: List[Dict[str, Any]],
    n_queries: int = 200,
    model_label: str = "qwen_sme_v3"
) -> Dict[str, Any]:
    """
    Run full 200-query benchmark comparing:
    - Mode A: Parametric Only (No RAG)
    - Mode B: Vector Augmented (With ChromaDB RAG)
    """
    eval_set = test_samples[:n_queries]
    queries = [ex.get("instruction") or ex.get("question") for ex in eval_set]
    references = [ex.get("response") or ex.get("answer") for ex in eval_set]

    print(f"\n" + "="*70)
    print(f"📊 Running 200-Query Benchmark for [{model_label}] (With vs Without RAG)")
    print("="*70)

    # ── 1. Benchmark: WITHOUT RAG ─────────────────────────────────────────────
    print(f"\n[1/2] 🧪 Evaluating WITHOUT RAG (Parametric Only) on {len(eval_set)} queries...")
    no_rag_preds = []
    no_rag_times = []
    for q in tqdm(queries, desc="Evaluating Without RAG"):
        res = pipeline.generate(q, use_rag=False, max_new_tokens=200)
        no_rag_preds.append(res["answer"])
        no_rag_times.append(res["latency_ms"])

    no_rag_scores = compute_metrics(no_rag_preds, references)
    no_rag_scores["avg_latency_ms"] = round(sum(no_rag_times) / len(no_rag_times), 2)

    # ── 2. Benchmark: WITH RAG (ChromaDB Retrieval) ───────────────────────────
    print(f"\n[2/2] 🚀 Evaluating WITH RAG (ChromaDB SOP Augmented) on {len(eval_set)} queries...")
    rag_preds = []
    rag_times = []
    retrieval_hits = 0

    for ex in tqdm(eval_set, desc="Evaluating With RAG"):
        q = ex.get("instruction") or ex.get("question")
        res = pipeline.generate(q, use_rag=True, top_k=3, max_new_tokens=200)
        rag_preds.append(res["answer"])
        rag_times.append(res["latency_ms"])

        # Check retrieval precision if expected context is present in test data
        if res["retrieved_sources"] and res["top_similarity"] >= 0.30:
            retrieval_hits += 1

    rag_scores = compute_metrics(rag_preds, references)
    rag_scores["avg_latency_ms"] = round(sum(rag_times) / len(rag_times), 2)
    retrieval_accuracy = round((retrieval_hits / len(eval_set)) * 100, 2)
    rag_scores["retrieval_hit_rate"] = retrieval_accuracy

    # ── 3. Calculate Lift ─────────────────────────────────────────────────────
    lift = {
        "rouge1_lift": round(rag_scores["rouge1"] - no_rag_scores["rouge1"], 2),
        "rouge2_lift": round(rag_scores["rouge2"] - no_rag_scores["rouge2"], 2),
        "rougeL_lift": round(rag_scores["rougeL"] - no_rag_scores["rougeL"], 2),
        "bleu4_lift": round(rag_scores["bleu4"] - no_rag_scores["bleu4"], 2),
        "bertscore_lift": round(rag_scores["bertscore"] - no_rag_scores["bertscore"], 2),
    }

    report = {
        "model_label": model_label,
        "n_queries_evaluated": len(eval_set),
        "no_rag_scores": no_rag_scores,
        "rag_scores": rag_scores,
        "lift": lift,
        "sample_comparisons": [
            {
                "query": queries[i],
                "reference": references[i],
                "no_rag_pred": no_rag_preds[i],
                "rag_pred": rag_preds[i]
            }
            for i in range(min(5, len(queries)))
        ]
    }

    return report
