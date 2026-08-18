"""
Retrieval Diagnostic (offline, KHÔNG cần LLM)
=============================================
RAGAS cần judge LLM. Khi không có credit API, script này vẫn đo được chất lượng
*tầng retrieval* — đủ để so sánh Naive baseline vs Production và tìm bottom-5.

Metric (proxy, KHÔNG phải RAGAS — dùng embeddings bge-m3 local):
  - gt_similarity  : max cosine(ground_truth, context_i)  → proxy cho context_recall
  - mean_similarity: mean cosine(ground_truth, context_i) → proxy cho context_precision
  - token_recall   : % content-word của ground_truth xuất hiện trong union(contexts)
  - source_hit     : top-1 context có đúng file mà ground_truth thuộc về không

Chạy: DISABLE_LLM=1 python retrieval_diagnostic.py
Kết quả: reports/retrieval_diagnostic.json
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from config import (COLLECTION_NAME, EMBEDDING_MODEL, NAIVE_COLLECTION,
                    RERANK_TOP_K, get_encoder)
from src.m1_chunking import chunk_basic, chunk_hierarchical, load_documents
from src.m2_search import DenseSearch, HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set
from src.m5_enrichment import enrich_chunks

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

# stopword tối thiểu cho tiếng Việt — bỏ để token_recall phản ánh content word
STOPWORDS = {
    "là", "và", "của", "cho", "các", "có", "được", "một", "những", "với", "trong",
    "khi", "nếu", "thì", "này", "đó", "về", "tại", "từ", "đến", "theo", "phải",
    "sẽ", "đã", "bị", "do", "mà", "nhưng", "hoặc", "cũng", "còn", "để",
}


def _tokens(text: str) -> set[str]:
    """Content word (bỏ stopword, giữ số vì số là thông tin chính trong quy định)."""
    raw = re.findall(r"[0-9]+(?:[.,][0-9]+)*|[^\W\d_]+", text.lower(), flags=re.UNICODE)
    return {t for t in raw if t not in STOPWORDS and len(t) > 1}


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _score_row(question: str, ground_truth: str, contexts: list[str],
               encoder, source_of_gt: str, top_source: str) -> dict:
    """Tính 4 proxy metric cho 1 câu hỏi."""
    if not contexts:
        return {"gt_similarity": 0.0, "mean_similarity": 0.0, "token_recall": 0.0,
                "source_hit": False, "n_contexts": 0}

    vectors = encoder.encode([ground_truth] + contexts, show_progress_bar=False)
    gt_vec, ctx_vecs = vectors[0], vectors[1:]
    sims = [_cos(gt_vec, v) for v in ctx_vecs]

    gt_tokens = _tokens(ground_truth)
    found = gt_tokens & _tokens(" ".join(contexts))

    return {
        "gt_similarity": round(max(sims), 4),
        "mean_similarity": round(sum(sims) / len(sims), 4),
        "token_recall": round(len(found) / len(gt_tokens), 4) if gt_tokens else 0.0,
        "missing_tokens": sorted(gt_tokens - found)[:12],
        "source_hit": bool(source_of_gt) and top_source == source_of_gt,
        "expected_source": source_of_gt,
        "top_source": top_source,
        "n_contexts": len(contexts),
    }


def _expected_sources(test_set: list[dict], docs: list[dict], encoder) -> list[str]:
    """Đoán file nguồn của mỗi ground_truth = doc có cosine cao nhất với ground_truth.

    Test set không ghi nhãn source → dùng để chẩn đoán "retrieve sai tài liệu"
    (VD lẫn nghi_phep_nam_v2023 với v2024, mat_khau_v1 với v2).
    """
    doc_vecs = encoder.encode([d["text"][:4000] for d in docs], show_progress_bar=False)
    gt_vecs = encoder.encode([t["ground_truth"] for t in test_set], show_progress_bar=False)

    expected = []
    for gt_vec in gt_vecs:
        sims = [_cos(gt_vec, dv) for dv in doc_vecs]
        expected.append(docs[int(np.argmax(sims))]["metadata"]["source"])
    return expected


def run_baseline(docs: list[dict], test_set: list[dict], encoder, expected: list[str]) -> dict:
    """Naive: chunk_basic + dense-only top-3 (giống naive_baseline.py)."""
    chunks = []
    for doc in docs:
        for c in chunk_basic(doc["text"], metadata=doc["metadata"]):
            chunks.append({"text": c.text, "metadata": c.metadata})

    search = DenseSearch()
    t0 = time.time()
    search.index(chunks, collection=NAIVE_COLLECTION)
    index_time = time.time() - t0

    rows = []
    for item, exp in zip(test_set, expected):
        results = search.search(item["question"], top_k=3, collection=NAIVE_COLLECTION)
        contexts = [r.text for r in results]
        top_source = results[0].metadata.get("source", "") if results else ""
        row = {"question": item["question"], "ground_truth": item["ground_truth"],
               "contexts": contexts}
        row.update(_score_row(item["question"], item["ground_truth"], contexts,
                              encoder, exp, top_source))
        rows.append(row)
        print(f"  [baseline {len(rows)}/{len(test_set)}] gt_sim={row['gt_similarity']:.3f} "
              f"token_recall={row['token_recall']:.2f}", flush=True)

    return {"config": "naive (chunk_basic + dense-only, top-3)",
            "n_chunks": len(chunks), "index_time_s": round(index_time, 1), "rows": rows}


def run_production(docs: list[dict], test_set: list[dict], encoder, expected: list[str]) -> dict:
    """Production: hierarchical child + enrichment + hybrid(BM25+dense+RRF) + rerank top-3."""
    chunks = []
    for doc in docs:
        _parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            chunks.append({"text": child.text,
                           "metadata": {**child.metadata, "parent_id": child.parent_id}})

    enriched = enrich_chunks(chunks)
    if enriched:
        chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]

    search = HybridSearch()
    t0 = time.time()
    search.index(chunks)
    index_time = time.time() - t0
    reranker = CrossEncoderReranker()

    rows = []
    for item, exp in zip(test_set, expected):
        hybrid = search.search(item["question"])
        docs_in = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in hybrid]
        reranked = reranker.rerank(item["question"], docs_in, top_k=RERANK_TOP_K)
        contexts = [r.text for r in reranked] if reranked else [r.text for r in hybrid[:3]]
        top_source = (reranked[0].metadata.get("source", "") if reranked
                      else (hybrid[0].metadata.get("source", "") if hybrid else ""))

        row = {"question": item["question"], "ground_truth": item["ground_truth"],
               "contexts": contexts, "n_hybrid_candidates": len(hybrid)}
        row.update(_score_row(item["question"], item["ground_truth"], contexts,
                              encoder, exp, top_source))
        rows.append(row)
        print(f"  [production {len(rows)}/{len(test_set)}] gt_sim={row['gt_similarity']:.3f} "
              f"token_recall={row['token_recall']:.2f}", flush=True)

    return {"config": f"production (hierarchical + enrich + hybrid RRF + rerank top-{RERANK_TOP_K})",
            "n_chunks": len(chunks), "index_time_s": round(index_time, 1), "rows": rows}


def _aggregate(rows: list[dict]) -> dict:
    n = len(rows) or 1
    return {
        "gt_similarity": round(sum(r["gt_similarity"] for r in rows) / n, 4),
        "mean_similarity": round(sum(r["mean_similarity"] for r in rows) / n, 4),
        "token_recall": round(sum(r["token_recall"] for r in rows) / n, 4),
        "source_hit_rate": round(sum(1 for r in rows if r["source_hit"]) / n, 4),
    }


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    print("=" * 60)
    print("RETRIEVAL DIAGNOSTIC (offline, no LLM judge)")
    print("=" * 60, flush=True)

    docs = load_documents()
    test_set = load_test_set()
    encoder = get_encoder(EMBEDDING_MODEL)
    print(f"  {len(docs)} documents · {len(test_set)} questions", flush=True)

    expected = _expected_sources(test_set, docs, encoder)

    print("\n[1/2] Naive baseline...", flush=True)
    baseline = run_baseline(docs, test_set, encoder, expected)
    print("\n[2/2] Production...", flush=True)
    production = run_production(docs, test_set, encoder, expected)

    baseline["aggregate"] = _aggregate(baseline["rows"])
    production["aggregate"] = _aggregate(production["rows"])

    # bottom-5 của production: xếp theo (token_recall, gt_similarity) tăng dần
    bottom = sorted(production["rows"],
                    key=lambda r: (r["token_recall"], r["gt_similarity"]))[:5]

    report = {
        "note": "Proxy metrics dựa trên embeddings bge-m3 local — KHÔNG phải RAGAS. "
                "Dùng khi không có credit LLM judge; xem reports/ragas_report.json cho RAGAS.",
        "num_questions": len(test_set),
        "baseline": {k: v for k, v in baseline.items() if k != "rows"},
        "production": {k: v for k, v in production.items() if k != "rows"},
        "delta": {k: round(production["aggregate"][k] - baseline["aggregate"][k], 4)
                  for k in production["aggregate"]},
        "bottom_5_production": [{k: v for k, v in r.items() if k != "contexts"} for r in bottom],
        "per_question": {"baseline": baseline["rows"], "production": production["rows"]},
    }

    path = os.path.join(REPORTS_DIR, "retrieval_diagnostic.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"{'Proxy metric':<20}{'Naive':>10}{'Production':>13}{'Δ':>10}")
    print("-" * 60)
    for k in baseline["aggregate"]:
        b, p = baseline["aggregate"][k], production["aggregate"][k]
        print(f"{k:<20}{b:>10.4f}{p:>13.4f}{p-b:>+10.4f}")
    print("=" * 60)
    print(f"Saved {path}")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Total: {time.time() - start:.1f}s")




