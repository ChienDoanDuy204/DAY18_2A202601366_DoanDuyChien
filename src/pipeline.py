from __future__ import annotations

"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4."""

import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K

# --- Latency instrumentation (bonus: latency breakdown report) ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")

BUILD_TIMINGS: dict[str, float] = {}   # stage name -> seconds (one-off indexing cost)
QUERY_TIMINGS: list[dict] = []          # per-query {search, rerank, llm} seconds
STATS: dict[str, int] = {}              # counts (docs, chunks, ...)


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    BUILD_TIMINGS["M1 chunking (hierarchical)"] = time.time() - t0
    STATS["documents"] = len(docs)
    STATS["chunks"] = len(all_chunks)
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)
    BUILD_TIMINGS["M5 enrichment (1 LLM call/chunk)"] = time.time() - t0

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    BUILD_TIMINGS["M2 indexing (BM25 + bge-m3 → Qdrant)"] = time.time() - t0
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    BUILD_TIMINGS["M3 reranker model load"] = time.time() - t0
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    t0 = time.time()
    results = search.search(query)
    t_search = time.time() - t0

    t0 = time.time()
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]
    t_rerank = time.time() - t0

    t0 = time.time()
    from config import HAS_LLM_KEY, get_llm_client, LLM_MODEL, LLM_EXTRA_BODY
    if HAS_LLM_KEY and contexts:
        try:
            client = get_llm_client()
            context_str = "\n\n".join(contexts)
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'"},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
                ],
                **({"extra_body": LLM_EXTRA_BODY} if LLM_EXTRA_BODY else {}),
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    t_llm = time.time() - t0

    QUERY_TIMINGS.append({"search": t_search, "rerank": t_rerank, "llm": t_llm})
    return answer, contexts


def save_latency_report(path: str = None) -> str:
    """Ghi bảng latency breakdown (bonus) → reports/latency_report.md."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = path or os.path.join(REPORTS_DIR, "latency_report.md")

    n = len(QUERY_TIMINGS) or 1

    def _avg(key: str) -> float:
        return sum(q[key] for q in QUERY_TIMINGS) / n

    def _p95(key: str) -> float:
        values = sorted(q[key] for q in QUERY_TIMINGS)
        return values[min(int(0.95 * len(values)), len(values) - 1)] if values else 0.0

    build_total = sum(v for k, v in BUILD_TIMINGS.items() if not k.startswith("_"))
    per_query_avg = _avg("search") + _avg("rerank") + _avg("llm")

    lines = [
        "# Latency Breakdown — Production RAG Pipeline",
        "",
        f"Corpus: {STATS.get('documents', 0)} documents → {STATS.get('chunks', 0)} child chunks · "
        f"{len(QUERY_TIMINGS)} queries · rerank top-{RERANK_TOP_K}",
        "",
        "## A. Build (one-off, offline)",
        "",
        "| Stage | Time (s) | % of build |",
        "|-------|---------:|-----------:|",
    ]
    for stage, seconds in BUILD_TIMINGS.items():
        if stage.startswith("_"):
            continue
        share = seconds / build_total * 100 if build_total else 0.0
        lines.append(f"| {stage} | {seconds:.1f} | {share:.1f}% |")
    lines += [
        f"| **Total build** | **{build_total:.1f}** | 100% |",
        "",
        "## B. Per-query (online)",
        "",
        "| Stage | Avg (ms) | p95 (ms) | % of query |",
        "|-------|---------:|---------:|-----------:|",
    ]
    for label, key in [("M2 hybrid search (BM25 + dense + RRF)", "search"),
                       ("M3 cross-encoder rerank (20 → 3)", "rerank"),
                       ("LLM answer generation", "llm")]:
        share = _avg(key) / per_query_avg * 100 if per_query_avg else 0.0
        lines.append(f"| {label} | {_avg(key)*1000:.0f} | {_p95(key)*1000:.0f} | {share:.1f}% |")
    lines += [
        f"| **Total per query** | **{per_query_avg*1000:.0f}** | — | 100% |",
        "",
        "## C. Evaluation",
        "",
        "| Stage | Time (s) |",
        "|-------|---------:|",
        f"| RAGAS (4 metrics × {len(QUERY_TIMINGS)} questions) | {BUILD_TIMINGS.get('_ragas', 0.0):.1f} |",
        "",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Latency report saved to {path}")
    return path


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    BUILD_TIMINGS["_ragas"] = time.time() - t0
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    save_latency_report()
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
