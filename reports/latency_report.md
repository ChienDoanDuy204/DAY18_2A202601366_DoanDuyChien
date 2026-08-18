# Latency Breakdown — Production RAG Pipeline

Corpus: 26 documents → 112 child chunks · 20 queries · rerank top-3

## A. Build (one-off, offline)

| Stage | Time (s) | % of build |
|-------|---------:|-----------:|
| M1 chunking (hierarchical) | 0.4 | 0.5% |
| M5 enrichment (1 LLM call/chunk) | 0.0 | 0.0% |
| M2 indexing (BM25 + bge-m3 → Qdrant) | 74.2 | 99.5% |
| M3 reranker model load | 0.0 | 0.0% |
| **Total build** | **74.6** | 100% |

## B. Per-query (online)

| Stage | Avg (ms) | p95 (ms) | % of query |
|-------|---------:|---------:|-----------:|
| M2 hybrid search (BM25 + dense + RRF) | 398 | 770 | 4.1% |
| M3 cross-encoder rerank (20 → 3) | 9391 | 16792 | 95.9% |
| LLM answer generation | 0 | 0 | 0.0% |
| **Total per query** | **9789** | — | 100% |

## C. Evaluation

| Stage | Time (s) |
|-------|---------:|
| RAGAS (4 metrics × 20 questions) | 0.0 |
