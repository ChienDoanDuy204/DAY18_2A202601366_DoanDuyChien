from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


# cache model theo tên → tránh load lại cross-encoder (~2GB) nhiều lần
_MODEL_CACHE: dict[str, object] = {}


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            # ⚠️ Dùng sentence_transformers.CrossEncoder, KHÔNG dùng FlagEmbedding.
            # FlagReranker crash với transformers>=5.0 (XLMRobertaTokenizer lỗi).
            if self.model_name not in _MODEL_CACHE:
                from sentence_transformers import CrossEncoder

                _MODEL_CACHE[self.model_name] = CrossEncoder(self.model_name)
            self._model = _MODEL_CACHE[self.model_name]
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []

        try:
            model = self._load_model()
            pairs = [(query, doc["text"]) for doc in documents]
            scores = model.predict(pairs)
        except Exception as e:
            print(f"  ⚠️  Rerank failed ({e}) — giữ nguyên thứ tự retrieval")
            scores = [doc.get("score", 0.0) for doc in documents]

        if isinstance(scores, (int, float)):
            scores = [scores]

        # sort theo cross-encoder score giảm dần (key chỉ lấy score → tránh so sánh dict)
        scored = sorted(zip(scores, documents), key=lambda pair: float(pair[0]), reverse=True)

        return [
            RerankResult(
                text=doc["text"],
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i + 1,
            )
            for i, (score, doc) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is None:
            from flashrank import Ranker

            self._model = Ranker()
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents:
            return []

        try:
            from flashrank import RerankRequest

            model = self._load_model()
            passages = [{"id": i, "text": doc["text"]} for i, doc in enumerate(documents)]
            ranked = model.rerank(RerankRequest(query=query, passages=passages))
        except Exception as e:
            print(f"  ⚠️  Flashrank unavailable ({e})")
            return []

        results = []
        for i, item in enumerate(ranked[:top_k]):
            doc = documents[item["id"]]
            results.append(RerankResult(
                text=doc["text"],
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(item["score"]),
                metadata=doc.get("metadata", {}),
                rank=i + 1,
            ))
        return results


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
