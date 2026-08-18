from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class _SharedEncoderEmbeddings:
    """LangChain-compatible embeddings dùng lại encoder bge-m3 đã cache trong config.

    Cần vì OpenRouter không có endpoint /embeddings (answer_relevancy của RAGAS
    bắt buộc phải có embeddings) — và dùng chung instance để không load model 2 lần.
    """

    def __init__(self, model_name: str):
        from config import get_encoder

        self._encoder = get_encoder(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._encoder.encode(texts, show_progress_bar=False)]

    def embed_query(self, text: str) -> list[float]:
        return self._encoder.encode(text, show_progress_bar=False).tolist()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


def _judge_backends():
    """Tạo (llm, embeddings) cho RAGAS.

    ⚠️ OpenRouter support: RAGAS đọc OPENAI_API_KEY / OPENAI_BASE_URL từ env,
    nên ta set từ config. Nhưng OpenRouter KHÔNG có endpoint /embeddings và model id
    phải ở dạng "openai/gpt-4o-mini" → phải truyền llm + embeddings tường minh:
      - llm: ChatOpenAI trỏ base_url OpenRouter, model = LLM_MODEL
      - embeddings: bge-m3 chạy local thay cho OpenAI embeddings
    """
    import os as _os

    from config import (USE_OPENROUTER, LLM_API_KEY, LLM_MODEL,
                        OPENROUTER_BASE_URL, EMBEDDING_MODEL, LLM_EXTRA_BODY)

    if USE_OPENROUTER:
        _os.environ["OPENAI_API_KEY"] = LLM_API_KEY
        _os.environ["OPENAI_BASE_URL"] = OPENROUTER_BASE_URL
    elif LLM_API_KEY:
        _os.environ["OPENAI_API_KEY"] = LLM_API_KEY

    if not USE_OPENROUTER:
        return None, None  # OpenAI thuần → dùng default của RAGAS

    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    llm = LangchainLLMWrapper(ChatOpenAI(
        model=LLM_MODEL, api_key=LLM_API_KEY,
        base_url=OPENROUTER_BASE_URL, temperature=0.0,
        extra_body=LLM_EXTRA_BODY or None,  # tắt reasoning với model :free
    ))
    embeddings = LangchainEmbeddingsWrapper(_SharedEncoderEmbeddings(EMBEDDING_MODEL))
    return llm, embeddings


def _clean(value) -> float:
    """NaN / None (metric fail) → 0.0."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if value != value else value  # NaN check


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    zeros = {"faithfulness": 0.0, "answer_relevancy": 0.0,
             "context_precision": 0.0, "context_recall": 0.0, "per_question": []}

    from config import HAS_LLM_KEY

    if not HAS_LLM_KEY:
        # RAGAS bắt buộc cần judge LLM → không có key (hoặc DISABLE_LLM=1) thì bỏ qua
        # thay vì để mọi metric timeout/retry rồi trả NaN.
        print("  ⚠️  Không có LLM key (hoặc DISABLE_LLM=1) → bỏ qua RAGAS, scores = 0.0")
        return zeros

    try:
        llm, embeddings = _judge_backends()

        from ragas import evaluate
        from ragas.metrics import (faithfulness, answer_relevancy,
                                   context_precision, context_recall)
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })

        kwargs = {}
        if llm is not None:
            kwargs["llm"] = llm
        if embeddings is not None:
            kwargs["embeddings"] = embeddings

        # Giới hạn concurrency + retry: judge LLM có rate limit (429) → tránh NaN hàng loạt
        try:
            from ragas.run_config import RunConfig
            from config import RAGAS_MAX_WORKERS, RAGAS_MAX_RETRIES, RAGAS_TIMEOUT

            kwargs["run_config"] = RunConfig(
                max_workers=RAGAS_MAX_WORKERS,
                max_retries=RAGAS_MAX_RETRIES,
                timeout=RAGAS_TIMEOUT,
            )
        except ImportError:
            pass

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            raise_exceptions=False,  # 1 câu lỗi → NaN, không kill cả run
            **kwargs,
        )
        df = result.to_pandas()

        per_question = [
            EvalResult(
                question=row["question"], answer=row["answer"],
                contexts=list(row["contexts"]), ground_truth=row["ground_truth"],
                faithfulness=_clean(row.get("faithfulness")),
                answer_relevancy=_clean(row.get("answer_relevancy")),
                context_precision=_clean(row.get("context_precision")),
                context_recall=_clean(row.get("context_recall")),
            )
            for _, row in df.iterrows()
        ]

        def _mean(metric: str) -> float:
            values = [getattr(r, metric) for r in per_question]
            return round(sum(values) / len(values), 4) if values else 0.0

        return {
            "faithfulness": _mean("faithfulness"),
            "answer_relevancy": _mean("answer_relevancy"),
            "context_precision": _mean("context_precision"),
            "context_recall": _mean("context_recall"),
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return zeros


# Diagnostic Tree: metric tệ nhất → nguyên nhân gốc → cách sửa
DIAGNOSTIC_TREE = {
    "faithfulness": ("LLM hallucinating — answer không grounded trong context",
                     "Tighten prompt, lower temperature"),
    "context_recall": ("Missing relevant chunks — retrieval bỏ sót thông tin",
                       "Improve chunking or add BM25"),
    "context_precision": ("Too many irrelevant chunks — context bị nhiễu",
                          "Add reranking or metadata filter"),
    "answer_relevancy": ("Answer doesn't match question — trả lời lệch câu hỏi",
                         "Improve prompt template"),
}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    scored = []
    for item in eval_results:
        metrics = {
            "faithfulness": item.faithfulness,
            "context_recall": item.context_recall,
            "context_precision": item.context_precision,
            "answer_relevancy": item.answer_relevancy,
        }
        worst_metric = min(metrics, key=lambda m: metrics[m])
        diagnosis, fix = DIAGNOSTIC_TREE[worst_metric]
        scored.append({
            "question": item.question,
            "answer": item.answer,
            "ground_truth": item.ground_truth,
            "avg_score": round(sum(metrics.values()) / len(metrics), 4),
            "metrics": {k: round(v, 4) for k, v in metrics.items()},
            "worst_metric": worst_metric,
            "score": round(metrics[worst_metric], 4),
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })

    scored.sort(key=lambda item: item["avg_score"])
    return scored[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    from config import HAS_LLM_KEY, LLM_MODEL

    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "judge_llm": LLM_MODEL if HAS_LLM_KEY else None,
        "failures": failures,
    }
    if not HAS_LLM_KEY:
        report["note"] = ("RAGAS bị bỏ qua: không có LLM judge khả dụng "
                          "(thiếu API key / hết credit / DISABLE_LLM=1). "
                          "Scores = 0.0 KHÔNG phản ánh chất lượng pipeline — "
                          "xem reports/retrieval_diagnostic.json để so sánh tầng retrieval.")

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
