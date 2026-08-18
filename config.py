"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- OpenRouter (alternative to OpenAI) ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

# Determine which LLM backend to use
USE_OPENROUTER = bool(OPENROUTER_API_KEY and OPENROUTER_API_KEY != "sk-or-v1-...")
LLM_API_KEY = OPENROUTER_API_KEY if USE_OPENROUTER else OPENAI_API_KEY
LLM_MODEL = OPENROUTER_MODEL if USE_OPENROUTER else "gpt-4o-mini"

# DISABLE_LLM=1 → chạy pipeline hoàn toàn offline (test fallback / khi hết quota API)
LLM_DISABLED = os.getenv("DISABLE_LLM", "").strip().lower() in ("1", "true", "yes")
HAS_LLM_KEY = (not LLM_DISABLED) and bool(LLM_API_KEY and LLM_API_KEY not in ("sk-...", "sk-or-v1-..."))

# Tham số gửi kèm mỗi request (OpenRouter-only). Model reasoning (VD nemotron) mặc định
# in cả chain-of-thought vào content → phá JSON parsing của M5/RAGAS, nên tắt reasoning.
LLM_EXTRA_BODY: dict = {}
if USE_OPENROUTER and ":free" in LLM_MODEL:
    LLM_EXTRA_BODY = {"reasoning": {"exclude": True, "enabled": False}}


def get_llm_client():
    """Return an OpenAI-compatible client configured for OpenRouter or OpenAI.

    Usage (drop-in replacement for `OpenAI()`):
        client = get_llm_client()
        resp = client.chat.completions.create(model=LLM_MODEL, messages=[...])
    """
    from openai import OpenAI

    if USE_OPENROUTER:
        return OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
    else:
        return OpenAI(api_key=OPENAI_API_KEY)


# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

_ENCODER_CACHE: dict = {}


def get_encoder(model_name: str = EMBEDDING_MODEL):
    """Trả về SentenceTransformer đã cache (bge-m3 ~2.3GB → chỉ load 1 lần/process).

    Dùng chung cho dense search (M2) và RAGAS answer_relevancy (M4).
    """
    if model_name not in _ENCODER_CACHE:
        from sentence_transformers import SentenceTransformer

        _ENCODER_CACHE[model_name] = SentenceTransformer(model_name)
    return _ENCODER_CACHE[model_name]

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Evaluation ---
# Model :free của OpenRouter chỉ cho 20 req/phút → giảm concurrency để tránh 429.
RAGAS_MAX_WORKERS = 4 if ":free" in LLM_MODEL else 8
RAGAS_MAX_RETRIES = 12
RAGAS_TIMEOUT = 300

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")

