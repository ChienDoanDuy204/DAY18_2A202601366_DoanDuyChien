from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


_SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"
_semantic_model = None


def _get_semantic_model():
    """Lazy-load + cache sentence encoder (tránh load lại mỗi lần gọi)."""
    global _semantic_model
    if _semantic_model is None:
        from sentence_transformers import SentenceTransformer

        _semantic_model = SentenceTransformer(_SEMANTIC_MODEL_NAME)
    return _semantic_model


def _split_sentences(text: str) -> list[str]:
    """Tách câu: sau dấu .!? hoặc ranh giới paragraph."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    from numpy import dot
    from numpy.linalg import norm

    metadata = metadata or {}
    sentences = _split_sentences(text)
    if not sentences:
        return []

    embeddings = _get_semantic_model().encode(sentences)

    chunks: list[Chunk] = []
    group = [sentences[0]]

    def _flush(group_sentences: list[str]) -> None:
        chunks.append(Chunk(
            text=" ".join(group_sentences).strip(),
            metadata={**metadata, "strategy": "semantic", "chunk_index": len(chunks)},
        ))

    for i in range(1, len(sentences)):
        prev, curr = embeddings[i - 1], embeddings[i]
        sim = float(dot(prev, curr) / (norm(prev) * norm(curr) + 1e-9))
        if sim < threshold:
            # Chủ đề đổi → đóng chunk hiện tại, mở chunk mới
            _flush(group)
            group = [sentences[i]]
        else:
            group.append(sentences[i])

    if group:
        _flush(group)

    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def _pack_units(units: list[str], max_size: int, joiner: str) -> list[str]:
    """Gộp tuần tự các unit thành group ≤ max_size chars (giữ nguyên thứ tự)."""
    groups: list[str] = []
    current: list[str] = []
    current_len = 0

    for unit in units:
        add_len = len(unit) + (len(joiner) if current else 0)
        if current and current_len + add_len > max_size:
            groups.append(joiner.join(current))
            current, current_len = [], 0
            add_len = len(unit)
        current.append(unit)
        current_len += add_len

    if current:
        groups.append(joiner.join(current))
    return groups


def _child_units(parent_text: str, child_size: int) -> list[str]:
    """Đơn vị nhỏ để đóng gói thành child: theo dòng, dòng quá dài → tách câu."""
    units: list[str] = []
    for line in [ln.strip() for ln in parent_text.split("\n") if ln.strip()]:
        if len(line) > child_size:
            units.extend(_split_sentences(line) or [line])
        else:
            units.append(line)
    return units


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return ([], [])

    parents: list[Chunk] = []
    children: list[Chunk] = []
    # prefix theo source để parent_id không trùng khi index nhiều document
    source = str(metadata.get("source", "")).strip()
    prefix = f"{source}::" if source else ""

    for parent_text in _pack_units(paragraphs, parent_size, "\n\n"):
        pid = f"{prefix}parent_{len(parents)}"
        parents.append(Chunk(
            text=parent_text,
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
        ))

        for j, child_text in enumerate(_pack_units(_child_units(parent_text, child_size),
                                                   child_size, " ")):
            children.append(Chunk(
                text=child_text,
                metadata={**metadata, "chunk_type": "child", "chunk_index": j},
                parent_id=pid,
            ))

    return (parents, children)


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    # capture group → giữ lại chính dòng header trong kết quả split
    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)

    chunks: list[Chunk] = []
    current_header = "Intro"
    current_content = ""

    def _flush() -> None:
        nonlocal current_content
        if current_content.strip():
            chunks.append(Chunk(
                text=f"{current_header}\n\n{current_content.strip()}",
                metadata={**metadata, "section": current_header,
                          "strategy": "structure", "chunk_index": len(chunks)},
            ))
        current_content = ""

    for sec in sections:
        if not sec.strip():
            continue
        if re.match(r'^#{1,3}\s+.+$', sec.strip()):
            _flush()                      # đóng section trước khi sang header mới
            current_header = sec.strip()
        else:
            current_content += sec + "\n"

    _flush()
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
