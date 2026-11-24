"""Три способа собрать {context}. Каждый возвращает Context, дальше — один и тот же answer.py."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Context:
    text: str
    chunk_ids: list[str] = field(default_factory=list)   # пусто для LC (весь корпус)
    retrieval_ms: float = 0.0
    full_corpus: bool = False


def format_chunks(chunks) -> str:
    """Чанки в порядке ранга, с идентификаторами документа и фрагмента."""
    return "\n\n".join(f"ДОКУМЕНТ {c.doc_id} (фрагмент {c.chunk_id})\n{c.text}" for c in chunks)


ORDERS = ("relevance", "document")


def reorder(got, all_chunks, order: str = "relevance") -> list:
    """Порядок найденных чанков в промпте. relevance — по убыванию релевантности (как вернул поиск, по умолчанию);
    document — в порядке следования в корпусе, как в OP-RAG (Yu et al., arXiv 2409.01666). Состав чанков не меняется."""
    if order == "relevance":
        return list(got)
    if order == "document":
        pos = {c.chunk_id: i for i, c in enumerate(all_chunks)}
        return sorted(got, key=lambda c: pos[c.chunk_id])
    raise ValueError(f"order должен быть одним из {ORDERS}, получено {order!r}")
