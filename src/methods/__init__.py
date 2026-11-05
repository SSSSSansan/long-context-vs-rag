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
