"""Детерминированные метрики: evidence recall, citation correctness, grounding цитаты, abstention."""
from __future__ import annotations

import re


def evidence_recall(gold_facts: list[dict], ctx) -> float:
    """Доля фактов вопроса, чей gold-чанк попал в контекст. У LC всегда 1.0 (весь корпус)."""
    if ctx.full_corpus:
        return 1.0
    got = set(ctx.chunk_ids)
    return sum(1 for f in gold_facts if got & set(f["chunk_ids"])) / len(gold_facts)


def citation_correct(gold_facts: list[dict], pred_doc_ids: list[str]) -> bool:
    """Все gold-документы вопроса названы в ответе."""
    return {f["doc_id"] for f in gold_facts} <= set(pred_doc_ids)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[«»\"']", "", s.lower())).strip()


def quote_grounded(quote: str, context_text: str) -> bool:
    """Цитата — дословный фрагмент контекста (ловит выдуманные цитаты). Пустая цитата = False."""
    q = _norm(quote)
    return bool(q) and q in _norm(context_text)
