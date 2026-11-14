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


# Метки источника, которые модель вставляет в цитату: "(doc_011)", "doc_011:", "[doc_011]", "doc_011 -".
_DOC_TAG = re.compile(r"[\(\[]?\bdoc_\d+\b[\)\]]?\s*[:\-–—]?", re.IGNORECASE)


def quote_segments(quote: str) -> list[str]:
    """Режет цитату по меткам doc_NNN на сегменты. Для T2/T3 модель склеивает цитаты из двух документов
    в одну строку; в корпусе такой цельной строки нет, поэтому проверять надо каждый кусок отдельно."""
    parts = _DOC_TAG.split(quote)
    return [p for p in (_norm(p).strip(" .,;:|/") for p in parts) if p]


def quote_grounded(quote: str, context_text: str) -> bool:
    """Цитата обоснована, если КАЖДЫЙ её сегмент — дословный фрагмент контекста (ловит выдуманные цитаты).
    Пустая цитата или цитата без текста (только метки) = False."""
    segs = quote_segments(quote or "")
    if not segs:
        return False
    ctx = _norm(context_text)
    return all(s in ctx for s in segs)
