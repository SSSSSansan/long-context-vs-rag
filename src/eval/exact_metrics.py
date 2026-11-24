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


# Все виды кавычек, которыми модель оформляет цитату или названия: «» „“ ” ‘ ’ " ' ‹ ›. Корпус использует «», модель часто
# пишет „“ или оборачивает цитату во внешние кавычки, поэтому на сравнение кавычки не влияют.
_QUOTES = "«»„“”‟‘’‚'\"`‹›"
_DASHES = str.maketrans({"–": "-", "—": "-", "−": "-", "‑": "-", "\u00a0": " "})


def _norm(s: str) -> str:
    """Нормализация ТОЛЬКО формы (регистр, ё/е, виды кавычек и тире, пробелы). Слова и числа не меняются:
    перефразированная цитата после неё по-прежнему не совпадёт с контекстом."""
    s = s.lower().replace("ё", "е").translate(_DASHES)
    return re.sub(r"\s+", " ", re.sub(f"[{re.escape(_QUOTES)}]", "", s)).strip()


# Метки источника, которые модель вставляет в цитату: "(doc_011)", "doc_011:", "[doc_011]", "doc_011 -", "doc_011, п. 3.17:".
_DOC_TAG = re.compile(r"[\(\[]?\bdoc_\d+(?:_c\d+)?\b[\)\]]?\s*[:\-–—]?", re.IGNORECASE)
# Ссылка на пункт сразу после метки документа: ", п. 3.17:" — это не текст документа.
_LOCATOR = re.compile(r"^[\s,;:.]*(?:п\.|пункт|пп\.|ст\.|статья|раздел)\s*\d+(?:\.\d+)*\s*[:,\-–—]?", re.IGNORECASE)
# Две цитаты подряд в одной строке: «…»; «…» / «…». «…» / «…» «…» (граница — закрывающая кавычка, разделитель, открывающая).
_QUOTE_JOIN = re.compile(r"[»”“]\s*[;.]?\s*[«„“]")
# Два пункта подряд через перевод строки или номер пункта: "4.6. … 4.9. …". Пункты не обязаны быть соседними в документе.
_CLAUSE_JOIN = re.compile(r"\n+|\s/\s|(?<=[.;»”“])\s+(?=\d+\.\d+\.\s)")


def quote_segments(quote: str, split_quotes: bool = True) -> list[str]:
    """Режет цитату на сегменты: по меткам doc_NNN и (если split_quotes) по границам «…»; «…» и «4.6. … 4.9. …» и «… / …». Для T2/T3 модель склеивает
    цитаты из двух документов в одну строку; в корпусе такой цельной строки нет, поэтому проверять надо каждый кусок отдельно.
    Каждый сегмент по-прежнему должен быть дословным (после нормализации формы) фрагментом контекста."""
    out: list[str] = []
    for part in _DOC_TAG.split(quote):
        pieces = [x for q in _QUOTE_JOIN.split(part) for x in _CLAUSE_JOIN.split(q)] if split_quotes else [part]
        for piece in pieces:
            piece = _LOCATOR.sub("", piece)
            piece = _norm(piece).strip(" .,;:|/")
            if piece:
                out.append(piece)
    return out


def quote_grounded(quote: str, context_text: str, split_quotes: bool = True) -> bool:
    """Цитата обоснована, если КАЖДЫЙ её сегмент — дословный фрагмент контекста (ловит выдуманные и пересказанные цитаты).
    Пустая цитата или цитата без текста (только метки) = False. split_quotes=False — без разбиения по границам «…»; «…»
    (нужно только для диагностики scripts/diagnose_quote_grounded.py)."""
    segs = quote_segments(quote or "", split_quotes)
    if not segs:
        return False
    ctx = _norm(context_text)
    return all(s in ctx for s in segs)


def context_text_of_row(row) -> str:
    """Контекст, относительно которого проверяется цитата, восстановленный из строки raw.jsonl (для пересчёта на старых логах).
    Обычные методы: всё после «ДОКУМЕНТЫ:» в промпте. AGENT: то, что модель увидела от search_corpus и get_chunk
    (сообщения role=tool; ответы compute/write_to_notion и ошибки не считаются), как и при прогоне (Context.text в agent.py)."""
    trace = row.get("agent_trace") if hasattr(row, "get") else None
    if isinstance(trace, list):
        names = {tc["id"]: tc["function"]["name"] for m in trace if m.get("role") == "assistant" for tc in (m.get("tool_calls") or [])}
        return "\n\n".join(m.get("content") or "" for m in trace if m.get("role") == "tool"
                            and names.get(m.get("tool_call_id")) in ("search_corpus", "get_chunk")
                            and not (m.get("content") or "").startswith("ошибка"))
    prompt = row.get("prompt") if hasattr(row, "get") else None
    return prompt.split("ДОКУМЕНТЫ:", 1)[-1] if isinstance(prompt, str) else ""
