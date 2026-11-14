"""NO-CONTEXT: шестой метод PLAN_V2 §3.3. {context} — заглушка, документов нет.

Нижняя граница: что модель отвечает из параметрической памяти. Ожидаемая accuracy ~0 (вопросы про выдуманные
сущности); заметно выше нуля = утечка, факт не контролируемый. Заглушка та же, что в scripts/check_leakage.py,
иначе вертикаль NO-CONTEXT и leakage-проверка не сопоставимы. Размер корпуса и k не влияют: контекст один и тот же.
"""
from __future__ import annotations

from . import Context

PLACEHOLDER = "(документы не предоставлены)"


def build_context(size: str, question: str, k: int | None = None, corpus_text: str | None = None) -> Context:
    # chunk_ids пуст и full_corpus=False => evidence_recall = 0.0 (ничего не извлечено), а не 1.0 как у LC.
    return Context(text=PLACEHOLDER, chunk_ids=[], retrieval_ms=0.0, full_corpus=False)
