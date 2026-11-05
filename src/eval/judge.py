"""LLM-судья. Слеп к методу: видит только вопрос, эталон и текст ответа."""
from __future__ import annotations

import json
import re

from ..llm_client import LLMResponse, complete

JUDGE_PROMPT = """Ты оцениваешь правильность ответа на вопрос по документам.

ВОПРОС: {question}
ЭТАЛОННЫЙ ОТВЕТ: {gold_answer}
ОТВЕТ СИСТЕМЫ: {prediction}

Оценка:
- correct — передаёт тот же факт, что и эталон (формулировка может отличаться)
- partial — часть факта верна, часть упущена или искажена
- incorrect — противоречит эталону или содержит выдуманный факт
- abstain — система заявила, что ответа в документах нет

Верни JSON: {"verdict": "...", "rationale": "одно предложение"}

Оценивай только фактическое совпадение. Стиль и длина значения не имеют."""

VERDICTS = {"correct", "partial", "incorrect", "abstain"}


def judge(question: str, gold_answer: str, prediction: str, salt: str = "") -> tuple[str, str, LLMResponse]:
    p = (JUDGE_PROMPT.replace("{question}", question).replace("{gold_answer}", gold_answer)
         .replace("{prediction}", prediction or "(пустой ответ)"))
    resp = complete([{"role": "user", "content": p}], role="judge", max_tokens=120, temperature=0.0, salt=salt)
    m = re.search(r"\{.*\}", resp.text, re.S)
    verdict, why = "incorrect", resp.text.strip()
    if m:
        try:
            d = json.loads(m.group(0))
            v = str(d.get("verdict", "")).lower().strip()
            verdict, why = (v if v in VERDICTS else "incorrect"), str(d.get("rationale", ""))
        except json.JSONDecodeError:
            pass
    return verdict, why, resp
