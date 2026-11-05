"""Единый промпт ответа. Для всех методов (LC / BM25 / EMB) различается ТОЛЬКО {context}."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .llm_client import LLMResponse, complete

PROMPT = """Ответь на вопрос, опираясь ТОЛЬКО на приведённые документы.

Правила:
1. Если ответа в документах нет — напиши ровно: "В предоставленных документах ответа нет."
   Не достраивай из общих знаний.
2. Укажи идентификатор документа, из которого взят ответ.
3. Приведи дословную цитату-обоснование.
4. Если документы противоречат друг другу — укажи это явно
   и перечисли обе версии с их источниками.
5. Отвечай кратко, без пересказа документов.

Верни JSON:
{"answer": "...", "doc_ids": ["..."], "quote": "...", "contradiction_found": true|false}

ДОКУМЕНТЫ:
{context}

ВОПРОС: {question}"""

ABSTAIN = "В предоставленных документах ответа нет."


def build_messages(context: str, question: str) -> list[dict]:
    # replace, а не format: в шаблоне есть фигурные скобки JSON
    return [{"role": "user", "content": PROMPT.replace("{context}", context).replace("{question}", question)}]


@dataclass
class Answer:
    text: str                 # сырой ответ модели
    answer: str
    doc_ids: list[str]
    quote: str
    contradiction_found: bool
    parsed_ok: bool
    abstained: bool
    resp: LLMResponse


def parse(text: str) -> tuple[dict, bool]:
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            if isinstance(d, dict):
                return d, True
        except json.JSONDecodeError:
            pass
    return {"answer": text.strip()}, False


def answer_question(context: str, question: str, *, salt: str = "", max_tokens: int | None = None) -> Answer:
    resp = complete(build_messages(context, question), salt=salt, max_tokens=max_tokens)
    d, ok = parse(resp.text)
    ans = d.get("answer", "")
    ans = ans if isinstance(ans, str) else json.dumps(ans, ensure_ascii=False)
    ids = d.get("doc_ids") or []
    ids = [str(i) for i in ids] if isinstance(ids, list) else [str(ids)]
    quote = d.get("quote") or ""
    return Answer(text=resp.text, answer=ans, doc_ids=ids,
                  quote=quote if isinstance(quote, str) else str(quote),
                  contradiction_found=bool(d.get("contradiction_found", False)),
                  parsed_ok=ok, abstained=ABSTAIN.rstrip(".") in resp.text, resp=resp)
