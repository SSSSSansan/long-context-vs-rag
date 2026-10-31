"""Единый счётчик токенов для сборки корпусов и чанкинга.

Токенайзер берётся из configs/base.yaml (поле `tokenizer`), чтобы размеры корпусов
(8K / 30K / 100K) всегда считались одним и тем же способом. Для русского текста
токенайзер Qwen тратит токенов заметно больше, поэтому локально 30K по o200k не влезет
в окно 32K — это учитывает llm_client.
"""
from __future__ import annotations

from functools import lru_cache

from .config import backend_cfg


@lru_cache(maxsize=1)
def _encoder():
    import tiktoken
    return tiktoken.get_encoding(backend_cfg()["tokenizer"])


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text))
