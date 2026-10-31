"""Шаг 1.4: чанкование со стабильными chunk_id вида doc_019_c04.

Чанки режутся по границам строк (пунктов), а не посередине предложения:
накапливаем строки, пока не наберётся ~chunk_tokens, затем начинаем следующий чанк
с хвоста предыдущего (перекрытие ~chunk_overlap токенов). Всё детерминировано:
одинаковый текст -> одинаковые chunk_id.

chunk_id нумеруется внутри документа, поэтому он меняется, если в документ вставить
пункт ДО этого места. Правило проекта: чанкуем ПОСЛЕ внедрения фактов, а chunk_id
в gold.json записываем уже по итоговому чанкингу.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .config import base_cfg
from .tokens import count_tokens

HEADER = re.compile(r"^ДОКУМЕНТ (doc_\d+)$")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    idx: int
    text: str
    n_tokens: int


def split_docs(corpus_text: str) -> list[tuple[str, list[str]]]:
    """Корпус -> [(doc_id, строки документа)] по заголовкам «ДОКУМЕНТ doc_NNN»."""
    docs: list[tuple[str, list[str]]] = []
    cur_id, cur = None, []
    for line in corpus_text.split("\n"):
        m = HEADER.match(line)
        if m:
            if cur_id is not None:
                docs.append((cur_id, cur))
            cur_id, cur = m.group(1), [line]
        elif cur_id is not None:
            cur.append(line)
    if cur_id is not None:
        docs.append((cur_id, cur))
    return docs


def chunk_doc(doc_id: str, lines: list[str], chunk_tokens: int, overlap: int,
              count: Callable[[str], int] = count_tokens) -> list[Chunk]:
    lt = [count(l) + 1 for l in lines]          # +1 на перевод строки
    chunks: list[Chunk] = []
    start = 0
    while start < len(lines):
        end, tok = start, 0
        while end < len(lines) and (end == start or tok + lt[end] <= chunk_tokens):
            tok += lt[end]
            end += 1
        text = "\n".join(lines[start:end]).strip()
        if text:
            idx = len(chunks)
            chunks.append(Chunk(f"{doc_id}_c{idx:02d}", doc_id, idx, text, count(text)))
        if end >= len(lines):
            break
        back, ot = end, 0                        # хвост предыдущего чанка -> перекрытие
        while back - 1 > start and ot + lt[back - 1] <= overlap:
            back -= 1
            ot += lt[back]
        start = back
    return chunks


def chunk_corpus(corpus_text: str, chunk_tokens: int | None = None, overlap: int | None = None,
                 count: Callable[[str], int] = count_tokens) -> list[Chunk]:
    r = base_cfg()["retrieval"]
    chunk_tokens = r["chunk_tokens"] if chunk_tokens is None else chunk_tokens
    overlap = r["chunk_overlap"] if overlap is None else overlap
    out: list[Chunk] = []
    for doc_id, lines in split_docs(corpus_text):
        out.extend(chunk_doc(doc_id, lines, chunk_tokens, overlap, count))
    return out
