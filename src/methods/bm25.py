"""BM25 top-k по чанкам (rank_bm25). Токенизация: нижний регистр + грубый стемминг (префикс 6 букв),
иначе русская морфология ломает совпадения («выплаты» / «выплата»). Выбор зафиксирован здесь и в README."""
from __future__ import annotations

import re
import time
from functools import lru_cache

from rank_bm25 import BM25Okapi

from ..chunking import Chunk, chunk_corpus
from ..config import base_cfg
from . import Context, format_chunks
from .long_context import load_corpus

STEM = 6


def tokenize(text: str) -> list[str]:
    return [w[:STEM] for w in re.findall(r"\w+", text.lower())]


@lru_cache(maxsize=None)
def chunks_of(size: str) -> tuple[Chunk, ...]:
    return tuple(chunk_corpus(load_corpus(size)))


@lru_cache(maxsize=None)
def _index(size: str) -> BM25Okapi:
    return BM25Okapi([tokenize(c.text) for c in chunks_of(size)])


def retrieve(size: str, question: str, k: int) -> list[Chunk]:
    chunks = chunks_of(size)
    scores = _index(size).get_scores(tokenize(question))
    order = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))[:k]
    return [chunks[i] for i in order]


def build_context(size: str, question: str, k: int | None = None, corpus_text: str | None = None) -> Context:
    k = k or base_cfg()["retrieval"]["top_k"]
    t0 = time.perf_counter()
    got = retrieve(size, question, k)
    return Context(text=format_chunks(got), chunk_ids=[c.chunk_id for c in got],
                   retrieval_ms=(time.perf_counter() - t0) * 1000)
