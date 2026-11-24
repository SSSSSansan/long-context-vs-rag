"""HYB: гибридный поиск. Два метода (PLAN_V2 §3.4), чтобы не приписать гибриду выигрыш реранкера:

  hyb_rrf     dense (EMB) + BM25, по top-N кандидатов каждый, слияние RRF, top-k без реранкера
  hyb_rerank  то же слияние, затем переранжирование BAAI/bge-reranker-v2-m3 (cross-encoder, sentence-transformers,
              Apache 2.0, локально и бесплатно), top-k

RRF: score(d) = sum_r 1 / (rrf_k + rank_r(d)), rank с 1. Без мультивектора (решение Panda): реранкер — cross-encoder.
Параметры в configs/base.yaml -> retrieval: hybrid_candidates (N=20), rrf_k (60), reranker_model.
Реранкер грузится лениво при первом hyb_rerank; в тестах подменяется через set_scorer().
"""
from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Callable

from ..config import base_cfg
from . import Context, format_chunks, reorder
from . import bm25, embedding_rag

Scorer = Callable[[str, list[str]], list[float]]
_scorer: Scorer | None = None


def set_scorer(fn: Scorer | None) -> None:
    """Подмена реранкера (тесты). None — вернуть настоящий."""
    global _scorer
    _scorer = fn


@lru_cache(maxsize=1)
def _cross_encoder():
    from sentence_transformers import CrossEncoder
    # CPU по умолчанию: на Mac MPS делит память с Ollama и падает с OutOfMemory (прогон Ш2 от 17.11). RERANK_DEVICE=mps|cuda — вернуть GPU.
    device = os.environ.get("RERANK_DEVICE", "cpu")
    return CrossEncoder(base_cfg()["retrieval"].get("reranker_model", "BAAI/bge-reranker-v2-m3"), max_length=512, device=device)


def score_pairs(question: str, texts: list[str]) -> list[float]:
    if _scorer is not None:
        return list(_scorer(question, texts))
    return [float(x) for x in _cross_encoder().predict([(question, t) for t in texts], show_progress_bar=False)]


def fuse(rankings: list[list[int]], rrf_k: int = 60) -> list[tuple[int, float]]:
    """RRF по спискам индексов (лучший первый). Детерминированно: при равенстве — меньший индекс."""
    score: dict[int, float] = {}
    for r in rankings:
        for pos, idx in enumerate(r, start=1):
            score[idx] = score.get(idx, 0.0) + 1.0 / (rrf_k + pos)
    return sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))


def retrieve(size: str, question: str, k: int, rerank: bool) -> list:
    cfg = base_cfg()["retrieval"]
    n, rrf_k = cfg.get("hybrid_candidates", 20), cfg.get("rrf_k", 60)
    fused = fuse([bm25.rank(size, question, n), embedding_rag.rank(size, question, n)], rrf_k)
    chunks = bm25.chunks_of(size)
    if not rerank:
        return [chunks[i] for i, _ in fused[:k]]
    pool = [i for i, _ in fused]                                       # до 2N уникальных кандидатов
    sc = score_pairs(question, [chunks[i].text for i in pool])
    order = sorted(range(len(pool)), key=lambda j: (-sc[j], j))[:k]    # при равенстве — выше в RRF
    return [chunks[pool[j]] for j in order]


def _build(rerank: bool, order: str = "relevance"):
    def build_context(size: str, question: str, k: int | None = None, corpus_text: str | None = None) -> Context:
        k = k or base_cfg()["retrieval"]["top_k"]
        t0 = time.perf_counter()
        got = reorder(retrieve(size, question, k, rerank), bm25.chunks_of(size), order)
        return Context(text=format_chunks(got), chunk_ids=[c.chunk_id for c in got],
                       retrieval_ms=(time.perf_counter() - t0) * 1000)
    return build_context


build_rrf = _build(False)
build_rerank = _build(True)
build_rerank_doc = _build(True, order="document")   # OP-RAG: те же чанки, порядок как в документе
