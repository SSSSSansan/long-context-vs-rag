"""Embedding RAG: чанки -> эмбеддинги -> Qdrant (in-memory) -> top-k по косинусу.
Эмбеддинги чанков кэшируются на диск (runs/emb_cache), чтобы не платить повторно.
Ключ кэша включает backend И id модели эмбеддингов (configs/base.yaml -> backends.<b>.embed_model): эмбеддинги разных моделей не смешиваются.
Метод emb_bge (постфактум-проверка, REPORT.md): те же чанки и поиск, но эмбеддинги BAAI/bge-m3 локально (бесплатно) при генерации на OpenAI."""
from __future__ import annotations

import hashlib
import json
import os
import time
from functools import lru_cache

from qdrant_client import QdrantClient, models

from ..config import ROOT, backend_name, base_cfg
from ..llm_client import embed
from . import Context, format_chunks, reorder
from .bm25 import chunks_of

_lock = __import__("threading").Lock()
# Модели, под которыми писался кэш ДО того, как id модели вошёл в ключ (ключ был sha256([backend, texts])). Для них старые файлы читаются как есть.
_LEGACY_MODEL = {"openai": "text-embedding-3-small", "local": "BAAI/bge-m3"}


def _embed_backend() -> str:
    return os.getenv("EMBED_BACKEND", backend_name()).lower()


def embed_model(backend: str | None = None) -> str:
    return base_cfg()["backends"][(backend or _embed_backend()).lower()]["embed_model"]


def _embed_cached(texts: list[str], backend: str | None = None) -> list[list[float]]:
    which = (backend or _embed_backend()).lower()
    model = embed_model(which)
    d = ROOT / "runs" / "emb_cache"
    d.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps([which, model, texts], ensure_ascii=False).encode()).hexdigest()
    p = d / f"{key}.json"
    if not p.exists() and model == _LEGACY_MODEL.get(which):          # прежний ключ без модели: тот же результат, платить повторно не надо
        legacy = d / f"{hashlib.sha256(json.dumps([which, texts], ensure_ascii=False).encode()).hexdigest()}.json"
        if legacy.exists():
            p = legacy
    if p.exists():
        return json.loads(p.read_text())
    vecs = embed(texts, backend=which)
    p.write_text(json.dumps(vecs))
    return vecs


@lru_cache(maxsize=None)
def _store(size: str, backend: str):
    chunks = chunks_of(size)
    vecs = _embed_cached([c.text for c in chunks], backend)
    client = QdrantClient(":memory:")
    client.create_collection("c", vectors_config=models.VectorParams(size=len(vecs[0]), distance=models.Distance.COSINE))
    client.upsert("c", points=[models.PointStruct(id=i, vector=v) for i, v in enumerate(vecs)])
    return client


def retrieve(size: str, question: str, k: int, backend: str | None = None):
    which = (backend or _embed_backend()).lower()
    with _lock:
        client = _store(size, which)
    qv = _embed_cached([question], which)[0]
    hits = client.query_points("c", query=qv, limit=k).points
    chunks = chunks_of(size)
    return [chunks[h.id] for h in hits]


def rank(size: str, question: str, n: int, backend: str | None = None) -> list[int]:
    """Индексы чанков по убыванию косинуса (для гибрида)."""
    which = (backend or _embed_backend()).lower()
    with _lock:
        client = _store(size, which)
    qv = _embed_cached([question], which)[0]
    return [h.id for h in client.query_points("c", query=qv, limit=n).points]


def build_context(size: str, question: str, k: int | None = None, corpus_text: str | None = None,
                  order: str = "relevance", backend: str | None = None) -> Context:
    k = k or base_cfg()["retrieval"]["top_k"]
    t0 = time.perf_counter()
    got = reorder(retrieve(size, question, k, backend), chunks_of(size), order)
    return Context(text=format_chunks(got), chunk_ids=[c.chunk_id for c in got],
                   retrieval_ms=(time.perf_counter() - t0) * 1000)


def build_context_bge(size: str, question: str, k: int | None = None, corpus_text: str | None = None) -> Context:
    """emb_bge: эмбеддинги bge-m3 локально, независимо от LLM_BACKEND."""
    return build_context(size, question, k, corpus_text, backend="local")
