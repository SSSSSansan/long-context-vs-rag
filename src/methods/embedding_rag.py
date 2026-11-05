"""Embedding RAG: чанки -> эмбеддинги -> Qdrant (in-memory) -> top-k по косинусу.
Эмбеддинги чанков кэшируются на диск (runs/emb_cache), чтобы не платить повторно."""
from __future__ import annotations

import hashlib
import json
import os
import time
from functools import lru_cache

from qdrant_client import QdrantClient, models

from ..config import ROOT, backend_name, base_cfg
from ..llm_client import embed
from . import Context, format_chunks
from .bm25 import chunks_of

_lock = __import__("threading").Lock()


def _embed_backend() -> str:
    return os.getenv("EMBED_BACKEND", backend_name()).lower()


def _embed_cached(texts: list[str]) -> list[list[float]]:
    d = ROOT / "runs" / "emb_cache"
    d.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps([_embed_backend(), texts], ensure_ascii=False).encode()).hexdigest()
    p = d / f"{key}.json"
    if p.exists():
        return json.loads(p.read_text())
    vecs = embed(texts)
    p.write_text(json.dumps(vecs))
    return vecs


@lru_cache(maxsize=None)
def _store(size: str):
    chunks = chunks_of(size)
    vecs = _embed_cached([c.text for c in chunks])
    client = QdrantClient(":memory:")
    client.create_collection("c", vectors_config=models.VectorParams(size=len(vecs[0]), distance=models.Distance.COSINE))
    client.upsert("c", points=[models.PointStruct(id=i, vector=v) for i, v in enumerate(vecs)])
    return client


def retrieve(size: str, question: str, k: int):
    with _lock:
        client = _store(size)
    qv = _embed_cached([question])[0]
    hits = client.query_points("c", query=qv, limit=k).points
    chunks = chunks_of(size)
    return [chunks[h.id] for h in hits]


def build_context(size: str, question: str, k: int | None = None, corpus_text: str | None = None) -> Context:
    k = k or base_cfg()["retrieval"]["top_k"]
    t0 = time.perf_counter()
    got = retrieve(size, question, k)
    return Context(text=format_chunks(got), chunk_ids=[c.chunk_id for c in got],
                   retrieval_ms=(time.perf_counter() - t0) * 1000)
