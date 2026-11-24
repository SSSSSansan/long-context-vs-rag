"""Бонус-проверки: порядок чанков (OP-RAG), ключ кэша эмбеддингов с моделью, регистрация методов. Без сети и без моделей."""
import json

import pytest

from src.chunking import Chunk
from src.methods import ORDERS, reorder
from src.methods import embedding_rag as er
from src.methods.registry import METHODS, NO_K_METHODS


def _chunks():
    return [Chunk(f"doc_00{d}_c0{i}", f"doc_00{d}", i, f"t{d}{i}", 10) for d in (1, 2) for i in range(3)]


def test_reorder_document_keeps_set_and_sorts_by_corpus_position():
    allc = _chunks()
    got = [allc[4], allc[0], allc[3]]                       # порядок релевантности
    assert [c.chunk_id for c in reorder(got, allc, "relevance")] == [allc[4].chunk_id, allc[0].chunk_id, allc[3].chunk_id]
    doc = reorder(got, allc, "document")
    assert [c.chunk_id for c in doc] == [allc[0].chunk_id, allc[3].chunk_id, allc[4].chunk_id]
    assert {c.chunk_id for c in doc} == {c.chunk_id for c in got}      # состав тот же
    with pytest.raises(ValueError):
        reorder(got, allc, "random")
    assert ORDERS == ("relevance", "document")


def test_new_methods_registered_and_use_k():
    assert {"emb_bge", "bm25_doc", "hyb_rerank_doc"} <= set(METHODS)
    assert not {"emb_bge", "bm25_doc", "hyb_rerank_doc"} & set(NO_K_METHODS)


def test_embed_cache_key_depends_on_model(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(er, "ROOT", tmp_path)
    monkeypatch.setattr(er, "embed", lambda texts, backend=None: calls.append(backend) or [[1.0, 0.0] for _ in texts])
    monkeypatch.setattr(er, "embed_model", lambda backend=None: "model-A")
    er._embed_cached(["x"], "local"); er._embed_cached(["x"], "local")
    assert len(calls) == 1                                  # второй раз — из кэша
    monkeypatch.setattr(er, "embed_model", lambda backend=None: "model-B")
    er._embed_cached(["x"], "local")
    assert len(calls) == 2                                  # другая модель — другой ключ, пересчёт
    er._embed_cached(["x"], "openai")
    assert len(calls) == 3 and calls[-1] == "openai"        # другой backend — другой ключ


def test_legacy_embed_cache_still_read_for_original_models(tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setattr(er, "ROOT", tmp_path)
    (tmp_path / "runs" / "emb_cache").mkdir(parents=True)
    legacy = hashlib.sha256(json.dumps(["openai", ["q"]], ensure_ascii=False).encode()).hexdigest()
    (tmp_path / "runs" / "emb_cache" / f"{legacy}.json").write_text("[[0.5, 0.5]]")
    monkeypatch.setattr(er, "embed", lambda *a, **k: pytest.fail("не должно считаться заново"))
    assert er._embed_cached(["q"], "openai") == [[0.5, 0.5]]          # text-embedding-3-small: старый ключ принят
