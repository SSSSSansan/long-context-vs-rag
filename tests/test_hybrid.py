import pytest

from src.chunking import Chunk
from src.methods import bm25, embedding_rag, hybrid
from src.methods.registry import METHODS, NO_K_METHODS

CH = tuple(Chunk(f"doc_001_c{i:02d}", "doc_001", i, f"текст {i}", 5) for i in range(30))


@pytest.fixture(autouse=True)
def fake(monkeypatch):
    monkeypatch.setattr(bm25, "chunks_of", lambda size: CH)
    monkeypatch.setattr(bm25, "rank", lambda size, q, n: [3, 1, 7, 2, 9, 0][:n])          # BM25: лучший 3
    monkeypatch.setattr(embedding_rag, "rank", lambda size, q, n: [7, 3, 5, 1, 11, 12][:n])  # dense: лучший 7
    yield
    hybrid.set_scorer(None)


def test_fuse_math_and_determinism():
    f = hybrid.fuse([[3, 1], [1, 3]], rrf_k=60)
    assert [i for i, _ in f] == [1, 3]                                    # равный счёт -> меньший индекс первым
    assert f[0][1] == pytest.approx(1 / 61 + 1 / 62)
    g = dict(hybrid.fuse([[5], [5, 6]], rrf_k=60))
    assert g[5] == pytest.approx(1 / 61 + 1 / 61) and g[6] == pytest.approx(1 / 62)


def test_fuse_dedupes_and_prefers_docs_found_by_both():
    f = [i for i, _ in hybrid.fuse([[3, 1, 7, 2], [7, 3, 5, 1]])]
    assert len(f) == len(set(f)) and f[:2] == [3, 7] and set(f) == {1, 2, 3, 5, 7}


def test_hyb_rrf_returns_top_k_without_reranker(monkeypatch):
    monkeypatch.setattr(hybrid, "score_pairs", lambda *a: pytest.fail("реранкер не должен вызываться в hyb_rrf"))
    ctx = METHODS["hyb_rrf"]("8k", "вопрос", 3)
    assert ctx.chunk_ids == ["doc_001_c03", "doc_001_c07", "doc_001_c01"] and not ctx.full_corpus
    assert ctx.text.count("ДОКУМЕНТ") == 3


def test_hyb_rerank_reorders_candidates_and_cuts_to_k():
    hybrid.set_scorer(lambda q, texts: [1.0 if t == "текст 5" else 0.0 for t in texts])   # реранкер поднимает чанк 5
    ctx = METHODS["hyb_rerank"]("8k", "вопрос", 2)
    assert ctx.chunk_ids[0] == "doc_001_c05" and len(ctx.chunk_ids) == 2
    assert ctx.chunk_ids[1] == "doc_001_c03"                              # ничья -> порядок RRF (3 выше)


def test_rerank_sees_whole_pool_not_only_top_k():
    seen = {}
    hybrid.set_scorer(lambda q, texts: seen.setdefault("n", len(texts)) and [0.0] * len(texts))
    METHODS["hyb_rerank"]("8k", "q", 1)
    assert seen["n"] == 9                                                 # объединение двух списков по 6: {0,1,2,3,5,7,9,11,12}


def test_k_defaults_to_config_and_registered():
    assert "hyb_rrf" in METHODS and "hyb_rerank" in METHODS
    assert "hyb_rrf" not in NO_K_METHODS and "hyb_rerank" not in NO_K_METHODS     # у HYB k разворачивается как у RAG
    assert len(METHODS["hyb_rrf"]("8k", "q").chunk_ids) == 8


def test_expand_tasks_gives_k_axis_for_hyb():
    from src.eval import run
    qs = [dict(qid="q1", type="T1", first_corpus="8k", set="A", anchor=True, fact_ids=["f"])]
    spec = dict(runs=[dict(sizes=["8k"], methods=["hyb_rrf", "hyb_rerank"], question_types=["T1"], n_per_type=1,
                           repeats=1, ks=[4, 8], question_set="A")])
    assert len(run.expand_tasks(spec, qs, 42)) == 4
