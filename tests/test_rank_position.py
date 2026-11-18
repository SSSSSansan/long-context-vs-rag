import pytest

from src.chunking import Chunk
from src.eval import rank_position as rp
from src.eval import run
from src.methods import bm25

CH = tuple(Chunk(f"doc_001_c{i:02d}", "doc_001", i, f"фон {i}", 5) for i in range(40))
CH = CH[:7] + (Chunk("doc_001_c07", "doc_001", 7, "ФАКТ-СТРОКА. СВЯЗКА-СТРОКА.", 5),) + CH[8:]
CH = CH[:9] + (Chunk("doc_001_c09", "doc_001", 9, "ещё раз ФАКТ-СТРОКА. (перекрытие)", 5),) + CH[10:]   # дубликат факта: не дистрактор
GOLD = {"f1": dict(fact_id="f1", chunk_ids=["doc_001_c07", "doc_001_c09"], statement="ФАКТ-СТРОКА.", doc_id="doc_001",
                   positions_pct={}),
        "l_f1": dict(fact_id="l_f1", chunk_ids=["doc_001_c07"], statement="СВЯЗКА-СТРОКА.", doc_id="doc_001", positions_pct={})}
Q = dict(qid="q1h", type="T1", set="A", anchor=False, fact_ids=["f1"], link_fact_ids=["l_f1"], question="?", gold_answer="1")


@pytest.fixture(autouse=True)
def fake(monkeypatch):
    monkeypatch.setattr(bm25, "chunks_of", lambda size: CH)
    monkeypatch.setattr(bm25, "rank", lambda size, q, n: [7, 9] + list(range(0, 7)) + list(range(10, 40)))


@pytest.mark.parametrize("rank", [1, 5, 10])
def test_gold_at_requested_rank_and_ten_chunks(rank):
    ctx = rp.build_ranked_context("30k", Q, GOLD, rank)
    assert len(ctx.chunk_ids) == 10 and len(set(ctx.chunk_ids)) == 10
    assert ctx.chunk_ids[rank - 1] == "doc_001_c07"
    assert "doc_001_c09" not in ctx.chunk_ids                       # перекрытие с фактом — не дистрактор
    assert ctx.text.count("ФАКТ-СТРОКА") == 1 and ctx.text.count("СВЯЗКА-СТРОКА") == 1


def test_only_gold_position_changes_between_ranks():
    a = [c for c in rp.build_ranked_context("30k", Q, GOLD, 1).chunk_ids if c != "doc_001_c07"]
    b = [c for c in rp.build_ranked_context("30k", Q, GOLD, 10).chunk_ids if c != "doc_001_c07"]
    assert a == b                                                   # те же 9 дистракторов в том же порядке


def test_rejects_multi_fact_and_bad_rank():
    with pytest.raises(ValueError):
        rp.build_ranked_context("30k", dict(Q, fact_ids=["f1", "f1"]), GOLD, 1)
    with pytest.raises(ValueError):
        rp.build_ranked_context("30k", Q, GOLD, 11)


def test_rank_tasks_count_and_keys():
    qs = [dict(Q, qid=f"q{i}h", type="T1") for i in range(8)] + [dict(Q, qid="qt2h", type="T2")]
    spec = dict(runs=[dict(question_set="A-hard", sizes=["30k"], ranks=[1, 5, 10], n_questions=6, repeats=1)])
    tasks, _ = rp.rank_tasks(spec, GOLD, qs)
    assert len(tasks) == 18 and {t["pos"] for t in tasks} == {1, 5, 10} and all(t["q"]["type"] == "T1" for t in tasks)
    assert len({run.task_key(t) for t in tasks}) == 18             # ключи различаются -> прогон возобновляем


def test_task_fn_runs_with_fake_llm(monkeypatch):
    class R:
        model = "m"; model_api = None; prompt_tokens = 10; completion_tokens = 5; cached_tokens = 0
        latency_ms = 1.0; cost_usd = 0.0; from_local_cache = False

    class A:
        text = answer = "x"; doc_ids = ["doc_001"]; quote = ""; contradiction_found = False
        parsed_ok = True; abstained = False; resp = R()

    monkeypatch.setattr(run, "answer_question", lambda ctx, q, salt="": A())
    monkeypatch.setattr(run, "judge", lambda *a, **k: ("incorrect", "", R()))
    spec = dict(runs=[dict(question_set="A-hard", sizes=["30k"], ranks=[5], n_questions=1, repeats=1)])
    tasks, fn = rp.rank_tasks(spec, GOLD, [Q])
    row = fn(tasks[0])
    assert row["gold_rank"] == 5 and row["method"] == "p2_bm25" and row["evidence_recall"] == 1.0 and row["k"] == 10
