from src.eval import run
from src.eval.exact_metrics import evidence_recall
from src.methods import no_context
from src.methods.registry import METHODS, NO_K_METHODS


def test_registered_and_context_is_placeholder_for_any_size_and_k():
    assert "nocontext" in METHODS and "nocontext" in NO_K_METHODS
    for size in ("8k", "30k", "100k"):
        for k in (None, 4, 8):
            c = METHODS["nocontext"](size, "любой вопрос", k)
            assert c.text == no_context.PLACEHOLDER and c.chunk_ids == [] and c.full_corpus is False


def test_placeholder_matches_leakage_script():
    from scripts.check_leakage import EMPTY
    assert no_context.PLACEHOLDER == EMPTY


def test_no_context_does_not_read_corpus_files(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("NO-CONTEXT не должен читать корпус")
    monkeypatch.setattr("src.methods.long_context.load_corpus", boom)
    assert METHODS["nocontext"]("100k", "q", None).text == no_context.PLACEHOLDER


def test_evidence_recall_is_zero_not_one():
    c = no_context.build_context("8k", "q")
    assert evidence_recall([{"chunk_ids": ["doc_001_c01"]}], c) == 0.0


def test_expand_tasks_single_task_per_question_without_k():
    qs = [dict(qid=f"q{i}", type="T1", first_corpus="8k", set="A", anchor=True, fact_ids=["f"]) for i in range(2)]
    spec = dict(runs=[dict(sizes=["8k"], methods=["nocontext", "bm25"], question_types=["T1"], n_per_type=2,
                           repeats=1, ks=[4, 8], question_set="A")])
    tasks = run.expand_tasks(spec, qs, 42)
    nc = [t for t in tasks if t["method"] == "nocontext"]
    assert len(nc) == 2 and all(t["k"] is None and t["k_label"] is None for t in nc)
    assert len([t for t in tasks if t["method"] == "bm25"]) == 4          # ks=[4, 8] у RAG по-прежнему разворачивается


def test_execute_runs_without_llm(monkeypatch):
    class R:
        model = "m"; model_api = None; prompt_tokens = 10; completion_tokens = 5; cached_tokens = 0
        latency_ms = 1.0; cost_usd = 0.0; from_local_cache = False

    class A:
        text = answer = "В предоставленных документах ответа нет."; doc_ids = []; quote = ""
        contradiction_found = False; parsed_ok = True; abstained = True; resp = R()

    seen = {}
    monkeypatch.setattr(run, "answer_question", lambda ctx, q, salt="": seen.setdefault("ctx", ctx) and A())
    monkeypatch.setattr(run, "judge", lambda *a, **k: ("incorrect", "нет ответа", R()))
    q = dict(qid="q1", type="T1", set="A", anchor=True, fact_ids=["f1"], question="?", gold_answer="42")
    gold = {"f1": dict(chunk_ids=["doc_001_c01"], positions_pct={}, doc_id="doc_001")}
    row = run.execute(dict(size="8k", method="nocontext", k=None, k_label=None, q=q, repeat=1, question_set="A"), gold)
    assert seen["ctx"] == no_context.PLACEHOLDER
    assert row["method"] == "nocontext" and row["evidence_recall"] == 0.0 and row["verdict"] == "incorrect"
