import pandas as pd
import pytest

from src.eval import aggregate as ag
from src.eval import run


def Q(qid, qtype, first, st, anchor):
    return dict(qid=qid, type=qtype, first_corpus=first, set=st, anchor=anchor, fact_ids=["f"])


QS = [Q("qa1", "T1", "8k", "A", True), Q("qa2", "T1", "8k", "A", True),
      Q("qh1", "T1", "8k", "A", False), Q("qh2", "T1", "8k", "A", False),
      Q("qb1", "T1", "30k", "B", True), Q("qb2", "T1", "100k", "B", True)]


def ids(qs):
    return sorted(q["qid"] for q in qs)


def test_filter_sets():
    assert ids(run.filter_set(QS, "A")) == ["qa1", "qa2"]
    assert ids(run.filter_set(QS, "A-hard")) == ["qh1", "qh2"]
    assert ids(run.filter_set(QS, "B")) == ["qb1", "qb2"]


def test_missing_question_set_is_error_when_data_has_sets():
    with pytest.raises(ValueError):
        run.filter_set(QS, None)
    with pytest.raises(ValueError):
        run.filter_set(QS, "C")


def test_v1_data_without_sets_still_works():
    v1 = [dict(qid="q1", type="T1", first_corpus="8k"), dict(qid="q2", type="T1", first_corpus="30k")]
    assert len(run.filter_set(v1, None)) == 2


def test_pick_questions_respects_set_and_size():
    assert ids(run.pick_questions(QS, "8k", "T1", 99, 42, "A")) == ["qa1", "qa2"]
    assert run.pick_questions(QS, "8k", "T1", 99, 42, "B") == []          # B на 8K не отвечаем
    assert ids(run.pick_questions(QS, "30k", "T1", 99, 42, "B")) == ["qb1"]
    assert ids(run.pick_questions(QS, "100k", "T1", 99, 42, "B")) == ["qb1", "qb2"]


def test_expand_tasks_carries_question_set_and_keys_differ():
    spec = {"runs": [{"sizes": ["8k"], "methods": ["lc"], "question_set": ["A", "A-hard"],
                      "question_types": ["T1"], "n_per_type": 9, "repeats": 1}]}
    tasks = run.expand_tasks(spec, QS, 42)
    assert sorted(t["question_set"] for t in tasks) == ["A", "A", "A-hard", "A-hard"]
    assert len({run.task_key(t) for t in tasks}) == 4


def test_task_key_and_row_key_include_question_set():
    t = dict(q=QS[0], size="8k", method="lc", k_label=None, repeat=1, question_set="A")
    row = dict(qid="qa1", size="8k", method="lc", k_label=None, repeat=1, question_set="A")
    assert run.task_key(t) == run.row_key(row)
    assert run.task_key({**t, "question_set": "A-hard"}) != run.row_key(row)


def test_resume_of_v1_rows_still_matches():
    t = dict(q=dict(qid="q1"), size="8k", method="bm25", k_label=8, repeat=1)
    row = dict(qid="q1", size="8k", method="bm25", k_label=8, repeat=1)       # строка V1 без question_set
    assert run.task_key(t) == run.row_key(row)


def test_aggregate_qset_label():
    df = pd.DataFrame([
        dict(set=None, anchor=None, question_set=None),
        dict(set="A", anchor=True, question_set=None),
        dict(set="A", anchor=False, question_set=None),
        dict(set="B", anchor=True, question_set=None),
        dict(set="A", anchor=False, question_set="A-hard"),
    ])
    assert list(df.apply(ag._qset, axis=1)) == ["V1", "A", "A-hard", "B", "A-hard"]
