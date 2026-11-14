import json
import random

import pytest

from src import build_corpus as b
from src import plant_facts as pf

LEVELS = {"8k": {"T1": 6, "T2": 6, "T3": 6}, "30k": {"T1": 7, "T2": 7, "T3": 7},
          "100k": {"T1": 7, "T2": 7, "T3": 7}}
ELIG = {k: [f"doc_{i:03d}" for i in range(1, 6)] for k in LEVELS}


def make_plan(seed="s"):
    return pf.plan(LEVELS, ELIG, random.Random(seed))


def test_good_numbers_have_single_form():
    for n in pf.good_numbers(5, 200):
        assert n % 10 in (0, 5, 6, 7, 8, 9) or 11 <= n % 100 <= 14


def test_counts():
    facts, questions = make_plan()
    assert len(questions) == 60 and len(facts) == 100
    from collections import Counter
    assert Counter(q["type"] for q in questions) == {"T1": 20, "T2": 20, "T3": 20}
    assert Counter(q["first_corpus"] for q in questions) == {"8k": 18, "30k": 21, "100k": 21}
    assert Counter(f["first_corpus"] for f in facts) == {"8k": 30, "30k": 35, "100k": 35}


def test_deterministic():
    assert make_plan("x") == make_plan("x")


def test_pairs_are_in_different_docs_and_differ():
    facts, _ = make_plan()
    by_id = {f["fact_id"]: f for f in facts}
    for f in facts:
        if f["type"] in ("T2", "T3"):
            o = by_id[f["paired_with"]]
            assert f["doc_id"] != o["doc_id"]
            assert f["value"] != o["value"]
            if f["type"] == "T3":
                assert f["name"] == o["name"] and f["attr"] == o["attr"]
                assert f["variant"] != o["variant"]
            else:
                assert f["name"] != o["name"] and f["attr"] == o["attr"]


def test_entity_names_unique_across_questions():
    facts, _ = make_plan()
    names = [f["name"] for f in facts if f["type"] != "T3"] + \
            [f["name"] for f in facts if f["type"] == "T3" and f["fact_id"].endswith("a")]
    assert len(names) == len(set(names))


def test_t2_answer_matches_direction():
    facts, questions = make_plan()
    by_id = {f["fact_id"]: f for f in facts}
    for q in (x for x in questions if x["type"] == "T2"):
        a, bb = (by_id[i] for i in q["fact_ids"])
        spec = pf.ATTRS[a["attr"]]
        more = spec["q_more"].format(A=a["name"], B=bb["name"])
        bigger = a if a["value"] > bb["value"] else bb
        smaller = bb if bigger is a else a
        winner = bigger if q["question"] == more else smaller
        assert q["gold_answer"].startswith(f"«{winner['name']}»")


def test_statements_contain_entity_and_value():
    facts, _ = make_plan()
    for f in facts:
        assert f["name"] in f["statement"]
        assert str(f["value"]) in f["statement"]
        assert "{" not in f["statement"]


def test_insert_clause_renumbers():
    text = "ДОКУМЕНТ doc_001\n\nРаздел 2. X\n2.1. aaa.\n2.2. bbb.\n2.3. ccc.\n\nРаздел 3. Y\n3.1. ddd.\n"
    out = pf.insert_clause(text, 2, random.Random(1), "НОВЫЙ ПУНКТ.")
    lines = [l for l in out.split("\n") if l.startswith("2.")]
    assert [l.split(". ")[0] for l in lines] == ["2.1", "2.2", "2.3", "2.4"]
    assert out.count("НОВЫЙ ПУНКТ.") == 1
    assert "3.1. ddd." in out                         # соседний раздел не тронут


def test_eligible_docs_regions():
    sizes = {"8k": 8000, "30k": 30000, "100k": 100000}
    offsets = {f"doc_{i:03d}": ((i - 1) * 2400, i * 2400) for i in range(1, 49)}
    e8 = pf.eligible_docs(offsets, sizes, "8k")
    e30 = pf.eligible_docs(offsets, sizes, "30k")
    assert e8 == ["doc_001", "doc_002"]
    assert min(e30) >= "doc_004" and all(offsets[d][1] <= 27000 for d in e30)
    assert not set(e8) & set(e30)


def test_finalize_gold_checks_presence(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    stmt = "1.2. Факт про «Тавро-47» составляет 10 МРП."
    stmt_plain = "Факт про «Тавро-47» составляет 10 МРП."
    doc = "ДОКУМЕНТ doc_001\nЗаголовок\n\n1.1. Фон.\n" + stmt + "\n1.3. Фон.\n"
    corpora = {"8k": doc, "30k": doc + "\nДОКУМЕНТ doc_002\nФон\n", "100k": doc + "\nДОКУМЕНТ doc_002\nФон\n"}
    gold = {"facts": [{"fact_id": "f1", "statement": stmt_plain, "first_corpus": "8k"}]}
    (tmp_path / "data" / "gold.json").write_text(json.dumps(gold, ensure_ascii=False), encoding="utf-8")
    b.finalize_gold(corpora, count=lambda t: len(t) // 4)
    f = json.loads((tmp_path / "data" / "gold.json").read_text(encoding="utf-8"))["facts"][0]
    assert f["corpus_sizes"] == ["8k", "30k", "100k"] and f["chunk_ids"] == ["doc_001_c00"]
    # факт пропал из 8k — должна быть ошибка
    corpora["8k"] = "ДОКУМЕНТ doc_001\nЗаголовок\n"
    (tmp_path / "data" / "gold.json").write_text(json.dumps(gold, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(RuntimeError):
        b.finalize_gold(corpora, count=lambda t: len(t) // 4)


def test_sets_core_is_A_rest_is_B():
    from collections import Counter
    facts, questions = make_plan()
    assert Counter(q["set"] for q in questions) == {"A": 18, "B": 42}
    assert Counter((q["set"], q["type"]) for q in questions)[("A", "T1")] == 6
    assert all((q["set"] == "A") == (q["first_corpus"] == "8k") for q in questions)
    assert all(q["anchor"] is True for q in questions)       # вопросы без якоря (A-hard) добавляются отдельным шагом
    by_id = {f["fact_id"]: f for f in facts}
    for q in questions:                                      # у факта тот же набор, что у его вопроса
        assert all(by_id[i]["set"] == q["set"] for i in q["fact_ids"])


def test_core_level_is_configurable():
    _, questions = pf.plan(LEVELS, ELIG, random.Random("s"), core_level="30k")
    assert {q["set"] for q in questions if q["first_corpus"] == "30k"} == {"A"}
    assert {q["set"] for q in questions if q["first_corpus"] == "8k"} == {"B"}


def test_core_questions_answerable_everywhere_check():
    keys = ["8k", "30k", "100k"]
    facts = [{"fact_id": "a", "corpus_sizes": keys}, {"fact_id": "b", "corpus_sizes": ["30k", "100k"]}]
    ok = [{"qid": "q1", "set": "A", "fact_ids": ["a"]}, {"qid": "q2", "set": "B", "fact_ids": ["b"]}]
    assert b.check_core_everywhere(facts, ok, keys) == 1
    with pytest.raises(RuntimeError):       # набор A, но факт есть не везде
        b.check_core_everywhere(facts, [{"qid": "q3", "set": "A", "fact_ids": ["b"]}], keys)
    with pytest.raises(RuntimeError):       # набор B, но отвечаем везде
        b.check_core_everywhere(facts, [{"qid": "q4", "set": "B", "fact_ids": ["a"]}], keys)
