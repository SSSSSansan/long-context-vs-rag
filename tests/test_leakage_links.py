import json
from pathlib import Path

import pytest

from scripts.check_leakage import leaks_value, link_only_context, qset

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data():
    g = ROOT / "data/gold.json"
    q = ROOT / "data/questions.json"
    if not g.exists() or not q.exists():
        pytest.skip("данные не собраны")
    facts = {f["fact_id"]: f for f in json.loads(g.read_text(encoding="utf-8"))["facts"]}
    qs = json.loads(q.read_text(encoding="utf-8"))["questions"]
    if not any("anchor" in x and not x["anchor"] for x in qs):
        pytest.skip("данные ещё без A-hard")
    return facts, qs


def test_link_only_context_has_no_fact_statement(data):
    facts, qs = data
    for q in (x for x in qs if not x.get("anchor", True)):
        ctx = link_only_context(q, facts)
        assert all(facts[i]["statement"] not in ctx for i in q["fact_ids"])
        assert all(facts[i]["statement"] in ctx for i in q["link_fact_ids"])
        assert not leaks_value(q, facts), q["qid"]


def test_qset_labels(data):
    _, qs = data
    from collections import Counter
    assert Counter(qset(q) for q in qs) == {"A": 18, "A-hard": 18, "B": 42}
