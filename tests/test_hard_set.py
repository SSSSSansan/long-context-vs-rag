"""Шаг 1.3: набор A-hard — связки, дистракторы, вопросы без имени сущности."""
import random
import re

import pytest

from src import hard_set as hs
from src import plant_facts as pf

OFFICES = [f"Центр {w}" for w in
           ("науки", "качества", "карьеры", "развития", "языка", "данных", "спорта", "культуры", "права", "труда",
            "связей", "оценки")]
LEVELS = {"8k": {"T1": 6, "T2": 6, "T3": 6}, "30k": {"T1": 7, "T2": 7, "T3": 7}}
ELIG = {"8k": ["doc_001", "doc_002"], "30k": ["doc_005", "doc_006", "doc_007"]}


def build(dist=1, style="full", seed="t"):
    facts, qs = pf.plan(LEVELS, ELIG, random.Random(seed), "8k")
    recs, hq = hs.plan_hard(facts, qs, ELIG["8k"], OFFICES, random.Random(seed + "h"), style, dist)
    return facts, qs, recs, hq


@pytest.fixture(scope="module", params=[0, 1])
def data(request):
    return build(request.param) + (request.param,)


def by_role(recs, role):
    return [r for r in recs if r["role"] == role]


def test_counts(data):
    facts, qs, recs, hq, dist = data
    assert len(by_role(recs, "link")) == 30                    # 6 T1 + 12 T2 + 12 T3 (по связке на каждый факт набора A)
    assert len(by_role(recs, "distractor_fact")) == (24 if dist else 0)
    assert len(by_role(recs, "distractor_link")) == (24 if dist else 0)
    assert len(hq) == 18 and all(q["qid"].endswith("h") and q["anchor"] is False and q["set"] == "A" for q in hq)
    assert all(q.get("anchor", True) is True for q in qs)      # исходные вопросы не тронуты


def test_features_find_exactly_one_entity(data):
    facts, qs, recs, hq, dist = data
    links = [r for r in recs if r["role"] in ("link", "distractor_link")]
    for q in hq:
        assert len(q["features"]) == (2 if q["type"] == "T2" else 1)
        want = {f["name"] for f in (x for x in facts if x["fact_id"] in q["fact_ids"])}
        for ft in q["features"]:
            found = hs.entities_matching(links, ft["kind"], ft)
            assert len(found) == 1 and found <= want, (q["qid"], found)
        assert len({f["name"] for f in facts if f["fact_id"] in q["fact_ids"]}) == len(q["features"])


def test_distractor_differs_in_exactly_one_feature(data):
    facts, qs, recs, hq, dist = data
    links = {r["fact_id"]: r for r in recs if r["role"] == "link"}
    for d in by_role(recs, "distractor_link"):
        q = next(x for x in hq if x["parent_qid"] == d["for_qid"])
        target = [f for f in q["features"]
                  if f["kind"] == d["kind"] and sum(f[k] == d[k] for k in hs.FEATURES) == 2]
        assert target, d["fact_id"]
        assert d["altered"] in hs.FEATURES and all(d[k] != target[0][k] for k in [d["altered"]])


def test_question_text_has_no_names_but_all_features(data):
    facts, qs, recs, hq, dist = data
    names = {r["name"] for r in facts + recs}
    for q in hq:
        assert not any(n in q["question"] for n in names), q["qid"]
        for ft in q["features"]:
            assert str(ft["year"]) in q["question"] and ft["audience"] in q["question"] and ft["office"] in q["question"]
        assert len(re.findall(r"курируем", q["question"])) == len(q["features"])      # T2: два описания
        assert q["gold_answer"]


def test_names_unique_and_not_in_gold(data):
    facts, qs, recs, hq, dist = data
    gold_names = {f["name"] for f in facts}
    dn = [r["name"] for r in by_role(recs, "distractor_fact")]
    assert len(dn) == len(set(dn)) and not (set(dn) & gold_names)


def test_office_feature_never_equals_fact_value(data):
    facts, qs, recs, hq, dist = data
    link = {r["of"]: r for r in by_role(recs, "link")}
    for f in facts:
        if f["attr"] == "office" and f["fact_id"] in link:
            assert link[f["fact_id"]]["office"] != f["value"]
    for d in by_role(recs, "distractor_fact"):
        if d["attr"] == "office":
            dl = next(r for r in recs if r.get("of") == d["fact_id"])
            assert dl["office"] != d["value"]


def test_same_entity_has_same_features_and_unique_statements(data):
    facts, qs, recs, hq, dist = data
    links = by_role(recs, "link")
    per = {}
    for r in links:
        per.setdefault(r["name"], set()).add(tuple(r[k] for k in hs.FEATURES))
    assert all(len(v) == 1 for v in per.values())
    st = [r["statement"] for r in links]
    assert len(st) == len(set(st))                              # T3: два утверждения об одной сущности -> разные строки


@pytest.mark.parametrize("style", ["full", "compact"])
def test_link_mentions_all_features(style):
    feats = dict(year=2019, audience="аспирантов", office="Центр науки")
    for v in (0, 1):
        s = hs.link_statement("olympiad", "Ориаль-51", feats, style, v) if "olympiad" in pf.KINDS else \
            hs.link_statement(next(iter(pf.KINDS)), "Ориаль-51", feats, style, v)
        assert "Ориаль-51" in s and "2019" in s and "аспирантов" in s and "Центр науки" in s


def test_all_pair_templates_have_marker():
    for attr, spec in pf.ATTRS.items():
        for d in ("more", "less"):
            if f"q_{d}" in spec:
                assert hs.PAIR_MARK in spec[f"q_{d}"], (attr, d)


def test_participle_agreement():
    feats = dict(year=2019, audience="аспирантов", office="Центр науки")
    kind = next(iter(pf.KINDS))
    for case in pf.CASES:
        s = hs.describe(feats, case, kind)
        assert f"учреждённ{hs.PARTICIPLE[case]}" in s and f"курируем{hs.PARTICIPLE[case]}" in s


def test_plant_places_fact_and_link_adjacent():
    docs = {"doc_001": "\n".join(f"1.{i}. Фоновая строка номер {i}." for i in range(1, 30)) + "\n"}
    rng = random.Random(1)
    out = pf.insert_clauses(docs["doc_001"], 1, rng, ["ФАКТ.", "СВЯЗКА."])
    lines = out.split("\n")
    i = next(n for n, l in enumerate(lines) if "ФАКТ." in l)
    assert "СВЯЗКА." in lines[i + 1]


def test_injected_share_decreases_with_size_and_dist_adds_rows():
    _, _, r0, _ = build(0)
    _, _, r1, _ = build(1)
    assert len(r1) - len(r0) == 48


def test_office_questions_distinguish_curator_from_target():
    facts, qs, recs, hq = build(0)
    by_id = {f["fact_id"]: f for f in facts}
    office_qs = [q for q in hq if by_id[q["fact_ids"][0]]["attr"] == "office"]
    assert office_qs
    for q in office_qs:
        assert "(куратор)" in q["question"]
        assert "другое подразделение, не куратор" in q["question"] or "(не кураторе)" in q["question"]
    other = [q for q in hq if by_id[q["fact_ids"][0]]["attr"] != "office"]
    assert all("куратор)" not in q["question"] for q in other)       # остальные вопросы не менялись
