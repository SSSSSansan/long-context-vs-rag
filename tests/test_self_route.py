"""Self-Route и роутер: логика выбора и стоимости на игрушечных данных (без сети)."""
import pandas as pd
import pytest

from src.methods.self_route import pair, strategies, summarize_strategies


def _rows(method_cost, correct, abstained, types):
    return pd.DataFrame({"qset": "A", "size": "100k", "qid": [f"q{i}" for i in range(len(correct))], "repeat": 1, "type": types,
                         "correct": correct, "abstained": abstained, "cost_nocache_usd": method_cost})


def test_self_route_falls_back_to_lc_only_on_abstain_and_pays_both():
    rag = _rows(0.001, [1.0, 0.0, 0.0, 1.0], [False, True, False, False], ["T1", "T1", "T2", "T3"])
    lc = _rows(0.01, [1.0, 1.0, 1.0, 0.0], [False] * 4, ["T1", "T1", "T2", "T3"])
    s = strategies(pair(rag, lc))
    r = summarize_strategies(s)
    assert r["rag"]["accuracy"] == pytest.approx(0.5) and r["lc"]["accuracy"] == pytest.approx(0.75)
    # q1 отказ -> ответ LC (верно); q2 не отказ -> остаётся неверный RAG; q3 верный RAG
    assert r["self_route"]["accuracy"] == pytest.approx(0.5 + 0.25)           # q0 + q1
    assert r["self_route"]["fallback_share"] == pytest.approx(0.25)
    assert r["self_route"]["cost_per_call"] == pytest.approx(0.001 + 0.25 * 0.01)
    # роутер: T1 (q0,q1) -> RAG [1,0], T2/T3 (q2,q3) -> LC [1,0]
    assert r["router"]["accuracy"] == pytest.approx(0.5)
    assert r["router"]["cost_per_call"] == pytest.approx((0.001 * 2 + 0.01 * 2) / 4)


def test_pairing_by_repeat_and_dedup():
    rag = pd.concat([_rows(0.001, [1.0], [False], ["T1"]).assign(repeat=1), _rows(0.001, [0.0], [False], ["T1"]).assign(repeat=2)])
    lc = pd.concat([_rows(0.01, [1.0], [False], ["T1"]).assign(repeat=1)] * 2)       # дубликат ключа (пилот + основной прогон)
    p = pair(rag, lc)
    assert len(p) == 1 and p.repeat.iloc[0] == 1                                 # повтор 2 без пары LC не попадает
