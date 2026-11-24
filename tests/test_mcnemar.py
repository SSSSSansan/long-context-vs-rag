"""McNemar в src/eval/aggregate.py: точный тест, поправка Холма, парные таблицы (на синтетике, без API)."""
import pandas as pd
import pytest

from src.eval.aggregate import holm_adjust, mcnemar_exact, mcnemar_tables, question_outcomes


def test_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0                       # нет несогласных пар
    assert mcnemar_exact(3, 3) == pytest.approx(1.0)        # поровну
    assert mcnemar_exact(5, 0) == pytest.approx(2 / 32)     # 0.0625
    assert mcnemar_exact(0, 8) == pytest.approx(2 / 256)    # симметрия
    assert mcnemar_exact(10, 2) == pytest.approx(2 * 79 / 4096)   # (1+12+66)/4096 хвост
    assert mcnemar_exact(1, 0) == 1.0                       # p ограничен сверху единицей


def test_holm_adjust():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.5]) == [0.5]
    assert holm_adjust([0.4, 0.6]) == pytest.approx([0.8, 0.8])   # монотонность и потолок 1.0


def _rows(model, qset, size, method, outcomes, repeats=3):
    """outcomes: список из 0/1 по вопросам; каждый вопрос повторяется repeats раз (последний повтор «шумит»)."""
    rows = []
    for i, ok in enumerate(outcomes):
        for r in range(repeats):
            val = float(ok) if r < repeats - 1 else float(1 - ok)   # 2 из 3 повторов = исход, 1 — шум
            rows.append({"model": model, "qset": qset, "size": size, "method": method,
                         "qid": f"q{i:02d}", "repeat": r, "correct": val})
    return rows


def test_question_outcomes_majority_of_repeats():
    df = pd.DataFrame(_rows("m", "A", "8k", "lc", [1, 0]))
    out = question_outcomes(df)
    assert bool(out[("m", "A", "8k", "lc", "q00")]) is True
    assert bool(out[("m", "A", "8k", "lc", "q01")]) is False


def test_method_pairs_within_size():
    # X верен на всех 18, Y — на 9 из 18 (девять несогласных, все в пользу X): p = 2/2^9
    rows = _rows("m", "A", "8k", "bm25", [1] * 18) + _rows("m", "A", "8k", "emb", [1] * 9 + [0] * 9)
    res = mcnemar_tables(pd.DataFrame(rows))
    r = res[(res.family == "method")].iloc[0]
    assert (r.a, r.b, r.context) == ("bm25", "emb", "8k")
    assert (r.n_pairs, r.only_a, r.only_b, r.both_correct, r.both_wrong) == (18, 9, 0, 9, 0)
    assert r.acc_a == 1.0 and r.acc_b == 0.5
    assert r.p_value == pytest.approx(2 / 512)
    assert r.p_holm == pytest.approx(r.p_value)             # одна пара — поправка не меняет


def test_size_pairs_for_each_method_use_common_questions_only():
    rows = (_rows("m", "A", "8k", "lc", [1] * 18) + _rows("m", "A", "30k", "lc", [1] * 12 + [0] * 6)
            + _rows("m", "A", "100k", "lc", [1] * 6 + [0] * 12))
    res = mcnemar_tables(pd.DataFrame(rows))
    sz = res[res.family == "size"].set_index(["a", "b"])
    assert set(sz.index) == {("8k", "30k"), ("8k", "100k"), ("30k", "100k")}
    assert sz.loc[("8k", "30k")].only_a == 6 and sz.loc[("8k", "30k")].p_value == pytest.approx(2 / 64)
    assert sz.loc[("8k", "100k")].only_a == 12
    # Холм внутри таблицы метода lc: три пары, самое малое p умножается на 3
    pv = sz.p_value.sort_values()
    assert sz.p_holm.max() <= 1.0 and sz.loc[pv.index[0]].p_holm == pytest.approx(min(1.0, pv.iloc[0] * 3))


def test_set_b_and_v1_are_ignored_and_models_do_not_mix():
    rows = (_rows("m1", "B", "100k", "lc", [1] * 10) + _rows("m1", "B", "100k", "bm25", [0] * 10)
            + _rows("m1", "A", "8k", "lc", [1] * 4) + _rows("m1", "A", "8k", "bm25", [1] * 4)
            + _rows("m2", "A", "8k", "lc", [0] * 4) + _rows("m2", "A", "8k", "bm25", [1] * 4))
    res = mcnemar_tables(pd.DataFrame(rows))
    assert set(res.qset) == {"A"}                           # набор B не участвует
    byl = res[res.family == "method"].set_index("model")
    assert byl.loc["m1"].p_value == 1.0                     # m1: оба метода равны
    assert byl.loc["m2"].only_b == 4                        # m2 считается отдельно от m1


def test_empty_when_no_paired_sets():
    df = pd.DataFrame(_rows("m", "B", "100k", "lc", [1, 0]))
    assert mcnemar_tables(df).empty


def test_anchor_family_pairs_by_fact_and_known_p():
    # метод X: с якорем верно на всех 18, без якоря — на 12 (шесть несогласных, все в пользу A): p = 2/2^6
    # метод Y: одинаково (по 18 верно в обоих) — p = 1.0; метод Z — только A-hard (нет пары), пропускается
    from src.eval.aggregate import anchor_tables
    key = {f"q{i:02d}": f"f{i:02d}" for i in range(18)} | {f"q{i:02d}h": f"f{i:02d}" for i in range(18)}
    def rows_h(method, outcomes):
        return [{**r, "qid": r["qid"] + "h"} for r in _rows("m", "A-hard", "30k", method, outcomes)]
    rows = (_rows("m", "A", "30k", "bm25", [1] * 18) + rows_h("bm25", [1] * 12 + [0] * 6)
            + _rows("m", "A", "30k", "lc", [1] * 18) + rows_h("lc", [1] * 18)
            + rows_h("agent_bm25", [1] * 18))
    res = anchor_tables(pd.DataFrame(rows), pair_key=key).set_index("method")
    assert set(res.index) == {"bm25", "lc"}                 # агент без пары пропущен
    b = res.loc["bm25"]
    assert (b.n_pairs, b.only_a, b.only_b, b.both_correct, b.both_wrong) == (18, 6, 0, 12, 0)
    assert (b.acc_a, b.acc_b) == (1.0, pytest.approx(12 / 18))
    assert b.p_value == pytest.approx(2 / 64)
    assert b.p_holm == pytest.approx(2 / 64 * 2)            # Холм: 2 теста в семье, наименьшее p умножается на 2
    assert res.loc["lc"].p_value == 1.0
