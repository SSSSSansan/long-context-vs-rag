import pytest

from scripts import estimate_cost as ec


def test_cost_matches_plan_numbers_for_luna():
    # PLAN_V2 §6.3: LC 8K на Luna ~ $0.00088 за первый вызов, с полным кэшем ~ $0.000142
    assert ec.cost("gpt-6-luna", 8_204, 120) == pytest.approx(0.00088, rel=0.02)
    assert ec.cost("gpt-6-luna", 8_204, 120, 1.0) == pytest.approx(0.000142, rel=0.02)


def test_long_context_tier_applies_above_threshold_only():
    base = ec.cost("gpt-6-luna", 272_000, 1_000)
    over = ec.cost("gpt-6-luna", 272_001, 1_000)
    # вход x2, выход x1.5 -> скачок заметно больше одного токена
    assert over == pytest.approx((272_001 * 0.10 * 2 + 1_000 * 0.50 * 1.5) / 1e6, rel=1e-9)
    assert over > base * 1.9


def test_ladder_has_eight_steps_and_is_cumulative_positive():
    rows = ec.ladder()
    assert [r["name"].split()[0] for r in rows] == [f"Ш{i}" for i in range(1, 9)]
    assert all(r["nocache"] >= r["cached"] >= 0 for r in rows)
    assert rows[0]["n"] == 108 and rows[1]["n"] == 864 and rows[7]["n"] == 900


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        ec.cost("gpt-9-nope", 1000, 10)


def test_confirmed_prices_come_from_base_yaml_not_draft():
    _, draft = ec.get_prices("gpt-6-luna")
    assert draft is False
    assert ec.get_prices("gpt-6.1-sol")[0]["output"] == 10.00
