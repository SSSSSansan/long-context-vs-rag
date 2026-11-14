from src import cost_guard
from src.config import base_cfg


def test_v2_counter_is_separate_from_v1():
    assert base_cfg()["budget"]["budget_file"] == "runs/budget_v2.json"


def test_fresh_v2_counter_allows_paid_calls(tmp_path, monkeypatch):
    # со старым счётчиком V1 ($6.04 >= лимита) check() блокировал бы всё; новый стартует с нуля
    monkeypatch.setattr(cost_guard, "_path", lambda: tmp_path / "budget_v2.json")
    monkeypatch.setenv("BUDGET_USD_LIMIT", "1.40")
    assert cost_guard.spent_usd() == 0.0
    cost_guard.check()
    cost_guard.add(1.39)
    cost_guard.check()
    cost_guard.add(0.02)
    import pytest
    with pytest.raises(cost_guard.BudgetExceeded):
        cost_guard.check()
