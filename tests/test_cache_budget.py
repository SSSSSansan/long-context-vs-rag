import pytest

from src import cache, cost_guard


def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "_dir", lambda: tmp_path)
    k = cache.make_key({"a": 1, "b": "привет"})
    assert cache.get(k) is None
    cache.put(k, {"x": 1})
    assert cache.get(k) == {"x": 1}


def test_cache_key_differs_by_salt():
    assert cache.make_key({"p": "q", "salt": "1"}) != cache.make_key({"p": "q", "salt": "2"})


def test_budget_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(cost_guard, "_path", lambda: tmp_path / "budget.json")
    monkeypatch.setenv("BUDGET_USD_LIMIT", "0.01")
    cost_guard.check()
    cost_guard.add(0.02)
    with pytest.raises(cost_guard.BudgetExceeded):
        cost_guard.check()


def test_budget_requires_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(cost_guard, "_path", lambda: tmp_path / "budget.json")
    monkeypatch.delenv("BUDGET_USD_LIMIT", raising=False)
    with pytest.raises(cost_guard.BudgetExceeded):
        cost_guard.check()
