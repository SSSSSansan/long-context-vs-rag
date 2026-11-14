import pytest

from src.config import base_cfg
from src.llm_client import LLMResponse, compute_cost

LUNA = base_cfg()["model_prices"]["gpt-6-luna"]


def test_compute_cost_normal_tier():
    # 100K вход, 90% из кэша, 120 выход
    c = compute_cost(100_000, 90_000, 120, LUNA)
    assert c == pytest.approx((10_000 * 0.10 + 90_000 * 0.01 + 120 * 0.50) / 1e6)


def test_compute_cost_long_context_tier():
    c = compute_cost(300_000, 0, 1_000, LUNA)
    assert c == pytest.approx((300_000 * 0.20 + 1_000 * 0.75) / 1e6)


def test_old_prices_without_tier_still_work():
    p = {"input": 0.15, "cached_input": 0.075, "output": 0.60}
    assert compute_cost(1_000_000, 0, 0, p) == pytest.approx(0.15)


def test_old_cache_entries_load_without_model_api():
    r = LLMResponse(text="x", prompt_tokens=1, completion_tokens=1, cached_tokens=0,
                    latency_ms=1.0, cost_usd=0.0, model="m", backend="local")
    assert r.model_api is None
