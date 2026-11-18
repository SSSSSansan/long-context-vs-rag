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


def test_cache_write_tokens_billed_at_write_rate():
    """18.11: usage.prompt_tokens_details.cache_write_tokens идут по cache_write_input (Luna 0.125), не по input."""
    from src.llm_client import compute_cost
    p = {"input": 0.10, "cached_input": 0.01, "cache_write_input": 0.125, "output": 0.50}
    # реальный ответ API из микро-проверки: 2797 вход, 0 из кэша, 2794 записано, 4 выход
    assert abs(compute_cost(2797, 0, 4, p, 2794) - (3 * 0.10 + 2794 * 0.125 + 4 * 0.50) / 1e6) < 1e-12
    assert abs(compute_cost(2797, 2794, 4, p, 0) - (3 * 0.10 + 2794 * 0.01 + 4 * 0.50) / 1e6) < 1e-12
    assert compute_cost(1000, 0, 0, {"input": 0.1, "cached_input": 0.01, "output": 0.5}, 500) == compute_cost(1000, 0, 0, {"input": 0.1, "cached_input": 0.01, "output": 0.5})
