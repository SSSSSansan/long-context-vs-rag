"""Проверка единого клиента: один вызов + повторный (должен прийти из кэша).

    LLM_BACKEND=local python -m scripts.smoke_llm
"""
from src import cost_guard
from src.llm_client import complete

msgs = [{"role": "user", "content": "Ответь одним словом: столица Казахстана?"}]
a = complete(msgs, max_tokens=20)
b = complete(msgs, max_tokens=20)
print("1-й вызов:", a.text.strip(), f"| in={a.prompt_tokens} out={a.completion_tokens} "
      f"lat={a.latency_ms:.0f}ms cost=${a.cost_usd:.6f}")
print("2-й вызов из кэша:", b.from_local_cache)
print("бюджет:", cost_guard.report())
