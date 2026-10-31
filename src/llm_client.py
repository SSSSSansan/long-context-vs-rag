"""Единый LLM-клиент. Весь остальной код ходит в модель ТОЛЬКО через complete() и embed().

Backend выбирается переменной LLM_BACKEND=openai|local. Ни один другой файл
не знает, какой backend активен.

  openai — OpenAI API (платно, под cost_guard)
  local  — vLLM с OpenAI-совместимым сервером (бесплатно, для отладки)
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass

from . import cache, cost_guard
from .config import backend_cfg, backend_name, base_cfg


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int       # токены, взятые из prompt-cache провайдера
    latency_ms: float
    cost_usd: float
    model: str
    backend: str
    from_local_cache: bool = False   # хит нашего дискового кэша (стоимость 0, не тратит бюджет)


_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        if backend_name() == "openai":
            _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        else:
            _client = OpenAI(
                base_url=os.getenv("LOCAL_BASE_URL", "http://localhost:8000/v1"),
                api_key=os.getenv("LOCAL_API_KEY", "EMPTY"),
            )
    return _client


def compute_cost(prompt_tokens: int, cached_tokens: int, completion_tokens: int) -> float:
    p = backend_cfg()["price_per_1m_usd"]
    fresh = prompt_tokens - cached_tokens
    return (fresh * p["input"] + cached_tokens * p["cached_input"]
            + completion_tokens * p["output"]) / 1_000_000


def complete(
    messages: list[dict],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    role: str = "chat",      # "chat" | "judge" — выбирает модель из конфига
    salt: str = "",          # номер повтора, чтобы кэш не склеивал repeat=1,2,3
) -> LLMResponse:
    cfg, gen = backend_cfg(), base_cfg()["generation"]
    temperature = gen["temperature"] if temperature is None else temperature
    max_tokens = gen["max_tokens"] if max_tokens is None else max_tokens
    model = cfg["judge_model"] if role == "judge" else cfg["chat_model"]

    key = cache.make_key({"b": backend_name(), "m": model, "msgs": messages,
                          "t": temperature, "mt": max_tokens, "salt": salt})
    hit = cache.get(key)
    if hit is not None:
        hit["from_local_cache"] = True
        hit["cost_usd"] = 0.0       # повторно не платим; оригинальная цена в raw-логе первого прогона
        return LLMResponse(**hit)

    if backend_name() == "openai":
        cost_guard.check()

    t0 = time.perf_counter()
    r = _get_client().chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    latency_ms = (time.perf_counter() - t0) * 1000

    u = r.usage
    details = getattr(u, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
    cost = compute_cost(u.prompt_tokens, cached, u.completion_tokens)

    resp = LLMResponse(
        text=r.choices[0].message.content or "",
        prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
        cached_tokens=cached, latency_ms=latency_ms, cost_usd=cost,
        model=model, backend=backend_name())

    if backend_name() == "openai":
        cost_guard.add(cost)
    cache.put(key, asdict(resp))
    return resp


_local_embedder = None


def embed(texts: list[str]) -> list[list[float]]:
    cfg = backend_cfg()
    if backend_name() == "openai":
        cost_guard.check()
        r = _get_client().embeddings.create(model=cfg["embed_model"], input=texts)
        cost_guard.add(r.usage.total_tokens * cfg["price_per_1m_usd"]["embed"] / 1_000_000)
        return [d.embedding for d in r.data]

    global _local_embedder
    if _local_embedder is None:
        from sentence_transformers import SentenceTransformer
        _local_embedder = SentenceTransformer(cfg["embed_model"])
    return _local_embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
