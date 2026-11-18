"""Единый LLM-клиент. Весь остальной код ходит в модель ТОЛЬКО через complete() и embed().

Backend выбирается переменной LLM_BACKEND=openai|local. Ни один другой файл
не знает, какой backend активен.

  openai — OpenAI API (платно, под cost_guard)
  local  — Ollama / vLLM с OpenAI-совместимым сервером (бесплатно, для отладки)

Защита от тихой обрезки контекста: Ollama молча режет промпт, если он длиннее окна.
Для бенчмарка это катастрофа (LC получит неполный корпус и результат будет невалиден),
поэтому клиент проверяет длину ДО вызова и сверяет prompt_tokens ПОСЛЕ.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass

from . import cache, cost_guard
from .config import backend_cfg, backend_name, base_cfg

FIT_RATIO = 0.95         # до вызова: оценка токенов не должна превышать 95% окна
TRUNC_RATIO = 0.80       # после вызова: prompt_tokens < 80% оценки => контекст обрезан
MIN_TOKENS_TO_CHECK = 2000


class ContextTooLong(RuntimeError):
    """Промпт заведомо не влезает в окно модели."""


class ContextTruncated(RuntimeError):
    """Провайдер принял меньше токенов, чем мы отправили (тихая обрезка)."""


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
    model_api: str | None = None     # id модели, который вернул сам API (у gpt-6-luna нет датированного снапшота)


_client = None
_enc = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        if backend_name() == "openai":
            _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=8)
        else:
            _client = OpenAI(
                base_url=os.getenv("LOCAL_BASE_URL", "http://localhost:11434/v1"),
                api_key=os.getenv("LOCAL_API_KEY", "EMPTY"),
            )
    return _client


def estimate_tokens(messages: list[dict]) -> int | None:
    """Оценка длины промпта токенайзером из конфига. None, если токенайзер недоступен.

    Для русского токенайзер Qwen тратит токенов БОЛЬШЕ, чем o200k, поэтому оценка
    — это нижняя граница реальной длины.
    """
    global _enc
    try:
        if _enc is None:
            import tiktoken
            _enc = tiktoken.get_encoding(backend_cfg()["tokenizer"])
        return sum(len(_enc.encode(m.get("content") or "")) for m in messages)   # у assistant с tool_calls content может быть None
    except Exception:
        return None


def check_fits(estimate: int | None, window: int) -> None:
    if estimate is not None and estimate > window * FIT_RATIO:
        raise ContextTooLong(
            f"Промпт ~{estimate} токенов (нижняя оценка) не влезает в окно {window}. "
            f"Уменьши корпус или смени backend/модель.")


def check_not_truncated(actual: int, estimate: int | None) -> None:
    if estimate is not None and estimate >= MIN_TOKENS_TO_CHECK and actual < estimate * TRUNC_RATIO:
        raise ContextTruncated(
            f"Отправлено ~{estimate} токенов, провайдер принял {actual}: контекст обрезан. "
            f"Проверь num_ctx в Modelfile / --max-model-len в vLLM.")


def compute_cost(prompt_tokens: int, cached_tokens: int, completion_tokens: int, p: dict | None = None) -> float:
    p = p or backend_cfg()["price_per_1m_usd"]
    inp, cin, out = p["input"], p["cached_input"], p["output"]
    lc = p.get("long_context")          # повышенный тариф: prompt_tokens > threshold => вход и выход дороже
    if lc and prompt_tokens > lc["threshold"]:
        inp, cin, out = inp * lc["input_mult"], cin * lc["input_mult"], out * lc["output_mult"]
    fresh = prompt_tokens - cached_tokens
    return (fresh * inp + cached_tokens * cin + completion_tokens * out) / 1_000_000



def _limit_kw(max_tokens, temperature):
    """Параметры генерации под бэкенд.
    OpenAI (gpt-6-*): только max_completion_tokens, temperature поддерживается лишь по умолчанию (1) — не передаём.
    Локальный Ollama: max_tokens и temperature как в конфиге."""
    if backend_name() == "local":
        return {"max_tokens": max_tokens, "temperature": temperature}
    return {"max_completion_tokens": max_tokens}


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
    if role == "judge":
        model = cfg["judge_model"]
        cfg = {**cfg, "price_per_1m_usd": base_cfg()["model_prices"].get(model, cfg["price_per_1m_usd"])} if backend_name() == "openai" else cfg
    else:
        model = cfg["chat_model"]

    key = cache.make_key({"b": backend_name(), "m": model, "msgs": messages,
                          "t": temperature, "mt": max_tokens, "salt": salt})
    hit = cache.get(key)
    if hit is not None:
        hit["from_local_cache"] = True
        hit["cost_usd"] = 0.0       # повторно не платим; оригинальная цена в raw-логе первого прогона
        return LLMResponse(**hit)

    estimate = estimate_tokens(messages)
    check_fits(estimate, cfg["context_window"])     # падаем ДО долгого/платного вызова

    if backend_name() == "openai":
        cost_guard.check()

    t0 = time.perf_counter()
    r = _get_client().chat.completions.create(
        model=model, messages=messages, **_limit_kw(max_tokens, temperature))
    latency_ms = (time.perf_counter() - t0) * 1000

    u = r.usage
    details = getattr(u, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
    cost = compute_cost(u.prompt_tokens, cached, u.completion_tokens, cfg["price_per_1m_usd"])

    if backend_name() == "openai":
        cost_guard.add(cost)        # деньги потрачены в любом случае
    check_not_truncated(u.prompt_tokens, estimate)  # обрезанный ответ в кэш не попадёт

    resp = LLMResponse(
        text=r.choices[0].message.content or "",
        prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
        cached_tokens=cached, latency_ms=latency_ms, cost_usd=cost,
        model=model, backend=backend_name(), model_api=getattr(r, "model", None))
    cache.put(key, asdict(resp))
    return resp


@dataclass
class ToolResponse:
    resp: LLMResponse            # text = content ответа (пустой, если модель вызвала инструменты)
    tool_calls: list             # [{"id", "name", "arguments"}]; arguments — сырая JSON-строка от модели


def complete_tools(
    messages: list[dict],
    tools: list[dict],
    *,
    tool_choice: str = "auto",   # "auto" | "none" (принудительно финальный ответ без инструментов)
    max_tokens: int | None = None,
    salt: str = "",
) -> ToolResponse:
    """Один шаг агентного цикла (function calling). Те же правила, что у complete(): кэш по хэшу, защита от
    обрезки контекста, cost_guard, учёт usage. Вызывается только из src/methods/agent.py."""
    cfg, gen = backend_cfg(), base_cfg()["generation"]
    temperature = gen["temperature"]
    max_tokens = gen["max_tokens"] if max_tokens is None else max_tokens
    model = cfg["chat_model"]

    key = cache.make_key({"b": backend_name(), "m": model, "msgs": messages, "tools": tools, "tc": tool_choice, "re": base_cfg().get("agent", {}).get("openai_reasoning_effort"),
                          "t": temperature, "mt": max_tokens, "salt": salt})
    hit = cache.get(key)
    if hit is not None:
        r = dict(hit["resp"])
        r["from_local_cache"] = True
        r["cost_usd"] = 0.0
        return ToolResponse(LLMResponse(**r), hit["tool_calls"])

    estimate = estimate_tokens(messages)
    check_fits(estimate, cfg["context_window"])
    if backend_name() == "openai":
        cost_guard.check()

    t0 = time.perf_counter()
    extra = {}
    eff = base_cfg().get("agent", {}).get("openai_reasoning_effort")
    if backend_name() == "openai" and eff:
        extra["reasoning_effort"] = eff
    r = _get_client().chat.completions.create(
        model=model, messages=messages, tools=tools, tool_choice=tool_choice,
        **_limit_kw(max_tokens, temperature), **extra)
    latency_ms = (time.perf_counter() - t0) * 1000

    u = r.usage
    details = getattr(u, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
    cost = compute_cost(u.prompt_tokens, cached, u.completion_tokens, cfg["price_per_1m_usd"])
    if backend_name() == "openai":
        cost_guard.add(cost)
    check_not_truncated(u.prompt_tokens, estimate)

    msg = r.choices[0].message
    calls = [{"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments or ""}
             for tc in (getattr(msg, "tool_calls", None) or [])]
    resp = LLMResponse(
        text=msg.content or "", prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
        cached_tokens=cached, latency_ms=latency_ms, cost_usd=cost,
        model=model, backend=backend_name(), model_api=getattr(r, "model", None))
    cache.put(key, {"resp": asdict(resp), "tool_calls": calls})
    return ToolResponse(resp, calls)


_local_embedder = None


def embed(texts: list[str]) -> list[list[float]]:
    """EMBED_BACKEND=openai|local переопределяет backend только для эмбеддингов
    (удобно: генерация локально, эмбеддинги text-embedding-3-small за копейки)."""
    which = os.getenv("EMBED_BACKEND", backend_name()).lower()
    cfg = base_cfg()["backends"][which]
    if which == "openai":
        cost_guard.check()
        out, total = [], 0
        for i in range(0, len(texts), 256):
            r = _get_openai_embed_client().embeddings.create(model=cfg["embed_model"], input=texts[i:i + 256])
            total += r.usage.total_tokens
            out += [d.embedding for d in r.data]
        cost_guard.add(total * cfg["price_per_1m_usd"]["embed"] / 1_000_000)
        return out

    global _local_embedder
    if _local_embedder is None:
        from sentence_transformers import SentenceTransformer
        _local_embedder = SentenceTransformer(cfg["embed_model"])
    return _local_embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


_embed_client = None


def _get_openai_embed_client():
    global _embed_client
    if _embed_client is None:
        from openai import OpenAI
        _embed_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=8)
    return _embed_client
