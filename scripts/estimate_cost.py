"""Смета ДО запуска: лестница прогонов Ш1-Ш8 из docs/PLAN_V2.md §5.2 + судья + точки безубыточности §6.

    python -m scripts.estimate_cost            # или: make estimate
    python -m scripts.estimate_cost --limit 3.00

Цены берутся из configs/base.yaml -> model_prices. Если модели там нет (цены ещё не подтверждены
по https://developers.openai.com/api/docs/pricing), берутся ЧЕРНОВЫЕ цены из DRAFT_PRICES ниже,
и скрипт печатает заметное предупреждение. Как только цены подтверждены и вписаны в base.yaml,
черновики не используются.

Повышенный тариф: если prompt_tokens > long_context.threshold, вход умножается на input_mult,
выход на output_mult (GPT-6: порог 272K, x2 вход, x1.5 выход; Gemini 3 Pro: 200K).
"""
from __future__ import annotations

import argparse
import os

from src.config import base_cfg

# --- ЧЕРНОВЫЕ ЦЕНЫ ($ за 1M токенов). НЕ ПОДТВЕРЖДЕНЫ. Подтвердить на странице цен OpenAI. ---
GPT6_TIER = {"threshold": 272_000, "input_mult": 2.0, "output_mult": 1.5}
DRAFT_PRICES = {
    "gpt-6-luna":    {"input": 0.10, "cached_input": 0.01, "output": 0.50,  "long_context": GPT6_TIER},
    "gpt-6.1-terra": {"input": 1.00, "cached_input": 0.05, "output": 5.00,  "long_context": GPT6_TIER},
    "gpt-6.1-sol":   {"input": 2.00, "cached_input": 0.10, "output": 10.00, "long_context": GPT6_TIER},
}

# --- Токены на вызов, измерены по results/summary_overall.csv (PLAN_V2 §5.1) ---
TOK = {"8k": 8_204, "30k": 30_191, "100k": 100_194, "rag": 2_550, "nc": 400, "agent": 10_000}
OUT = 120                     # измерено ~90-103 токена на ответ; 120 - с запасом
AGENT_OUT = 400
JUDGE_IN, JUDGE_OUT = 1_000, 60
NA, NB, REP, RAGM = 18, 60, 3, 4   # вопросов в наборе A, в наборе B, повторов, RAG-методов (bm25, emb, hyb_rrf, hyb_rerank)
CS_LC, CS_RAG = 0.90, 0.50         # допущение о доле кэша (cached_tokens / prompt_tokens)
MODEL = "gpt-6-luna"


def get_prices(model: str) -> tuple[dict, bool]:
    """(цены, это_черновик). Подтверждённые цены из base.yaml имеют приоритет."""
    confirmed = base_cfg().get("model_prices", {})
    if model in confirmed:
        return confirmed[model], False
    if model in DRAFT_PRICES:
        return DRAFT_PRICES[model], True
    raise KeyError(f"Нет цен для {model}: впиши их в configs/base.yaml -> model_prices")


def cost(model: str, t_in: int, t_out: int, cached_share: float = 0.0) -> float:
    """Стоимость одного вызова, $. Учитывает порог повышенного тарифа."""
    p, _ = get_prices(model)
    i, c, o = p["input"], p.get("cached_input", p["input"]), p["output"]
    lc = p.get("long_context")
    if lc and t_in > lc["threshold"]:
        i, c, o = i * lc["input_mult"], c * lc["input_mult"], o * lc["output_mult"]
    return (t_in * (1 - cached_share) * i + t_in * cached_share * c + t_out * o) / 1e6


def ladder(model: str = MODEL) -> list[dict]:
    """Шаги Ш1-Ш8. Каждый шаг: список (вызовов, токенов_входа, токенов_выхода, доля_кэша)."""
    A2 = 2 * NA * REP                  # набор A + A-hard, REP повторов = 108 вызовов
    steps = [
        ("Ш1 NO-CONTEXT (A + A-hard)",            [(A2, TOK["nc"], OUT, 0.0)]),
        ("Ш2 RAG 8K+30K: BM25/EMB/HYB×2, A+A-hard",  [(RAGM * 2 * A2, TOK["rag"], OUT, CS_RAG)]),
        ("Ш3 LC 8K, A + A-hard",                  [(A2, TOK["8k"], OUT, CS_LC)]),
        ("Ш4 LC 30K, A + A-hard",                 [(A2, TOK["30k"], OUT, CS_LC)]),
        ("Ш5 RAG 100K: BM25/EMB/HYB×2, A+A-hard",    [(RAGM * A2, TOK["rag"], OUT, CS_RAG)]),
        ("Ш6 AGENT (A-hard, 3 индекса)",          [(NA * REP * 3, TOK["agent"], AGENT_OUT, 0.0)]),
        ("Ш7 LC 100K, A + A-hard",                [(A2, TOK["100k"], OUT, CS_LC)]),
        ("Ш8 набор B на 100K: LC + 4 RAG",        [(NB * REP, TOK["100k"], OUT, CS_LC),
                                                   (RAGM * NB * REP, TOK["rag"], OUT, CS_RAG)]),
    ]
    rows = []
    for name, parts in steps:
        n = sum(p[0] for p in parts)
        nocache = sum(c * cost(model, ti, to, 0.0) for c, ti, to, _ in parts)
        cached = sum(c * cost(model, ti, to, cs) for c, ti, to, cs in parts)
        judge = n * cost(model, JUDGE_IN, JUDGE_OUT)
        rows.append(dict(name=name, n=n, nocache=nocache, cached=cached, judge=judge))
    return rows


def print_ladder(model: str = MODEL, limit: float = 0.0, soft: float = 0.0, spent: float = 0.0) -> None:
    _, draft = get_prices(model)
    if draft:
        print("!" * 78)
        print(f"! ЧЕРНОВЫЕ ЦЕНЫ для {model}: их нет в configs/base.yaml -> model_prices.")
        print("! Сверь с https://developers.openai.com/api/docs/pricing и подтверди, иначе смете верить нельзя.")
        print("!" * 78)
    p, _ = get_prices(model)
    lc = p.get("long_context")
    print(f"Модель: {model}   вход/кэш/выход, $/1M: {p['input']} / {p.get('cached_input')} / {p['output']}"
          + (f"   порог повышенного тарифа: {lc['threshold']:,} ток." if lc else ""))
    print(f"Допущения: повторов={REP}, кэш LC={CS_LC:.0%}, кэш RAG={CS_RAG:.0%}, ответ={OUT} ток. "
          f"Судья ({JUDGE_IN}+{JUDGE_OUT} ток.) считается на каждый вызов шага.\n")
    h = f"{'шаг':44s}{'вызовов':>8s}{'без кэша':>10s}{'с кэшем':>9s}{'судья':>8s}{'накопл.макс':>13s}{'накопл.кэш':>12s}"
    print(h)
    print("-" * len(h))
    cum_n = cum_c = 0.0
    fits = ""
    if limit:
        print(f"Уже потрачено (runs/budget_v2.json): ${spent:.4f}; жёсткий лимит ${limit:.2f}; мягкий стоп ${soft:.2f}; "
              f"осталось до лимита ${max(limit - spent, 0):.2f}\n")
    for r in ladder(model):
        cum_n += r["nocache"] + r["judge"]
        cum_c += r["cached"] + r["judge"]
        mark = ""
        if limit:
            mark = ("  <= влезает" if spent + cum_n <= limit else
                    ("  <= влезает только с кэшем" if spent + cum_c <= limit else "  <= НЕ влезает"))
            if soft and spent + cum_n > soft:
                mark += "  [выше мягкого стопа: СПРОСИ у Panda]"
            if spent + cum_n <= limit:
                fits = r["name"].split()[0]
        print(f"{r['name']:44s}{r['n']:8d}{r['nocache']:10.4f}{r['cached']:9.4f}{r['judge']:8.4f}"
              f"{cum_n:13.4f}{cum_c:12.4f}{mark}")
    print(f"\nИТОГО худший случай (без кэша, с судьёй): ${cum_n:.2f}   с кэшем: ${cum_c:.2f}")
    if limit:
        print(f"Лимит бюджета: ${limit:.2f}.  Последний шаг, влезающий без кэша: {fits or 'нет'}")


def print_breakeven(models=("gpt-6-luna", "gpt-6.1-terra")) -> None:
    """N* = (C_full - C_cached) / (C_rag - C_cached): после скольких вопросов к одному корпусу LC с кэшем дешевле RAG."""
    print("\nБезубыточность LC (с кэшем) против RAG, PLAN_V2 §6:")
    for m in models:
        for s in ("8k", "30k", "100k"):
            cf, cc, cr = cost(m, TOK[s], OUT), cost(m, TOK[s], OUT, 1.0), cost(m, TOK["rag"], OUT)
            ns = f"{(cf - cc) / (cr - cc):.1f}" if cc < cr else "нет"
            print(f"  {m:14s}{s:>5s}  LC={cf:.6f}  LC_кэш={cc:.6f}  RAG={cr:.6f}  N*={ns}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--limit", type=float, default=None, help="жёсткий лимит, $ (по умолчанию BUDGET_USD_LIMIT из .env)")
    ap.add_argument("--soft", type=float, default=None, help="мягкий стоп, $ (по умолчанию BUDGET_SOFT_STOP_USD или 2.50)")
    args = ap.parse_args()
    from src import cost_guard
    limit = args.limit if args.limit is not None else (cost_guard.limit_usd() or 3.00)
    soft = args.soft if args.soft is not None else float(os.getenv("BUDGET_SOFT_STOP_USD", "2.50"))
    print_ladder(args.model, limit, soft, cost_guard.spent_usd())
    print_breakeven()


if __name__ == "__main__":
    main()
