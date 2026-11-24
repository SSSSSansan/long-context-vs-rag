"""Self-Route (Li et al., arXiv 2407.16833) и роутер по типу вопроса. Только логика выбора; поиск и ответ — существующие методы.

Self-Route: RAG отвечает первым; если он отказался («В предоставленных документах ответа нет.»), берётся ответ LC на тот же вопрос.
Стоимость = вызов RAG + (если отказ) вызов LC. Роутер по типу (oracle): T1 -> RAG, T2/T3 -> LC (тип вопроса известен по построению).

Функции работают с таблицами строк прогона (колонки qid, size, repeat, correct, abstained, cost_nocache_usd, type) и ничего не вызывают в сети:
используются и для симуляции по сохранённым ответам (scripts/simulate_routing.py), и могут использоваться реальным прогоном.
"""
from __future__ import annotations

import pandas as pd

KEYS = ["qset", "size", "qid", "repeat"]
ROUTER_RAG_TYPES = ("T1",)          # типы, которые роутер направляет в RAG; остальное — в LC


def pair(rag: pd.DataFrame, lc: pd.DataFrame) -> pd.DataFrame:
    """Склеивает ответы RAG и LC на один и тот же вопрос и повтор (пары по номеру повтора). Дубликаты ключа отбрасываются (первый)."""
    cols = ["correct", "abstained", "cost_nocache_usd"]
    r = rag.drop_duplicates(KEYS).set_index(KEYS)[cols + ["type"]]
    c = lc.drop_duplicates(KEYS).set_index(KEYS)[cols]
    p = r.join(c, lsuffix="_rag", rsuffix="_lc", how="inner").reset_index()
    p["abstained_rag"] = p.abstained_rag.fillna(False).astype(bool)
    return p


def strategies(p: pd.DataFrame) -> pd.DataFrame:
    """Для каждой пары — исход (correct, cost) под четырьмя стратегиями: rag, lc, self_route, router."""
    out = p[KEYS + ["type"]].copy()
    out["correct_rag"], out["cost_rag"] = p.correct_rag, p.cost_nocache_usd_rag
    out["correct_lc"], out["cost_lc"] = p.correct_lc, p.cost_nocache_usd_lc
    fb = p.abstained_rag
    out["fallback"] = fb
    out["correct_self_route"] = p.correct_rag.where(~fb, p.correct_lc)
    out["cost_self_route"] = p.cost_nocache_usd_rag + p.cost_nocache_usd_lc.where(fb, 0.0)   # RAG платится всегда, LC — только при отказе
    to_rag = p.type.isin(ROUTER_RAG_TYPES)
    out["correct_router"] = p.correct_rag.where(to_rag, p.correct_lc)
    out["cost_router"] = p.cost_nocache_usd_rag.where(to_rag, p.cost_nocache_usd_lc)
    return out


def summarize_strategies(s: pd.DataFrame) -> dict[str, dict[str, float]]:
    """accuracy, средняя стоимость вызова, $/correct и доля отказов-фолбэков по стратегиям."""
    res = {}
    for name in ("rag", "lc", "self_route", "router"):
        acc, cost = float(s[f"correct_{name}"].mean()), float(s[f"cost_{name}"].mean())
        res[name] = {"accuracy": acc, "cost_per_call": cost, "usd_per_correct": cost / acc if acc > 0 else float("nan")}
    res["self_route"]["fallback_share"] = float(s.fallback.mean())
    return res
