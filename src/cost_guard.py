"""Жёсткий лимит бюджета. Счётчик лежит в runs/budget.json и переживает перезапуски.

Считаются только платные вызовы (backend=openai). Локальный backend стоит 0.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from .config import ROOT, base_cfg


_lock = threading.Lock()


class BudgetExceeded(RuntimeError):
    pass


def _path() -> Path:
    return ROOT / base_cfg()["budget"]["budget_file"]


def _read() -> dict:
    p = _path()
    if p.exists():
        return json.loads(p.read_text())
    return {"spent_usd": 0.0, "calls": 0}


def limit_usd() -> float:
    return float(os.getenv("BUDGET_USD_LIMIT", "0"))


def spent_usd() -> float:
    return _read()["spent_usd"]


def check() -> None:
    """Вызывать ПЕРЕД каждым платным запросом."""
    lim = limit_usd()
    if lim <= 0:
        raise BudgetExceeded("BUDGET_USD_LIMIT не задан в .env — платные вызовы заблокированы.")
    if spent_usd() >= lim:
        raise BudgetExceeded(f"Бюджет исчерпан: потрачено ${spent_usd():.4f} из ${lim:.2f}.")


def add(cost_usd: float) -> None:
    """Вызывать ПОСЛЕ запроса с фактической стоимостью."""
    with _lock:
        st = _read()
        st["spent_usd"] = round(st["spent_usd"] + cost_usd, 6)
        st["calls"] += 1
        p = _path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(st, indent=2))


def report() -> str:
    return f"spent ${spent_usd():.4f} / limit ${limit_usd():.2f}"
