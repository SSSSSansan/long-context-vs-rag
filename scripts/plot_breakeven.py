"""График §6.4 PLAN_V2: суммарная стоимость N вопросов к одному корпусу, LC (с кэшем) против RAG.

    python -m scripts.plot_breakeven [--model gpt-6-luna]

Формулы §6.2: LC(N) = C_full + (N-1)*C_cached;  RAG(N) = N*C_rag. Только расчёт, API не вызывается.
Пишет results/charts/breakeven_<model>.png и results/breakeven_<model>.csv.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.estimate_cost import OUT, TOK, cost

SIZES = ("8k", "30k", "100k")
COL = {"8k": "#2a9d8f", "30k": "#e9a23b", "100k": "#c0392b"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-6-luna")
    m = ap.parse_args().model
    n = np.unique(np.round(np.logspace(0, np.log10(500), 200)).astype(int))
    c_rag = cost(m, TOK["rag"], OUT)
    fig, ax = plt.subplots(figsize=(8, 5))
    rows = []
    for s in SIZES:
        cf, cc = cost(m, TOK[s], OUT), cost(m, TOK[s], OUT, 1.0)
        lc = cf + (n - 1) * cc
        ax.plot(n, lc, color=COL[s], label=f"LC {s.upper()} (с кэшем)")
        ax.plot(n, n * c_rag, color=COL[s], ls="--", alpha=0.0)   # RAG одинаков для всех размеров, рисуем один раз ниже
        ns = (cf - cc) / (c_rag - cc) if cc < c_rag else None
        if ns and ns <= 500:
            ax.scatter([ns], [cf + (ns - 1) * cc], color=COL[s], zorder=5)
            ax.annotate(f"N*≈{ns:.1f}", (ns, cf + (ns - 1) * cc), textcoords="offset points", xytext=(6, -12), fontsize=8)
        rows.append({"size": s, "C_full": cf, "C_cached": cc, "C_rag": c_rag, "N_star": f"{ns:.1f}" if ns else "нет"})
    ax.plot(n, n * c_rag, color="#333", ls="--", label=f"RAG (k=8, ~{TOK['rag']} ток., не зависит от размера)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N вопросов к одному корпусу"); ax.set_ylabel("суммарная стоимость, $")
    ax.set_title(f"Безубыточность LC и RAG, {m}")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    out = Path("results"); (out / "charts").mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out / "charts" / f"breakeven_{m}.png", dpi=150)
    with open(out / f"breakeven_{m}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
