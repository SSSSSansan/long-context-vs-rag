"""Графики V2 (gpt-6-luna) из runs/*/raw.jsonl — без API, бесплатно.   python -m scripts.make_charts_v2   (или make charts)

Пишет в results/charts/: accuracy_by_type, anchor_effect, usd_per_correct, cost_per_call, evidence_recall, position_curve.
Цвет метода, маркер и штриховка — из configs/style.yaml (общий стиль с демо). Доверительные интервалы — Уилсон 95% по числу вопросов.
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.eval import aggregate as ag  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/charts"
STYLE = yaml.safe_load((ROOT / "configs/style.yaml").read_text(encoding="utf-8"))
M = STYLE["methods"]
SIZES = ["8k", "30k", "100k"]
RAG5 = ["lc", "bm25", "emb", "hyb_rrf", "hyb_rerank"]
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "figure.dpi": 100, "axes.titlesize": 11, "axes.labelsize": 9.5, "legend.fontsize": 8.5})


def load() -> pd.DataFrame:
    df = ag.load([Path(d) for d in sorted(glob.glob(str(ROOT / "runs/v2_s[1-7]")))])
    return df[df.model.str.startswith("gpt-6-luna")]


def _yerr(g: pd.DataFrame, col: str = "accuracy") -> np.ndarray:
    return np.vstack([(g[col] - g[f"{'acc' if col == 'accuracy' else col}_ci_low"]).clip(lower=0), (g[f"{'acc' if col == 'accuracy' else col}_ci_high"] - g[col]).clip(lower=0)])


def _legend_methods(fig, methods, ncol=None, y=1.0):
    h = [plt.Rectangle((0, 0), 1, 1, facecolor=M[m]["color"], hatch=M[m]["hatch"], edgecolor="white" if M[m]["hatch"] else M[m]["color"]) for m in methods]
    fig.legend(h, [M[m]["label"] for m in methods], loc="upper center", ncol=ncol or len(methods), bbox_to_anchor=(0.5, y), frameon=False)


def accuracy_by_type(df):
    s = ag.summarize(df[df.method.isin(RAG5)], ["qset", "size", "method", "type"])
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    w = 0.8 / len(RAG5)
    for ri, qs in enumerate(["A", "A-hard"]):
        for ci, t in enumerate(["T1", "T2", "T3"]):
            ax = axes[ri][ci]
            for mi, m in enumerate(RAG5):
                g = s[(s.qset == qs) & (s.type == t) & (s.method == m)].set_index("size").reindex(SIZES)
                x = np.arange(3) + (mi - (len(RAG5) - 1) / 2) * w
                v = g.accuracy.fillna(0).to_numpy()
                ax.bar(x, np.maximum(v, 0.015), w * 0.92, color=M[m]["color"], hatch=M[m]["hatch"], edgecolor="white", linewidth=0.4)
                ax.errorbar(x, v, yerr=_yerr(g.fillna(0)), fmt="none", ecolor="#333", elinewidth=0.8, capsize=1.5)
            ax.set_xticks(range(3)); ax.set_xticklabels([z.upper() for z in SIZES]); ax.set_ylim(0, 1.12)
            ax.set_title(f"{'A (с якорем)' if qs == 'A' else 'A-hard (без якоря)'} · {t}")
            if ci == 0:
                ax.set_ylabel("accuracy")
    fig.suptitle("Accuracy по размеру корпуса, типу вопроса и методу (gpt-6-luna; 6 вопросов × 3 повтора на столбец; усы — 95% CI Уилсона)", y=1.0, fontsize=11)
    _legend_methods(fig, RAG5, y=0.965)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(OUT / "accuracy_by_type.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def anchor_effect(df):
    s = ag.summarize(df[df.method.isin(RAG5)], ["qset", "size", "method"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    for ci, z in enumerate(SIZES):
        ax = axes[ci]
        for mi, m in enumerate(RAG5):
            for qs, dx in (("A", -0.12), ("A-hard", 0.12)):
                g = s[(s["size"] == z) & (s.method == m) & (s.qset == qs)]
                if g.empty:
                    continue
                r = g.iloc[0]; st = STYLE["anchor"][qs]
                ax.errorbar(mi + dx, r.accuracy, yerr=_yerr(g), fmt=st["marker"], ms=8, capsize=3, color=M[m]["color"],
                            mfc=M[m]["color"] if st["filled"] else "white", mew=1.6, elinewidth=1.2)
            a = s[(s["size"] == z) & (s.method == m) & (s.qset == "A")].accuracy
            h = s[(s["size"] == z) & (s.method == m) & (s.qset == "A-hard")].accuracy
            if len(a) and len(h):
                ax.plot([mi - 0.12, mi + 0.12], [a.iloc[0], h.iloc[0]], color="#888", lw=0.8, zorder=0)
        ax.set_xticks(range(len(RAG5))); ax.set_xticklabels([M[m]["label"] for m in RAG5], rotation=20); ax.set_ylim(0.2, 1.08)
        ax.set_title(f"Корпус {z.upper()}")
        if ci == 0:
            ax.set_ylabel("accuracy")
    h = [plt.Line2D([], [], marker="o", ls="", color="#444", mfc="#444", ms=8, label=STYLE["anchor"]["A"]["label"] + " — закрашенный круг"),
         plt.Line2D([], [], marker="D", ls="", color="#444", mfc="white", mew=1.6, ms=8, label=STYLE["anchor"]["A-hard"]["label"] + " — пустой ромб")]
    fig.legend(handles=h, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.suptitle("Эффект лексического якоря: accuracy с якорем и без по методам и размерам (n = 18 вопросов × 3 повтора; усы — 95% CI Уилсона)", y=1.08, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(OUT / "anchor_effect.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def usd_per_correct(df):
    s = ag.summarize(df[df.method.isin(RAG5 + ["agent_bm25", "agent_emb", "agent_hyb"])], ["size", "method"])
    methods = [m for m in RAG5 + ["agent_bm25", "agent_emb", "agent_hyb"] if m in set(s.method)]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    w = 0.8 / len(methods)
    for mi, m in enumerate(methods):
        for si, z in enumerate(SIZES):
            g = s[(s["size"] == z) & (s.method == m)]
            if g.empty:
                continue
            v = g.usd_per_correct.iloc[0]
            x = si + (mi - (len(methods) - 1) / 2) * w
            ax.bar(x, v, w * 0.92, color=M[m]["color"], hatch=M[m]["hatch"], edgecolor="white", linewidth=0.4)
            ax.text(x, v * 1.08, f"{v * 1000:.2f}", ha="center", fontsize=6.5, rotation=90)
    ax.set_yscale("log"); ax.set_xticks(range(3)); ax.set_xticklabels([z.upper() for z in SIZES]); ax.set_ylim(1e-4, 0.1)
    ax.set_ylabel("$ на верный ответ (лог.)")
    ax.set_title("Цена верного ответа: холодная цена без кэша и судьи, A + A-hard, gpt-6-luna (подписи — милли-доллары; агенты только на 100K)", fontsize=9.5)
    _legend_methods(fig, methods, ncol=len(methods), y=0.0); fig.subplots_adjust(bottom=0.2)
    fig.savefig(OUT / "usd_per_correct.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def cost_per_call(df):
    d = df[df.method.isin(["lc", "bm25", "agent_bm25"])].copy()
    d["grp"] = d.method.map({"lc": "lc", "bm25": "bm25", "agent_bm25": "agent_bm25"})
    s = d.groupby(["size", "grp"]).cost_nocache_usd.mean().reset_index()
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for m in ("lc", "bm25", "agent_bm25"):
        g = s[s.grp == m].set_index("size").reindex(SIZES).dropna()
        ax.plot([SIZES.index(z) for z in g.index], g.cost_nocache_usd, marker=M[m]["marker"], ms=9, lw=2, color=M[m]["color"], label=M[m]["label"] if m != "bm25" else "RAG (BM25; EMB и HYB — те же ≈$0.00045)")
        for z, v in g.cost_nocache_usd.items():
            ax.annotate(f"${v:.5f}", (SIZES.index(z), v), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)
    lc, rag = s[s.grp == "lc"].set_index("size").cost_nocache_usd, s[s.grp == "bm25"].set_index("size").cost_nocache_usd
    ax.set_yscale("log"); ax.set_xticks(range(3)); ax.set_xticklabels([z.upper() for z in SIZES]); ax.set_xlim(-0.3, 2.3); ax.set_ylim(2e-4, 3e-2)
    ax.set_ylabel("$ за один вызов (лог. шкала)"); ax.set_xlabel(f"размер корпуса (LC дороже RAG: ×{lc['8k'] / rag['8k']:.1f} / ×{lc['30k'] / rag['30k']:.1f} / ×{lc['100k'] / rag['100k']:.0f})")
    ax.set_title("Стоимость вызова: LC растёт с корпусом, RAG и агент почти не зависят от него"); ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "cost_per_call.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def evidence_recall(df):
    d = df[df.method.isin(["bm25", "emb", "hyb_rrf", "hyb_rerank", "agent_bm25", "agent_emb", "agent_hyb"])]
    s = ag.summarize(d, ["qset", "size", "method"])
    methods = ["bm25", "emb", "hyb_rrf", "hyb_rerank"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    w = 0.8 / len(methods)
    for ai, qs in enumerate(["A", "A-hard"]):
        ax = axes[ai]
        for mi, m in enumerate(methods):
            g = s[(s.qset == qs) & (s.method == m)].set_index("size").reindex(SIZES)
            x = np.arange(3) + (mi - (len(methods) - 1) / 2) * w
            ax.bar(x, g.evidence_recall.fillna(0), w * 0.92, color=M[m]["color"], hatch=M[m]["hatch"], edgecolor="white", linewidth=0.4)
            ax.errorbar(x, g.evidence_recall.fillna(0), yerr=_yerr(g.fillna(0), "evidence_recall"), fmt="none", ecolor="#333", elinewidth=0.8, capsize=1.5)
            for xi, v in zip(x, g.evidence_recall):
                if not np.isnan(v):
                    ax.text(xi, 0.03, f"{v:.2f}", ha="center", fontsize=6.5, rotation=90, color="white" if M[m]["color"] in ("#0072B2", "#D55E00", "#009E73") else "black")
        ax.set_xticks(range(3)); ax.set_xticklabels([z.upper() for z in SIZES]); ax.set_ylim(0, 1.08)
        ax.set_title("A (с якорем)" if qs == "A" else "A-hard (без якоря)")
        if ai == 0:
            ax.set_ylabel("evidence recall")
    fig.suptitle("Evidence recall: у EMB ломается именно поиск (LC = 1.0 по определению). 95% CI Уилсона по вопросам", y=1.02, fontsize=11)
    _legend_methods(fig, methods, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(OUT / "evidence_recall.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def position_curve():
    p = ag.load([Path(d) for d in (ROOT / "runs/v2_p1_30k", ROOT / "runs/v2_p1_100k", ROOT / "runs/v2_p2_luna") if d.exists()])
    p = p[p.model.str.startswith("gpt-6-luna")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    p1 = p[p.target_pct.notna()]
    ax = axes[0]
    for z, mk, off in (("30k", "o", -2.0), ("100k", "s", 2.0)):
        g = ag.summarize(p1[p1["size"] == z], ["target_pct"]).sort_values("target_pct")
        if g.empty:
            continue
        ax.errorbar(g.target_pct + off, g.accuracy, yerr=_yerr(g), marker=mk, ms=8, capsize=3, lw=1.6, label=f"{z.upper()} (n = {int(g.n_questions.iloc[0])})",
                    mfc="white" if z == "100k" else None)
    ax.set_xlabel("позиция факта в корпусе, % (точки разнесены по x для читаемости)"); ax.set_ylabel("accuracy (LC)"); ax.set_ylim(0, 1.1)
    ax.set_xticks([0, 50, 100]); ax.set_title("P1: положение факта в длинном контексте"); ax.legend(frameon=False, loc="lower center")
    p2 = p[p.gold_rank.notna()]
    ax = axes[1]
    for qs, mk in (("A", "o"), ("A-hard", "D")):
        g = ag.summarize(p2[p2.qset == qs], ["gold_rank"]).sort_values("gold_rank")
        if g.empty:
            continue
        ax.errorbar(g.gold_rank + (-0.12 if qs == "A" else 0.12), g.accuracy, yerr=_yerr(g), marker=mk, ms=8, capsize=3, lw=1.6, label=f"{qs} (n = {int(g.n_questions.iloc[0])})", mfc="white" if qs == "A-hard" else None)
    ax.set_xlabel("ранг gold-чанка среди 10 переданных"); ax.set_ylim(0, 1.1); ax.set_xticks([1, 5, 10])
    ax.set_title("P2: ранг нужного чанка в выдаче RAG (30K, T1)"); ax.legend(frameon=False, loc="lower center")
    fig.suptitle("Эффекта позиции не обнаружено — потолок ≈1.0 при малой выборке (n = 6–13), это не доказательство отсутствия эффекта", y=1.02, fontsize=10.5)
    fig.tight_layout(); fig.savefig(OUT / "position_curve.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    for fn in (accuracy_by_type, anchor_effect, usd_per_correct, cost_per_call, evidence_recall):
        fn(df); print("ok", fn.__name__)
    position_curve(); print("ok position_curve")


if __name__ == "__main__":
    main()
