"""Агрегация: results/summary_*.csv и графики. mean ± std считается по повторам (std между повторами, не между вопросами).

  python -m src.eval.aggregate [runs/dir1 runs/dir2 ...]     # по умолчанию — все runs/*/raw.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..config import ROOT, base_cfg  # noqa: E402

RES = ROOT / "results"


def load(dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for d in dirs:
        p = d / "raw.jsonl"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                r["run"] = d.name
                rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["correct"] = (df.verdict == "correct").astype(float)
    df["lenient"] = df.verdict.isin(["correct", "partial"]).astype(float)
    df["system"] = df.model.str.replace(r"-\d{4}-\d{2}-\d{2}$", "", regex=True) + ":" + df.method + df.k_label.map(lambda k: "" if k is None or pd.isna(k) else f"@k={k}")
    mp = base_cfg()["model_prices"]
    pin = df.model.map(lambda m: mp.get(m, {"input": 0.0})["input"])
    pout = df.model.map(lambda m: mp.get(m, {"output": 0.0})["output"])
    df["cost_nocache_usd"] = (df.prompt_tokens * pin + df.completion_tokens * pout) / 1e6
    return df


def summarize(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    # сначала среднее по повторам внутри каждого повтора -> потом mean ± std по повторам
    per_rep = df.groupby(by + ["repeat"]).agg(
        accuracy=("correct", "mean"), lenient=("lenient", "mean"), evidence_recall=("evidence_recall", "mean"),
        citation=("citation_correct", "mean"), quote_grounded=("quote_grounded", "mean"),
        abstain=("abstained", "mean")).reset_index()
    g = per_rep.groupby(by)
    out = g.mean(numeric_only=True).drop(columns="repeat")
    out["accuracy_std"] = g.accuracy.std(ddof=0)
    tot = df.groupby(by).agg(
        n_calls=("qid", "size"), n_questions=("qid", "nunique"), repeats=("repeat", "nunique"),
        prompt_tokens=("prompt_tokens", "mean"), cached_share=("cached_tokens", lambda s: s.sum() / max(1, df.loc[s.index, "prompt_tokens"].sum())),
        lat_p50_ms=("generation_ms", lambda s: s.quantile(0.5)), lat_p95_ms=("generation_ms", lambda s: s.quantile(0.95)),
        retr_p50_ms=("retrieval_ms", lambda s: s.quantile(0.5)),
        cost_usd=("cost_usd", "sum"), cost_nocache_usd=("cost_nocache_usd", "sum"), n_correct=("correct", "sum"))
    out = out.join(tot)
    # $/correct: стоимость ответов / число верных (стоимость judge не входит; цена из первого, не из кэш-хитов)
    out["usd_per_correct"] = out.cost_nocache_usd / out.n_correct.replace(0, np.nan)
    out["usd_per_correct_cached"] = out.cost_usd / out.n_correct.replace(0, np.nan)
    return out.reset_index()


def charts(df: pd.DataFrame):
    (RES / "charts").mkdir(parents=True, exist_ok=True)
    main_ = df[(df.get("target_pct").isna() if "target_pct" in df else True) & df.k_label.isna() | df.method.ne("lc") & df.k_label.eq(8)]
    main_ = main_[main_.get("target_pct", pd.Series(index=main_.index, dtype=float)).isna()] if "target_pct" in main_ else main_
    if main_.empty:
        return
    s = summarize(main_, ["size", "method", "type"])
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for a, t in zip(ax, ["T1", "T2", "T3"]):
        piv = s[s.type == t].pivot(index="size", columns="method", values="accuracy").reindex([x for x in ["8k", "30k", "100k"] if x in set(s["size"])])
        piv.plot.bar(ax=a, rot=0, width=0.75)
        a.set_title(t); a.set_ylim(0, 1.05); a.set_xlabel("")
    ax[0].set_ylabel("accuracy")
    fig.suptitle("Accuracy по размеру корпуса и типу вопроса"); fig.tight_layout()
    fig.savefig(RES / "charts/accuracy_by_type.png", dpi=140); plt.close(fig)

    s2 = summarize(main_, ["size", "method"])
    fig, a = plt.subplots(figsize=(6.5, 3.8))
    s2.pivot(index="size", columns="method", values="usd_per_correct").reindex([x for x in ["8k", "30k", "100k"] if x in set(s2["size"])]).plot.bar(ax=a, rot=0, logy=True)
    a.set_ylabel("$ на один верный ответ (лог.)"); a.set_title("$/correct"); fig.tight_layout()
    fig.savefig(RES / "charts/usd_per_correct.png", dpi=140); plt.close(fig)


def position_chart(df: pd.DataFrame):
    p = df[df.get("target_pct", pd.Series(index=df.index, dtype=float)).notna()] if "target_pct" in df else df.iloc[0:0]
    if p.empty:
        return
    s = summarize(p, ["model", "size", "target_pct"])
    s.to_csv(RES / "summary_position.csv", index=False)
    fig, a = plt.subplots(figsize=(6.5, 3.8))
    for size, g in s.groupby("size"):
        a.errorbar(g.target_pct, g.accuracy, yerr=g.accuracy_std, marker="o", label=size, capsize=3)
    a.set_xlabel("позиция факта в корпусе, %"); a.set_ylabel("accuracy (LC)"); a.set_ylim(0, 1.05); a.legend()
    a.set_title("Lost in the Middle: LC"); fig.tight_layout()
    fig.savefig(RES / "charts/position_curve.png", dpi=140); plt.close(fig)


def main():
    dirs = [Path(a) for a in sys.argv[1:]] or sorted((ROOT / "runs").glob("*/"))
    df = load([d if d.is_absolute() else ROOT / d for d in dirs])
    if df.empty:
        raise SystemExit("нет данных в runs/*/raw.jsonl")
    RES.mkdir(exist_ok=True)
    has_pos = "target_pct" in df
    base = df[df.target_pct.isna()] if has_pos else df
    main_ = base[base.k_label.isna() | base.k_label.eq(base_cfg()["retrieval"]["top_k"])]
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    for name, by, d in [("summary_by_type", ["model", "size", "method", "type"], main_), ("summary_overall", ["model", "size", "method"], main_),
                        ("summary_ablation_k", ["model", "size", "system", "type"], base[base.method.ne("lc")])]:
        if d.empty:
            continue
        s = summarize(d, by)
        s.to_csv(RES / f"{name}.csv", index=False)
        print(f"\n== {name} ==\n", s.round(3).to_string(index=False))
    if has_pos:
        position_chart(df)
    base_ = df[df.target_pct.isna()] if has_pos else df
    charts_df = base_
    try:
        charts_df = charts_df.assign(k_label=charts_df.k_label.where(charts_df.method.ne("lc"), np.nan))
        charts(charts_df[charts_df.model.str.startswith('gpt-4o-mini')])
    except Exception as e:  # графики не должны ломать таблицы
        print("charts:", e)


if __name__ == "__main__":
    main()
