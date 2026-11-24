"""Self-Route и роутер по типу вопроса — СИМУЛЯЦИЯ по сохранённым ответам gpt-6-luna (не отдельный прогон; новых вызовов нет, $0).

Пары RAG/LC берутся по номеру повтора (A и A-hard: 3 повтора, B: 1). Стоимость — холодная цена вызова (cost_nocache_usd), без судьи.
Пишет results/routing_sim.csv и results/routing_sim.md.   python -m scripts.simulate_routing
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

from src.eval import aggregate as ag
from src.eval.aggregate import wilson_ci
from src.methods.self_route import pair, strategies, summarize_strategies

ROOT = Path(__file__).resolve().parent.parent
RAG_METHODS = ["bm25", "hyb_rerank", "hyb_rrf", "emb"]     # первые два — по заданию, остальные показывают случай с заметной долей отказов
LABEL = {"bm25": "BM25", "hyb_rerank": "HYB-rerank", "hyb_rrf": "HYB-RRF", "emb": "EMB"}
STRAT_LABEL = {"rag": "чистый RAG", "lc": "чистый LC", "self_route": "Self-Route", "router": "роутер по типу (oracle)"}


def main():
    runs = [ROOT / "runs/v2_b_main", ROOT / "runs/v2_pilot_b"] + [Path(d) for d in sorted(glob.glob(str(ROOT / "runs/v2_s[1-7]")))]
    df = ag.load([r for r in runs if r.exists()])        # b_main раньше пилота: дубликаты ключа берутся из основного прогона
    df = df[df.model.str.startswith("gpt-6-luna")]
    rows = []
    for (qset, size), g in df[df.qset.isin(["A", "A-hard", "B"])].groupby(["qset", "size"]):
        lc = g[g.method == "lc"]
        for m in RAG_METHODS:
            rag = g[(g.method == m) & (g.k_label.isna() | g.k_label.eq(8))]
            if lc.empty or rag.empty:
                continue
            p = pair(rag, lc)
            if p.empty:
                continue
            s = strategies(p)
            res = summarize_strategies(s)
            nq = s.qid.nunique()
            for name, v in res.items():
                lo, hi = wilson_ci(v["accuracy"] * nq, nq)
                rows.append({"qset": qset, "size": size, "rag_method": m, "strategy": name, "n_pairs": len(s), "n_questions": nq,
                             "accuracy": v["accuracy"], "acc_ci_low": lo, "acc_ci_high": hi, "cost_per_call": v["cost_per_call"],
                             "usd_per_correct": v["usd_per_correct"], "fallback_share": v.get("fallback_share")})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/routing_sim.csv", index=False)
    md = ["# Self-Route и роутер по типу вопроса — симуляция по сохранённым ответам (gpt-6-luna; не отдельный прогон, $0)", "",
          "Постфактум, отдельно от главной таблицы. Self-Route: RAG отвечает; если ответ «в документах ответа нет» — берётся ответ LC на тот же вопрос и повтор; "
          "стоимость = RAG + доля отказов × LC. Роутер (oracle): T1 → RAG, T2/T3 → LC. Холодная цена вызова без кэша и судьи. "
          "Ячейка: accuracy [CI Уилсона по вопросам] · $ за вызов · $/correct. Доля отказов RAG = доля вызовов LC в Self-Route.", ""]
    for (qset, size), g in out.groupby(["qset", "size"], sort=False):
        md += [f"## {qset}, {size.upper()} (n вопросов = {int(g.n_questions.iloc[0])}, пар = {int(g.n_pairs.iloc[0])})", "",
               "| RAG-метод | отказов RAG | " + " | ".join(STRAT_LABEL.values()) + " |", "|---|---|" + "---|" * 4]
        for m, gm in g.groupby("rag_method", sort=False):
            cells = []
            for k in STRAT_LABEL:
                r = gm[gm.strategy == k].iloc[0]
                cells.append(f"{r.accuracy:.2f} [{r.acc_ci_low:.2f}–{r.acc_ci_high:.2f}] · ${r.cost_per_call:.5f} · ${r.usd_per_correct:.5f}")
            fb = gm[gm.strategy == "self_route"].iloc[0].fallback_share
            md.append(f"| {LABEL[m]} | {fb:.0%} | " + " | ".join(cells) + " |")
        md.append("")
    (ROOT / "results/routing_sim.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    pd.set_option("display.width", 200)
    print("\n".join(md))


if __name__ == "__main__":
    main()
