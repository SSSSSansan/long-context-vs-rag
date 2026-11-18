"""Набор B (вопросы с якорем, 42, 100K, 1 повтор): таблица в двух вариантах метрики + судья, McNemar exact + Холм.
B считается ОТДЕЛЬНО от A и A-hard. Пилот (6 вопросов LC/EMB) входит через дисковый кэш (стоимость пилота $0.0787 учтена отдельно).
python -m scripts.analyze_b   -> results/main_table_b.md, results/mcnemar_b_{main,strict}.csv"""
from pathlib import Path
import pandas as pd
from src.eval import aggregate as ag

ROOT = Path(__file__).resolve().parent.parent
df = ag.load([ROOT / "runs/v2_b_main"])
df = df[df.model.str.startswith("gpt-6-luna")].copy()
order = [("lc", "LC"), ("bm25", "BM25"), ("emb", "EMB"), ("hyb_rerank", "HYB-rerank"), ("agent_bm25", "AGENT(bm25)")]
pilot_cost = 0.0787
real_cost = {}
for m, _ in order:
    x = df[df.method == m]
    real_cost[m] = (x.cost_usd + x.judge_cost_usd.fillna(0)).sum() + (pilot_cost if m in ("lc",) else 0)
out = ["# Набор B (с якорем): gpt-6-luna, 100K, 1 повтор, n = 42 вопроса на метод", "",
       "Основная = судья correct И value_in_answer (T2: число победителя); строгая = T2 оба числа; судья = вердикт судьи (отказ=0). "
       "Пилот (6 вопросов LC и EMB) включён. Стоимость LC включает пилот $0.0787 (EMB пилота ≈ копейки, не добавлена).", "",
       "| Метод | основная [CI] | строгая | судья | отказы | evidence recall | стоимость, $ (ответ+судья) |", "|---|---|---|---|---|---|---|"]
sm = ag.summarize(df, ["method"])
st = df.copy(); st["correct"] = st["correct_strict"]; ss = ag.summarize(st, ["method"])
for m, lab in order:
    r = sm[sm.method == m].iloc[0]; r2 = ss[ss.method == m].iloc[0]
    out.append(f"| {lab} | {r.accuracy:.2f} [{r.acc_ci_low:.2f}–{r.acc_ci_high:.2f}] | {r2.accuracy:.2f} | {r.accuracy_judge:.2f} | {r.abstain:.2f} | {r.evidence_recall:.2f} | {real_cost[m]:.3f} |")
out += ["", "По типам (основная / строгая / судья), n=14 на ячейку:", "", "| Метод | T1 | T2 | T3 |", "|---|---|---|---|"]
for m, lab in order:
    cs = []
    for t in ("T1", "T2", "T3"):
        x = df[(df.method == m) & (df.type == t)]
        cs.append(f"{x.correct.mean():.2f} / {x.correct_strict.mean():.2f} / {x.correct_judge.mean():.2f}")
    out.append(f"| {lab} | " + " | ".join(cs) + " |")
for variant, col in (("main", "correct"), ("strict", "correct_strict")):
    d = df.copy(); d["correct"] = d[col]
    t = ag.mcnemar_tables(d, qsets=("B",))
    t = t[t.family == "method"]
    t.to_csv(ROOT / f"results/mcnemar_b_{variant}.csv", index=False)
    out += ["", f"McNemar exact + Холм ({variant}), пар {len(t)}, n пар = 42; p_Холм<0.1 или p<0.05:", "",
            "| a | b | только a | только b | acc a | acc b | p | p_Холм |", "|---|---|---|---|---|---|---|---|"]
    for _, r in t[(t.p_value < 0.05) | (t.p_holm < 0.1)].iterrows():
        out.append(f"| {r.a} | {r.b} | {r.only_a} | {r.only_b} | {r.acc_a:.2f} | {r.acc_b:.2f} | {r.p_value:.4f} | {r.p_holm:.3f} |")
    if t[(t.p_value < 0.05) | (t.p_holm < 0.1)].empty:
        out.append("| (нет таких пар) | | | | | | | |")
(ROOT / "results/main_table_b.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
