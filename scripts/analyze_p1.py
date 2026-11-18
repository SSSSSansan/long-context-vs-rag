"""P1: положение факта (0% / 50% / 100% корпуса) для LC на gpt-6-luna. T1 с якорем, 1 повтор. Отдельный набор (не A/A-hard/B).
Метрика: судья correct И value_in_answer (для T1 строгая = основная). Парный McNemar exact между позициями + Холм (3 пары на размер).
python -m scripts.analyze_p1 -> results/p1_luna.md"""
import json
from itertools import combinations
from pathlib import Path
import pandas as pd
from src.eval.aggregate import holm_adjust, mcnemar_exact, wilson_ci
from src.eval.value_check import value_in_answer

ROOT = Path(__file__).resolve().parent.parent
rows = [json.loads(l) for d in ("runs/v2_p1_30k", "runs/v2_p1_100k") for l in open(ROOT / d / "raw.jsonl")]
df = pd.DataFrame(rows)
df["judge"] = ((df.verdict == "correct") & ~df.abstained).astype(float)
df["acc"] = df.judge * [float(value_in_answer(g, a or "", q or "")) for g, a, q in zip(df.gold_answer, df.answer, df.quote)]
out = ["# P1: положение факта в LC (gpt-6-luna), T1 с якорем, 1 повтор — отдельный набор", "",
       "Один факт переносится в начало / середину / конец корпуса, вопрос тот же. Основная = судья И значение в ответе. n = число вопросов.", "",
       "| Размер | Позиция | n | основная [CI] | судья | отказы | факт на самом деле, % корпуса (ср.) |", "|---|---|---|---|---|---|---|"]
for size in ("30k", "100k"):
    for pct in (0, 50, 100):
        x = df[(df["size"] == size) & (df.target_pct == pct)]
        lo, hi = wilson_ci(x.acc.sum(), len(x))
        out.append(f"| {size.upper()} | {pct}% | {len(x)} | {x.acc.mean():.2f} [{lo:.2f}–{hi:.2f}] | {x.judge.mean():.2f} | {x.abstained.mean():.2f} | {x.actual_pct.mean():.1f} |")
out += ["", "Парные тесты между позициями (McNemar exact, поправка Холма по 3 парам на размер):", "",
        "| Размер | пара | только первая | только вторая | p | p_Холм |", "|---|---|---|---|---|---|"]
for size in ("30k", "100k"):
    w = df[df["size"] == size].pivot(index="qid", columns="target_pct", values="acc").astype(bool)
    pr = [(a, b, int((w[a] & ~w[b]).sum()), int((~w[a] & w[b]).sum())) for a, b in combinations([0, 50, 100], 2)]
    ps = [mcnemar_exact(oa, ob) for _, _, oa, ob in pr]
    for (a, b, oa, ob), p, ph in zip(pr, ps, holm_adjust(ps)):
        out.append(f"| {size.upper()} | {a}% против {b}% | {oa} | {ob} | {p:.3f} | {ph:.3f} |")
wrong = df[df.acc < 1][["size", "target_pct", "qid", "verdict", "gold_answer"]]
out += ["", "Неверные ответы:", ""] + ([f"- {r['size']} {r.target_pct}% {r.qid}: {r.verdict} (эталон: {r.gold_answer})" for _, r in wrong.iterrows()] or ["- нет"])
(ROOT / "results/p1_luna.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
