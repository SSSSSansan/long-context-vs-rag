"""P2: ранг gold-чанка (1/5/10) в контексте из 10 чанков, RAG на gpt-6-luna, T1, 30K. Наборы A и A-hard отдельно (A-hard — те же факты,
вопросы не независимы). Основная метрика = судья И значение в ответе. Парный McNemar exact + Холм (3 пары на набор).
python -m scripts.analyze_p2 -> results/p2_luna.md"""
import json
from itertools import combinations
from pathlib import Path
import pandas as pd
from src.eval.aggregate import holm_adjust, mcnemar_exact, wilson_ci
from src.eval.value_check import value_in_answer

ROOT = Path(__file__).resolve().parent.parent
df = pd.DataFrame([json.loads(l) for l in open(ROOT / "runs/v2_p2_luna/raw.jsonl")])
df["judge"] = ((df.verdict == "correct") & ~df.abstained).astype(float)
df["acc"] = df.judge * [float(value_in_answer(g, a or "", q or "")) for g, a, q in zip(df.gold_answer, df.answer, df.quote)]
df["qset"] = df.question_set
out = ["# P2: ранг gold-чанка в контексте из 10 чанков (RAG, gpt-6-luna, T1, 30K, 1 повтор) — отдельный набор", "",
       "Меняется только ранг gold-чанка (1 / 5 / 10), остальные 9 — трудные BM25-дистракторы без факта. Evidence recall = 1.0 по построению. n = число вопросов.", "",
       "| Набор | Ранг | n | основная [CI] | судья | отказы | evidence recall |", "|---|---|---|---|---|---|---|"]
for qs in ("A", "A-hard"):
    for r in (1, 5, 10):
        x = df[(df.qset == qs) & (df.gold_rank == r)]
        lo, hi = wilson_ci(x.acc.sum(), len(x))
        out.append(f"| {qs} | {r} | {len(x)} | {x.acc.mean():.2f} [{lo:.2f}–{hi:.2f}] | {x.judge.mean():.2f} | {x.abstained.mean():.2f} | {x.evidence_recall.mean():.2f} |")
out += ["", "Парные тесты между рангами (McNemar exact, Холм по 3 парам на набор):", "", "| Набор | пара | только первая | только вторая | p | p_Холм |", "|---|---|---|---|---|---|"]
for qs in ("A", "A-hard"):
    w = df[df.qset == qs].pivot(index="qid", columns="gold_rank", values="acc").astype(bool)
    pr = [(a, b, int((w[a] & ~w[b]).sum()), int((~w[a] & w[b]).sum())) for a, b in combinations([1, 5, 10], 2)]
    ps = [mcnemar_exact(oa, ob) for _, _, oa, ob in pr]
    for (a, b, oa, ob), p, ph in zip(pr, ps, holm_adjust(ps)):
        out.append(f"| {qs} | ранг {a} против {b} | {oa} | {ob} | {p:.3f} | {ph:.3f} |")
wrong = df[df.acc < 1]
out += ["", "Неверные ответы:", ""] + ([f"- {r.qset} ранг {r.gold_rank} {r.qid}: {r.verdict} (эталон: {r.gold_answer})" for _, r in wrong.iterrows()] or ["- нет"])
(ROOT / "results/p2_luna.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
