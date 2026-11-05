"""Калибровка судьи: 40 ответов размечаются вручную, считается согласованность судья↔человек.

  python -m scripts.calibrate_judge export runs/L1_mini      # -> results/judge_calibration.csv (колонка human пустая)
  # открой CSV, в колонке human впиши: correct / partial / incorrect / abstain (судья скрыт в отдельном столбце справа)
  python -m scripts.calibrate_judge score                    # -> согласованность и kappa в results/judge_agreement.json
"""
from __future__ import annotations

import csv
import json
import random
import sys

from src.config import ROOT

CSV = ROOT / "results/judge_calibration.csv"


def export(run: str, n: int = 40) -> None:
    rows = [json.loads(l) for l in (ROOT / run / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["repeat"] == 1]
    rnd = random.Random(7)
    pick = []
    for v in ("correct", "incorrect", "abstain", "partial"):   # стратификация по вердиктам судьи
        pool = [r for r in rows if r["verdict"] == v]
        rnd.shuffle(pool)
        pick += pool[: n // 4 if v != "partial" else n - len(pick) - 0]
    rest = [r for r in rows if r not in pick]
    rnd.shuffle(rest)
    pick = (pick + rest)[:n]
    rnd.shuffle(pick)
    CSV.parent.mkdir(exist_ok=True)
    with CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "question", "gold_answer", "system_answer", "contradiction_found", "quote", "human", "judge"])
        qs = {q["qid"]: q["question"] for q in json.loads((ROOT / "data/questions.json").read_text(encoding="utf-8"))["questions"]}
        for r in pick:
            w.writerow([r["qid"], qs[r["qid"]], r["gold_answer"], r["answer"], r["contradiction_found"], r["quote"], "", r["verdict"]])
    print(f"записано {len(pick)} строк -> {CSV}. Заполни колонку human.")


def score() -> None:
    rows = [r for r in csv.DictReader(CSV.open(encoding="utf-8")) if r["human"].strip()]
    h, j = [r["human"].strip().lower() for r in rows], [r["judge"] for r in rows]
    n = len(rows)
    agree = sum(a == b for a, b in zip(h, j)) / n
    labs = sorted(set(h) | set(j))
    pe = sum((h.count(l) / n) * (j.count(l) / n) for l in labs)
    kappa = (agree - pe) / (1 - pe) if pe < 1 else 1.0
    # бинарно: correct против остального (то, что идёт в accuracy)
    bin_agree = sum((a == "correct") == (b == "correct") for a, b in zip(h, j)) / n
    out = dict(n=n, agreement=round(agree, 3), cohen_kappa=round(kappa, 3), agreement_binary_correct=round(bin_agree, 3))
    (ROOT / "results/judge_agreement.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    export(sys.argv[2]) if sys.argv[1] == "export" else score()
