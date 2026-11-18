"""Калибровка судьи: 40 ответов размечаются вручную, считается согласованность судья↔человек.

Старый режим (export/score) положил вердикт судьи в тот же CSV, что получает размечающий —
это не слепая разметка. Разметку по нему в прошлый раз сделал Claude, а не человек; эти цифры
помечены в REPORT.md как предварительные. Новый режим (export_blind/score_blind) этого не
повторяет: человек видит только question/gold_answer/prediction, вердикт судья лежит в другом
файле и подставляется к ответам человека только на этапе score_blind.

  python -m scripts.calibrate_judge export_blind
      -> results/judge_calibration_for_human.csv: row_id, question, gold_answer, prediction,
         human_verdict (пусто). Никаких колонок с вердиктом судьи или методом (LC/BM25/EMB) —
         человек размечает вслепую.
  # заполни human_verdict значениями: correct / partial / incorrect / abstain
  python -m scripts.calibrate_judge score_blind
      -> сравнивает заполненный human_verdict с вердиктом судьи (results/judge_calibration.csv,
         сопоставление по row_id = номеру строки), пишет results/judge_agreement.json

  python -m scripts.calibrate_judge rejudge   # перекалибровка текущего судьи на тех же 40 слепых примерах

Старый режим (export/score) оставлен ниже без изменений — он читает runs/*/raw.jsonl напрямую
и генерирует results/judge_calibration.csv, на основе которого сделан export_blind.
"""
from __future__ import annotations

import csv
import json
import random
import sys

from src.config import ROOT

CSV = ROOT / "results/judge_calibration.csv"
CSV_BLIND = ROOT / "results/judge_calibration_for_human.csv"


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


def _prediction_text(system_answer: str, contradiction_found: str, quote: str) -> str:
    """Воспроизводит pred_text() из src/eval/run.py — то, что реально увидел судья."""
    if system_answer.strip() == "В предоставленных документах ответа нет.":
        return system_answer
    found = "да" if contradiction_found.strip().lower() == "true" else "нет"
    return f"{system_answer}\nПротиворечие найдено: {found}\nЦитата: {quote}"


def export_blind() -> None:
    """results/judge_calibration.csv -> results/judge_calibration_for_human.csv,
    без вердикта судьи и без привязки к методу. Сопоставление с judge на score_blind —
    только по номеру строки (row_id), qid не годится как ключ: один и тот же вопрос
    встречается на нескольких размерах корпуса (gold.json вложенный), поэтому qid в
    judge_calibration.csv не уникален."""
    rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    CSV_BLIND.parent.mkdir(exist_ok=True)
    with CSV_BLIND.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "question", "gold_answer", "prediction", "human_verdict"])
        for i, r in enumerate(rows, start=1):
            pred = _prediction_text(r["system_answer"], r["contradiction_found"], r["quote"])
            w.writerow([i, r["question"], r["gold_answer"], pred, ""])
    print(f"записано {len(rows)} строк -> {CSV_BLIND}. Заполни human_verdict "
          f"(correct/partial/incorrect/abstain). Вердикт судьи в этом файле отсутствует.")


def score_blind() -> None:
    judge_rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    human_rows = {int(r["row_id"]): r["human_verdict"].strip().lower()
                  for r in csv.DictReader(CSV_BLIND.open(encoding="utf-8")) if r["human_verdict"].strip()}
    if not human_rows:
        print(f"human_verdict пуст в {CSV_BLIND} — сначала заполни разметку.")
        return
    h, j = [], []
    for row_id, hv in sorted(human_rows.items()):
        h.append(hv)
        j.append(judge_rows[row_id - 1]["judge"])
    n = len(h)
    agree = sum(a == b for a, b in zip(h, j)) / n
    labs = sorted(set(h) | set(j))
    pe = sum((h.count(l) / n) * (j.count(l) / n) for l in labs)
    kappa = (agree - pe) / (1 - pe) if pe < 1 else 1.0
    bin_agree = sum((a == "correct") == (b == "correct") for a, b in zip(h, j)) / n
    out = dict(n=n, agreement=round(agree, 3), cohen_kappa=round(kappa, 3),
               agreement_binary_correct=round(bin_agree, 3), rater="human", blind=True)
    (ROOT / "results/judge_agreement.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(out)


def rejudge() -> None:
    """Перекалибровка: тот же слепой файл (40 примеров) и ТА ЖЕ ручная разметка human_verdict, но судит текущая
    модель судьи (backends.<backend>.judge_model в configs/base.yaml). Новая ручная разметка не нужна.
    Пишет results/judge_calibration_<model>.csv и results/judge_agreement_<model>.json; старый judge_agreement.json не трогает.

      LLM_BACKEND=local  python -m scripts.calibrate_judge rejudge   # бесплатно, qwen как судья
      LLM_BACKEND=openai python -m scripts.calibrate_judge rejudge   # gpt-6-luna, ~$0.01 (cost_guard считает)
    """
    from src.config import backend_name, base_cfg
    from src.eval.judge import judge

    model = base_cfg()["backends"][backend_name()]["judge_model"]
    rows = [r for r in csv.DictReader(CSV_BLIND.open(encoding="utf-8")) if r["human_verdict"].strip()]
    if not rows:
        print(f"human_verdict пуст в {CSV_BLIND}.")
        return
    h, j, out_rows, cost = [], [], [], 0.0
    for r in rows:
        verdict, why, resp = judge(r["question"], r["gold_answer"], r["prediction"], salt="calib")
        cost += resp.cost_usd
        h.append(r["human_verdict"].strip().lower()); j.append(verdict)
        out_rows.append([r["row_id"], h[-1], verdict, why])
    n = len(h)
    agree = sum(a == b for a, b in zip(h, j)) / n
    labs = sorted(set(h) | set(j))
    pe = sum((h.count(l) / n) * (j.count(l) / n) for l in labs)
    kappa = (agree - pe) / (1 - pe) if pe < 1 else 1.0
    bin_agree = sum((a == "correct") == (b == "correct") for a, b in zip(h, j)) / n
    res = dict(n=n, judge_model=model, agreement=round(agree, 3), cohen_kappa=round(kappa, 3),
               agreement_binary_correct=round(bin_agree, 3), cost_usd=round(cost, 6), rater="human", blind=True)
    tag = model.replace("/", "_")
    with (ROOT / f"results/judge_calibration_{tag}.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["row_id", "human", "judge", "rationale"]); w.writerows(out_rows)
    (ROOT / f"results/judge_agreement_{tag}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(res)
    print("расхождения (row_id: человек -> судья):", [(r[0], r[1], r[2]) for r in out_rows if r[1] != r[2]])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "export":
        export(sys.argv[2])
    elif cmd == "score":
        score()
    elif cmd == "export_blind":
        export_blind()
    elif cmd == "score_blind":
        score_blind()
    elif cmd == "rejudge":
        rejudge()
    else:
        raise SystemExit(f"неизвестная команда: {cmd}")
