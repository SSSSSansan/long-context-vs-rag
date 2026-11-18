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


V2_HUMAN = ROOT / "results/judge_calibration_v2_for_human.csv"
V2_KEY = ROOT / "results/judge_calibration_v2_key.csv"     # НЕ показывать размечающему: вердикт судьи и метод


def export_blind_v2(n: int = 40) -> None:
    """Слепой набор из данных V2 (runs/v2_s1..s7, repeat=1): человек видит question/gold/prediction, ключ (judge, метод,
    набор, value_in_answer) лежит в отдельном файле. Страты: отказы, ошибки/partial судьи, верные, и строки
    «судья correct, но value_in_answer=False» (проверка строгости итоговой метрики)."""
    rows = []
    for d in sorted((ROOT / "runs").glob("v2_s[1-7]")):
        rows += [json.loads(l) for l in (d / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["repeat"] == 1]
    rnd = random.Random(11)
    groups = {
        "abstain": [r for r in rows if r["abstained"]],
        "wrong": [r for r in rows if not r["abstained"] and r["verdict"] in ("incorrect", "partial")],
        "judge_ok_value_no": [r for r in rows if r["verdict"] == "correct" and not r["abstained"] and not r["value_in_answer"]],
        "correct": [r for r in rows if r["verdict"] == "correct" and not r["abstained"] and r["value_in_answer"]],
    }
    quota = {"abstain": 8, "wrong": 10, "judge_ok_value_no": 8, "correct": 14}
    pick = []
    for g, k in quota.items():
        pool = groups[g][:]; rnd.shuffle(pool); pick += pool[:k]
    rest = [r for r in rows if r not in pick]; rnd.shuffle(rest)
    pick = (pick + rest)[:n]; rnd.shuffle(pick)
    with V2_HUMAN.open("w", encoding="utf-8", newline="") as fh, V2_KEY.open("w", encoding="utf-8", newline="") as fk:
        wh, wk = csv.writer(fh), csv.writer(fk)
        wh.writerow(["row_id", "question", "gold_answer", "prediction", "human_verdict"])
        wk.writerow(["row_id", "qid", "question_set", "size", "method", "judge_verdict", "value_in_answer", "abstained"])
        qs = {q["qid"]: q["question"] for q in json.loads((ROOT / "data/questions.json").read_text(encoding="utf-8"))["questions"]}
        for i, r in enumerate(pick, start=1):
            pred = _prediction_text(r["answer"], str(r["contradiction_found"]), r["quote"])
            wh.writerow([i, qs.get(r["qid"], r["qid"]), r["gold_answer"], pred, ""])
            wk.writerow([i, r["qid"], r["question_set"], r["size"], r["method"], r["verdict"], r["value_in_answer"], r["abstained"]])
    print(f"записано {len(pick)} строк -> {V2_HUMAN} (слепой), ключ -> {V2_KEY}. Заполни human_verdict: correct/partial/incorrect/abstain.")


def score_blind_v2() -> None:
    hum = {r["row_id"]: r["human_verdict"].strip().lower() for r in csv.DictReader(V2_HUMAN.open(encoding="utf-8")) if r["human_verdict"].strip()}
    key = {r["row_id"]: r for r in csv.DictReader(V2_KEY.open(encoding="utf-8"))}
    ids = [i for i in hum if i in key]
    h, j = [hum[i] for i in ids], [key[i]["judge_verdict"] for i in ids]
    n = len(ids)
    agree = sum(a == b for a, b in zip(h, j)) / n
    labs = sorted(set(h) | set(j))
    pe = sum((h.count(l) / n) * (j.count(l) / n) for l in labs)
    kappa = (agree - pe) / (1 - pe) if pe < 1 else 1.0
    bin_agree = sum((a == "correct") == (b == "correct") for a, b in zip(h, j)) / n
    # итоговая метрика (judge correct И value_in_answer И не отказ) против человека; значения пересчитываются из runs/ по ТЕКУЩЕМУ
    # правилу (main = winner-value-only для T2, strict = «оба числа»). ВНИМАНИЕ: правило подобрано постфактум по этим же 40 строкам.
    from src.eval.value_check import value_in_answer, value_in_answer_strict
    raw = {}
    for d in (ROOT / "runs").glob("v2_s[1-7]"):
        for l in (d / "raw.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            if r["repeat"] == 1:
                raw[(r["qid"], r["question_set"], r["size"], r["method"])] = r

    def final_agree(fn):
        res = []
        for i in ids:
            k = key[i]; r = raw[(k["qid"], k["question_set"], k["size"], k["method"])]
            res.append(k["judge_verdict"] == "correct" and k["abstained"] != "True" and fn(r["gold_answer"], r["answer"], r["quote"]))
        return sum(f == (a == "correct") for f, a in zip(res, h)) / n
    fin_strict, fin_main = final_agree(value_in_answer_strict), final_agree(value_in_answer)
    out = dict(n=n, agreement=round(agree, 3), cohen_kappa=round(kappa, 3), agreement_binary_correct=round(bin_agree, 3),
               agreement_final_metric_strict_before=round(fin_strict, 3), agreement_final_metric_winner_value_after=round(fin_main, 3),
               note="правило winner-value-only выбрано ПОСТФАКТУМ по расхождениям этих же 40 строк: оценка на тех же данных, не независимая проверка")
    (ROOT / "results/judge_agreement_v2.json").write_text(json.dumps(out, indent=1))
    print(out)


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
    elif cmd == "export_blind_v2":
        export_blind_v2()
    elif cmd == "score_blind_v2":
        score_blind_v2()
    elif cmd == "rejudge":
        rejudge()
    else:
        raise SystemExit(f"неизвестная команда: {cmd}")
