"""Проверка на утечку: все вопросы с ПУСТЫМ контекстом. Правильный ответ без контекста => факт не контролируемый.

    python -m scripts.check_leakage            # пишет data/leakage_report.json
"""
from __future__ import annotations

import json

from src.answer import answer_question
from src.config import ROOT, backend_name
from src.eval.judge import judge

EMPTY = "(документы не предоставлены)"


def main() -> None:
    qs = json.loads((ROOT / "data/questions.json").read_text(encoding="utf-8"))["questions"]
    rows = []
    for q in qs:
        a = answer_question(EMPTY, q["question"], salt="leak")
        v, why, _ = judge(q["question"], q["gold_answer"], a.answer if not a.abstained else a.text, salt="leak")
        rows.append(dict(qid=q["qid"], type=q["type"], verdict=v, abstained=a.abstained, answer=a.answer))
    leaked = [r for r in rows if r["verdict"] in ("correct", "partial")]
    rep = dict(backend=backend_name(), n=len(rows), leaked=len(leaked), leaked_qids=[r["qid"] for r in leaked], rows=rows)
    (ROOT / "data/leakage_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    by = {t: sum(1 for r in leaked if r["type"] == t) for t in ("T1", "T2", "T3")}
    print(f"вопросов: {len(rows)}, угадано/утекло без контекста: {len(leaked)} {by}; отказов: {sum(r['abstained'] for r in rows)}")
    for r in leaked:
        print(" ", r["qid"], r["verdict"], "|", r["answer"][:100])


if __name__ == "__main__":
    main()
