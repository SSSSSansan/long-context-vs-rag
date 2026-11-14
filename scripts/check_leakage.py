"""Шаг 1.4: проверка на утечку. Нужна локальная модель (LLM_BACKEND=local), платных вызовов нет.

    python -m scripts.check_leakage                 # оба режима
    python -m scripts.check_leakage --mode empty    # только пустой контекст
    python -m scripts.check_leakage --mode links    # только «связки без фактов»

Режим empty: ВСЕ вопросы (A, A-hard, B) с пустым контекстом. Правильный ответ => факт не контролируемый.
Режим links: для вопросов A-hard контекст состоит ТОЛЬКО из строк-связок их сущностей (без самого факта).
  Модель обязана отказаться. Если отвечает — связка сама выдаёт ответ или модель угадывает.
Структурная проверка (без модели): значение факта не встречается в тексте связок и вопроса.
Отчёты: data/leakage_report.json (empty), data/leakage_links_report.json (links).
"""
from __future__ import annotations

import argparse
import json

from src.answer import answer_question
from src.config import ROOT, backend_name
from src.eval.judge import judge

EMPTY = "(документы не предоставлены)"


def qset(q: dict) -> str:
    return q.get("set", "A") if q.get("anchor", True) else "A-hard"


def link_only_context(q: dict, by_id: dict) -> str:
    """Контекст из одних связок вопроса, сгруппированных по документам, как в корпусе. Факт не включается."""
    by_doc: dict[str, list[str]] = {}
    for i in q["link_fact_ids"]:
        r = by_id[i]
        by_doc.setdefault(r["doc_id"], []).append(r["statement"])
    return "\n\n".join(f"ДОКУМЕНТ {d}\n" + "\n".join(lines) for d, lines in sorted(by_doc.items()))


def leaks_value(q: dict, by_id: dict) -> list[str]:
    """Факты, чьё значение лежит в тексте связок или вопроса (должно быть пусто). Значение — число или название."""
    text = link_only_context(q, by_id) + "\n" + q["question"]
    from src.plant_facts import value_str       # «6 недель» целиком: голое число «6» совпало бы с чем угодно
    return [i for i in q["fact_ids"] if value_str(by_id[i]["attr"], by_id[i]["value"]).lower() in text.lower()]


def run_empty(qs: list[dict]) -> dict:
    rows = []
    for q in qs:
        a = answer_question(EMPTY, q["question"], salt="leak")
        v, _, _ = judge(q["question"], q["gold_answer"], a.answer if not a.abstained else a.text, salt="leak")
        rows.append(dict(qid=q["qid"], qset=qset(q), type=q["type"], verdict=v, abstained=a.abstained, answer=a.answer))
    return summarize(rows, "empty")


def run_links(qs: list[dict], by_id: dict) -> dict:
    rows = []
    for q in (x for x in qs if not x.get("anchor", True)):
        ctx = link_only_context(q, by_id)
        a = answer_question(ctx, q["question"], salt="leak-links")
        v, _, _ = judge(q["question"], q["gold_answer"], a.answer if not a.abstained else a.text, salt="leak-links")
        rows.append(dict(qid=q["qid"], qset="A-hard", type=q["type"], verdict=v, abstained=a.abstained,
                         answer=a.answer, value_in_context=leaks_value(q, by_id)))
    return summarize(rows, "links")


def summarize(rows: list[dict], mode: str) -> dict:
    leaked = [r for r in rows if r["verdict"] in ("correct", "partial")]
    return dict(mode=mode, backend=backend_name(), n=len(rows), leaked=len(leaked),
                leaked_qids=[r["qid"] for r in leaked], abstained=sum(r["abstained"] for r in rows),
                by_qset={s: sum(1 for r in leaked if r["qset"] == s) for s in ("A", "A-hard", "B")}, rows=rows)


def show(rep: dict) -> None:
    n_by = {s: sum(1 for r in rep["rows"] if r["qset"] == s) for s in ("A", "A-hard", "B")}
    print(f"[{rep['mode']}] backend={rep['backend']} вопросов: {rep['n']} {n_by}; утекло/угадано: {rep['leaked']} "
          f"{rep['by_qset']}; отказов: {rep['abstained']}")
    for r in rep["rows"]:
        if r["verdict"] in ("correct", "partial"):
            print("  УТЕЧКА", r["qid"], r["verdict"], "|", r["answer"][:100])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["empty", "links", "both"], default="both")
    mode = ap.parse_args().mode
    qs = json.loads((ROOT / "data/questions.json").read_text(encoding="utf-8"))["questions"]
    by_id = {f["fact_id"]: f for f in json.loads((ROOT / "data/gold.json").read_text(encoding="utf-8"))["facts"]}
    if mode in ("links", "both"):
        bad = [(q["qid"], leaks_value(q, by_id)) for q in qs if not q.get("anchor", True) and leaks_value(q, by_id)]
        print("Структура: значение факта в связках/вопросе:", bad or "нет (ок)")
    if mode in ("empty", "both"):
        rep = run_empty(qs)
        (ROOT / "data/leakage_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        show(rep)
    if mode in ("links", "both"):
        rep = run_links(qs, by_id)
        (ROOT / "data/leakage_links_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        show(rep)


if __name__ == "__main__":
    main()
