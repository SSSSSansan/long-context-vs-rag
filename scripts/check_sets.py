"""Проверка шага 1.2: наборы вопросов и распределение фактов. Ничего не вызывает по сети, ничего не тратит.

    python -m scripts.check_sets

Шаг 1.3 добавляет A-hard: 18 вопросов без имени сущности, связки (role=link) рядом с фактами, признаки вопроса
находят ровно одну сущность. Ожидается (PLAN_V2 §2.1, configs/corpus.yaml): набор A = 18 вопросов (6 T1 / 6 T2 / 6 T3), все факты которых
есть во всех трёх корпусах; набор B = 42 вопроса, ни один не отвечаем на 8K; факты в корпусах 30 / 65 / 100.
"""
from __future__ import annotations

import json
import sys
from collections import Counter

from src.config import ROOT

SIZES = ["8k", "30k", "100k"]


def main() -> int:
    allrec = json.loads((ROOT / "data/gold.json").read_text(encoding="utf-8"))["facts"]
    gold = {f["fact_id"]: f for f in allrec if f.get("role", "gold") == "gold"}
    links = [f for f in allrec if f.get("role") in ("link", "distractor_link")]
    qs = json.loads((ROOT / "data/questions.json").read_text(encoding="utf-8"))["questions"]
    meta = {s: json.loads((ROOT / f"corpus/built/{s}/meta.json").read_text(encoding="utf-8")) for s in SIZES}
    bad: list[str] = []

    def check(ok: bool, msg: str) -> None:
        print(("  OK   " if ok else "  FAIL ") + msg)
        if not ok:
            bad.append(msg)

    print("Корпуса:")
    for s in SIZES:
        m = meta[s]
        print(f"  {s:>4}: {m['tokens']} токенов, документов {m['n_docs']}, последний оборван: {m['last_doc_truncated']}")

    by_set = Counter(q["set"] if q.get("anchor", True) else "A-hard" for q in qs)
    print("\nВопросы:", dict(by_set), "| типы в A:", dict(Counter(q["type"] for q in qs if q["set"] == "A" and q.get("anchor", True))))
    check(by_set == {"A": 18, "A-hard": 18, "B": 42}, "A=18, A-hard=18, B=42")
    qa = [q for q in qs if q.get("anchor", True)]
    check(Counter(q["type"] for q in qa if q["set"] == "A") == {"T1": 6, "T2": 6, "T3": 6}, "в A по 6 вопросов каждого типа")
    hard = [q for q in qs if not q.get("anchor", True)]
    check(all(q["anchor"] is False for q in hard) and all(q["anchor"] is True for q in qa), "anchor: A/B=True, A-hard=False")
    qs = qa                                                    # дальнейшие проверки — как в шаге 1.2

    everywhere = [q["qid"] for q in qs if all(gold[i]["corpus_sizes"] == SIZES for i in q["fact_ids"])]
    check(len(everywhere) == 18 and all(next(q for q in qs if q["qid"] == i)["set"] == "A" for i in everywhere),
          f"ровно 18 вопросов отвечаемы на всех размерах, и это набор A (найдено {len(everywhere)})")
    b_on_8k = [q["qid"] for q in qs if q["set"] == "B" and "8k" in gold[q["fact_ids"][0]]["corpus_sizes"]]
    check(not b_on_8k, "ни один вопрос набора B не отвечаем на 8K")

    per = {s: sum(1 for f in gold.values() if s in f["corpus_sizes"]) for s in SIZES}
    check(per == {"8k": 30, "30k": 65, "100k": 100}, f"факты в корпусах {per} (ожидалось 30 / 65 / 100)")
    check(all(f["set"] == "A" for f in gold.values() if f["first_corpus"] == "8k"), "факты уровня 8k помечены set=A")

    docs8 = Counter(f["doc_id"] for f in gold.values() if f["set"] == "A")
    print("\nФакты набора A по документам:", dict(sorted(docs8.items())))
    last = meta["8k"]["doc_ids"][-1]
    check(last not in docs8 or not meta["8k"]["last_doc_truncated"], f"факты A не лежат в оборванном {last}")
    print("\nA-hard:")
    n_link = sum(1 for r in links if r["role"] == "link")
    check(n_link == 30, f"связок 30 (6 T1 + 12 T2 + 12 T3), найдено {n_link}")
    from src.hard_set import entities_matching
    bad_u = [q["qid"] for q in hard for ft in q["features"]
             if len(entities_matching(links, ft["kind"], ft)) != 1]
    check(not bad_u, f"признаки каждого вопроса A-hard находят ровно одну сущность (плохие: {bad_u})")
    byid = {r["fact_id"]: r for r in allrec}
    colo = [r["fact_id"] for r in links if not set(r.get("chunk_ids", [])) & set(byid[r["of"]].get("chunk_ids", []))]
    check(not colo, f"каждая связка лежит в одном чанке со своим фактом (нарушения: {colo})")
    check(all(q.get("link_chunk_ids") for q in hard), "у всех вопросов A-hard записаны link_chunk_ids")
    leak = [q["qid"] for q in hard
            if any(r["name"] in q["question"] for r in allrec if r.get("name"))]
    check(not leak, f"в тексте вопросов A-hard нет имён сущностей (нарушения: {leak})")
    print("\nИТОГ:", "всё в порядке" if not bad else f"{len(bad)} проверок не прошли")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
