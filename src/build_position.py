"""Варианты корпуса для позиционного эксперимента (Lost in the Middle).

Для каждого из N T1-фактов берём пункт-носитель, ВЫРЕЗАЕМ его из исходного места и вставляем
на заданную долю длины корпуса (0/25/50/75/100%). Вопрос и ответ те же — меняется только позиция.
Пункт вставляется отдельной строкой между пунктами другого документа, стиль окружающего текста сохраняется.

Запуск:  python -m src.build_position      (после make corpus)
Выход:   corpus/built/position/<size>/<fact_id>_p<pct>.txt  и  data/position_index.json
"""
from __future__ import annotations

import json
import re

from .config import ROOT, load_yaml

CLAUSE = re.compile(r"^\d+\.\d+\. ")


def move_fact(corpus: str, statement: str, pct: int) -> tuple[str, str, float]:
    """-> (новый корпус, doc_id документа-приёмника, фактическая позиция %)."""
    lines = corpus.split("\n")
    src = [i for i, l in enumerate(lines) if statement in l]
    assert len(src) == 1, "факт должен встречаться ровно в одной строке"
    clause = lines.pop(src[0])
    clause = CLAUSE.sub("", clause)
    total = sum(len(l) + 1 for l in lines)
    # допустимые точки вставки: сразу после строки-пункта (между пунктами)
    cand, off = [], 0
    for i, l in enumerate(lines):
        off += len(l) + 1
        if CLAUSE.match(l) and (i + 1 >= len(lines) or lines[i + 1].strip() != "" or pct == 100):
            cand.append((i + 1, off))
    if pct == 0:
        at = cand[0][0]
    elif pct == 100:
        at = cand[-1][0]
    else:
        at = min(cand, key=lambda c: abs(c[1] - total * pct / 100))[0]
    lines.insert(at, f"9.9. {clause}")
    doc = next(m.group(1) for l in reversed(lines[:at]) if (m := re.match(r"^ДОКУМЕНТ (doc_\d+)$", l)))
    text = "\n".join(lines)
    return text, doc, round(100 * text.index(statement) / len(text), 1)


def main() -> None:
    cfg = load_yaml("configs/corpus.yaml")
    gold = {f["fact_id"]: f for f in json.loads((ROOT / "data/gold.json").read_text(encoding="utf-8"))["facts"]}
    qs = json.loads((ROOT / "data/questions.json").read_text(encoding="utf-8"))["questions"]
    t1 = sorted((q for q in qs if q["type"] == "T1"), key=lambda q: q["qid"])[:cfg["facts"]["position_facts"]]
    index = []
    for size in ("30k", "100k"):
        corpus = (ROOT / cfg["built_dir"] / size / "corpus.txt").read_text(encoding="utf-8")
        out = ROOT / cfg["built_dir"] / "position" / size
        out.mkdir(parents=True, exist_ok=True)
        for q in t1:
            f = gold[q["fact_ids"][0]]
            for pct in cfg["facts"]["position_levels_pct"]:
                text, doc, real = move_fact(corpus, f["statement"], pct)
                assert text.count(f["statement"]) == 1
                name = f"{f['fact_id']}_p{pct}.txt"
                (out / name).write_text(text, encoding="utf-8")
                index.append(dict(size=size, qid=q["qid"], fact_id=f["fact_id"], target_pct=pct,
                                  actual_pct=real, doc_id=doc, file=f"{cfg['built_dir']}/position/{size}/{name}"))
    (ROOT / "data/position_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"вариантов: {len(index)}")
    for r in index[:10]:
        print(r["size"], r["fact_id"], r["target_pct"], "->", r["actual_pct"], r["doc_id"])


if __name__ == "__main__":
    main()
