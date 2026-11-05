"""P1: LC с фактом на 0/25/50/75/100% корпуса. Использует варианты из build_position."""
from __future__ import annotations

import json

from ..config import ROOT
from .run import execute


def position_tasks(spec, gold, qs):
    idx = json.loads((ROOT / "data/position_index.json").read_text(encoding="utf-8"))
    byq = {q["qid"]: q for q in qs}
    tasks = []
    for r in spec["runs"]:
        facts = sorted({it["fact_id"] for it in idx})[: r.get("n_facts", 99)]
        for it in idx:
            if it["fact_id"] in facts and it["size"] in r["sizes"] and it["target_pct"] in r["positions_pct"]:
                for rep in range(1, r["repeats"] + 1):
                    tasks.append(dict(size=it["size"], method="lc", k=None, k_label=None, q=byq[it["qid"]], repeat=rep,
                                      pos=it["target_pct"], item=it))

    def fn(t):
        it = t["item"]
        corpus = (ROOT / it["file"]).read_text(encoding="utf-8")
        row = execute(t, gold, corpus_text=corpus,
                      extra=dict(target_pct=it["target_pct"], actual_pct=it["actual_pct"], pos_doc_id=it["doc_id"]))
        # документ-носитель в варианте другой: цитата верна, если назван документ позиции
        row["citation_correct"] = it["doc_id"] in row["doc_ids"]
        return row
    return tasks, fn
