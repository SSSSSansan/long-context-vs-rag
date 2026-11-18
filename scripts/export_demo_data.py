"""Выгрузка компактных данных для демо (demo/data/): без промптов, только готовые результаты. Без API и без пересчёта моделей.
python -m scripts.export_demo_data"""
from __future__ import annotations

import csv
import glob
import json
import re
from pathlib import Path

from src.config import ROOT
from src.eval.value_check import value_in_answer, value_in_answer_strict
from src.methods.bm25 import chunks_of

OUT = ROOT / "demo/data"
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = [(d, None) for d in sorted(glob.glob(str(ROOT / "runs/v2_s[1-7]")))] + [(str(ROOT / "runs/v2_b_main"), "B")]
questions = {q["qid"]: q for q in json.loads((ROOT / "data/questions.json").read_text(encoding="utf-8"))["questions"]}
gold = json.loads((ROOT / "data/gold.json").read_text(encoding="utf-8"))["facts"]


def qset_of(q: dict) -> str:
    return "A-hard" if not q.get("anchor", True) else q.get("set", "A")


def cold(r: dict) -> float:
    """Холодная цена вызова (кэш не сработал): промпт >=1024 токенов пишется в кэш по 0.125, выход 0.50, плюс судья."""
    p = r["prompt_tokens"]
    return (p * (0.125 if p >= 1024 else 0.10) + r["completion_tokens"] * 0.50) / 1e6 + (r.get("judge_cost_usd") or 0.0)


def row_out(r: dict, src: str) -> dict:
    q = questions[r["qid"]]
    ab = bool(r["abstained"])
    judge_ok = r["verdict"] == "correct" and not ab
    ans, quote = r.get("answer") or "", r.get("quote") or ""
    main = judge_ok and value_in_answer(r["gold_answer"], ans, quote)
    strict = judge_ok and value_in_answer_strict(r["gold_answer"], ans, quote)
    chunk_ids = r.get("chunk_ids") or []
    return dict(
        qid=r["qid"], qset=qset_of(q), size=r["size"], method=r["method"], type=r["type"], repeat=r["repeat"],
        answer=ans, quote=quote, verdict=r["verdict"], abstained=ab, judge_ok=int(judge_ok), main_ok=int(bool(main)), strict_ok=int(bool(strict)),
        evidence_recall=r.get("evidence_recall"), citation=r.get("citation_correct"), quote_grounded=r.get("quote_grounded"),
        gen_ms=r.get("generation_ms"), retr_ms=r.get("retrieval_ms"), prompt_tokens=r["prompt_tokens"], context_tokens=r.get("context_tokens"),
        cost_measured=(r.get("cost_usd") or 0.0) + (r.get("judge_cost_usd") or 0.0), cost_cold=cold(r), from_cache=bool(r.get("from_cache")),
        chunk_ids=chunk_ids if r["method"] != "lc" else [], fact_pos=(r.get("fact_positions_pct") or [None])[0],
        n_steps=r.get("n_steps"), n_tool_calls=r.get("n_tool_calls"), source=src)


rows = []
for d, tag in SOURCES:
    for line in open(Path(d) / "raw.jsonl", encoding="utf-8"):
        rows.append(row_out(json.loads(line), Path(d).name))
(OUT / "rows.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")

# P1 и P2 отдельно
def special(dirs, extra):
    out = []
    for d in dirs:
        for line in open(ROOT / d / "raw.jsonl", encoding="utf-8"):
            r = json.loads(line); x = row_out(r, Path(d).name)
            x.update({k: r.get(k) for k in extra}); out.append(x)
    return out
p1 = special(["runs/v2_p1_30k", "runs/v2_p1_100k"], ["target_pct", "actual_pct"])
p2 = special(["runs/v2_p2_luna"], ["gold_rank"])
(OUT / "p1.json").write_text(json.dumps(p1, ensure_ascii=False), encoding="utf-8")
(OUT / "p2.json").write_text(json.dumps(p2, ensure_ascii=False), encoding="utf-8")

# вопросы + золотые факты + тексты показанных фрагментов (для подсветки)
by_fact = {f["fact_id"]: f for f in gold}
qout = {}
for qid, q in questions.items():
    qout[qid] = dict(question=q["question"], gold_answer=q["gold_answer"], type=q["type"], qset=qset_of(q), first_corpus=q["first_corpus"],
                     facts=[by_fact[i]["statement"] for i in q["fact_ids"]], gold_chunk_ids=sorted({c for i in q["fact_ids"] for c in by_fact[i]["chunk_ids"]}))
(OUT / "questions.json").write_text(json.dumps(qout, ensure_ascii=False), encoding="utf-8")
need = {}
for x in rows + p1 + p2:
    for c in x["chunk_ids"]:
        need.setdefault(x["size"], set()).add(c)
for q in qout.values():
    for c in q["gold_chunk_ids"]:
        need.setdefault("100k", set()).add(c)
chunks = {}
for size in ("8k", "30k", "100k"):
    by_id = {c.chunk_id: c.text for c in chunks_of(size)}
    chunks[size] = {c: by_id[c] for c in sorted(need.get(size, set()) | (need.get("100k", set()) if size != "100k" else set())) if c in by_id}
(OUT / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")

# готовые таблицы из результатов репозитория
for name in ("mcnemar_v2_main", "mcnemar_v2_strict", "mcnemar_b_main", "mcnemar_b_strict", "breakeven_gpt-6-luna", "breakeven_gpt-6.1-terra"):
    src = ROOT / "results" / f"{name}.csv"
    if src.exists():
        (OUT / f"{name}.csv").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
meta = dict(spent_usd=json.loads((ROOT / "runs/budget_v2.json").read_text())["spent_usd"], model="gpt-6-luna", rows=len(rows), p1=len(p1), p2=len(p2))
(OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
print(meta)
for f in sorted(OUT.iterdir()):
    print(f"{f.name:34s}{f.stat().st_size/1024:9.0f} КБ")
