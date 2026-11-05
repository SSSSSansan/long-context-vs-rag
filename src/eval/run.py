"""Раннер экспериментов. Каждый вызов -> строка runs/<ts>/raw.jsonl (полный промпт, ответ, usage, latency, метрики).

  python -m src.eval.run --exp configs/experiments/level1_8k_30k.yaml [--workers 4] [--limit N] [--out runs/NAME]

Конфиг эксперимента: runs: [{sizes, methods, question_types, n_per_type, repeats, ks}]  (ks: числа или "aligned")
или mode: position (см. run_position). Повторный запуск тех же вызовов бесплатен (кэш ответов).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import yaml

from .. import cost_guard
from ..answer import answer_question, build_messages
from ..config import ROOT, backend_name, base_cfg
from ..methods.bm25 import chunks_of
from ..methods.long_context import load_corpus
from ..methods.registry import METHODS
from ..tokens import count_tokens
from .exact_metrics import citation_correct, evidence_recall, quote_grounded
from .judge import judge

SIZE_ORDER = ["8k", "30k", "100k"]
_lock = threading.Lock()


def load_data():
    gold = {f["fact_id"]: f for f in json.loads((ROOT / "data/gold.json").read_text(encoding="utf-8"))["facts"]}
    qs = json.loads((ROOT / "data/questions.json").read_text(encoding="utf-8"))["questions"]
    return gold, qs


def pick_questions(qs, size, qtype, n, seed):
    """Вопросы, чьи факты уже есть в корпусе `size`; детерминированная выборка по seed (без перекоса к началу корпуса)."""
    avail = sorted((q for q in qs if q["type"] == qtype
                    and SIZE_ORDER.index(q["first_corpus"]) <= SIZE_ORDER.index(size)), key=lambda q: q["qid"])
    return random.Random(f"{seed}-{size}-{qtype}").sample(avail, min(n, len(avail)))


def aligned_k(size: str) -> int:
    """k, при котором RAG занимает бюджет токенов LC (раздел 2.3 плана)."""
    cs = chunks_of(size)
    return math.ceil(count_tokens(load_corpus(size)) / (sum(c.n_tokens for c in cs) / len(cs)))


def expand_tasks(spec: dict, qs, seed):
    tasks = []
    for r in spec["runs"]:
        for size in r["sizes"]:
            for method in r["methods"]:
                ks = [None] if method == "lc" else r.get("ks", [base_cfg()["retrieval"]["top_k"]])
                for k in ks:
                    kk = aligned_k(size) if k == "aligned" else k
                    for qt in r.get("question_types", ["T1", "T2", "T3"]):
                        for q in pick_questions(qs, size, qt, r["n_per_type"], seed):
                            for rep in range(1, r["repeats"] + 1):
                                tasks.append(dict(size=size, method=method, k=kk, k_label=k, q=q, repeat=rep))
    return tasks


def pred_text(a) -> str:
    """Что видит судья: ответ + флаг противоречия + цитата (метод ему не виден)."""
    if a.abstained:
        return a.text
    return f"{a.answer}\nПротиворечие найдено: {'да' if a.contradiction_found else 'нет'}\nЦитата: {a.quote}"


def execute(t, gold, corpus_text=None, extra=None) -> dict:
    q, size, method, k = t["q"], t["size"], t["method"], t["k"]
    facts = [gold[f] for f in q["fact_ids"]]
    ctx = METHODS[method](size, q["question"], k, corpus_text) if corpus_text is not None else METHODS[method](size, q["question"], k)
    salt = f"r{t['repeat']}"
    a = answer_question(ctx.text, q["question"], salt=salt)
    verdict, why, jr = judge(q["question"], q["gold_answer"], pred_text(a), salt=salt)
    r = a.resp
    row = dict(
        qid=q["qid"], type=q["type"], size=size, method=method, k=k, k_label=t["k_label"], repeat=t["repeat"],
        backend=backend_name(), model=r.model,
        fact_positions_pct=[f["positions_pct"].get(size) for f in facts],
        context_tokens=count_tokens(ctx.text), chunk_ids=ctx.chunk_ids,
        answer=a.answer, doc_ids=a.doc_ids, quote=a.quote, contradiction_found=a.contradiction_found,
        parsed_ok=a.parsed_ok, abstained=a.abstained, raw_text=a.text,
        gold_answer=q["gold_answer"], verdict=verdict, judge_rationale=why,
        evidence_recall=evidence_recall(facts, ctx), citation_correct=citation_correct(facts, a.doc_ids),
        quote_grounded=quote_grounded(a.quote, ctx.text),
        prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens, cached_tokens=r.cached_tokens,
        retrieval_ms=round(ctx.retrieval_ms, 2), generation_ms=round(r.latency_ms, 1),
        cost_usd=r.cost_usd, judge_cost_usd=jr.cost_usd, from_cache=r.from_local_cache,
        prompt=build_messages(ctx.text, q["question"])[0]["content"],
    )
    if extra:
        row.update(extra)
    return row


def task_key(t) -> tuple:
    return (t["q"]["qid"], t["size"], t["method"], t["k_label"], t["repeat"], t.get("pos"))


def row_key(r) -> tuple:
    return (r["qid"], r["size"], r["method"], r.get("k_label"), r["repeat"], r.get("target_pct"))


def run_tasks(tasks, fn, out_path: Path, workers: int, label: str):
    if out_path.exists():   # докатываем прерванный прогон: уже записанные вызовы не повторяем
        have = {row_key(json.loads(l)) for l in out_path.read_text(encoding="utf-8").splitlines()}
        tasks = [t for t in tasks if task_key(t) not in have]
        print(f"[{label}] уже есть {len(have)}, осталось {len(tasks)}")
    done = 0
    stop = threading.Event()
    t0 = time.time()
    with out_path.open("a", encoding="utf-8") as f:
        def work(t):
            nonlocal done
            if stop.is_set():
                return None
            try:
                row = fn(t)
            except Exception as e:      # кончились кредиты/бюджет: останавливаем весь прогон сразу
                if "insufficient_quota" in str(e) or "Бюджет исчерпан" in str(e):
                    stop.set()
                    print(f"[{label}] СТОП: {str(e)[:120]}", flush=True)
                    return None
                raise
            with _lock:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                done += 1
                if done % 10 == 0 or done == len(tasks):
                    print(f"[{label}] {done}/{len(tasks)}  {time.time() - t0:.0f}s  {cost_guard.report() if backend_name() == 'openai' else ''}",
                          flush=True)
            return row
        if workers <= 1:
            for t in tasks:
                work(t)
        else:
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(work, tasks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="только первые N вызовов (smoke)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    spec = yaml.safe_load(Path(args.exp).read_text(encoding="utf-8"))
    seed = base_cfg()["seed"]
    gold, qs = load_data()
    out_dir = ROOT / (args.out or f"runs/{datetime.now():%Y%m%d_%H%M%S}_{spec['name']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        dict(spec=spec, backend=backend_name(), base=base_cfg(), seed=seed), ensure_ascii=False, indent=2), encoding="utf-8")

    if spec.get("mode") == "position":
        from .position import position_tasks
        tasks, fn = position_tasks(spec, gold, qs)
    else:
        tasks = expand_tasks(spec, qs, seed)
        fn = lambda t: execute(t, gold)
    # LC идёт группами по размеру подряд: так срабатывает prompt cache провайдера
    tasks.sort(key=lambda t: (t["size"], t["method"] != "lc", t.get("pos", 0), t["q"]["qid"], t["repeat"]))
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"{spec['name']}: {len(tasks)} вызовов, backend={backend_name()}, out={out_dir}")
    run_tasks(tasks, fn, out_dir / "raw.jsonl", args.workers, spec["name"])
    print("готово:", out_dir / "raw.jsonl")


if __name__ == "__main__":
    main()
