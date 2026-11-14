"""Шаг 2.1: «на 5 вопросах глазами: реранкер поднимает gold-чанк выше, чем RRF». Платных вызовов нет.

    EMBED_BACKEND=local python -m scripts.check_hyb [--size 30k] [--n 5]

Нужны: bge-m3 (эмбеддинги, как у EMB) и BAAI/bge-reranker-v2-m3 (скачается при первом запуске).
Для каждого вопроса набора A-hard печатает ранг худшего gold-чанка вопроса (1 = первый, '-' = не попал в top-k) у BM25 / EMB / HYB-RRF / HYB-RERANK.
"""
from __future__ import annotations

import argparse
import json

from src.config import ROOT, base_cfg
from src.methods import bm25, embedding_rag, hybrid

N_RANK = 40


def gold_rank(order: list[str], golds: list[set[str]]) -> str:
    """Ранг ХУДШЕГО из gold-фактов вопроса (у T2/T3 нужны оба): все факты найдены — максимум рангов, иначе '-'."""
    ranks = []
    for g in golds:
        r = next((i for i, cid in enumerate(order, 1) if cid in g), None)
        if r is None:
            return "-"
        ranks.append(r)
    return str(max(ranks))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="30k")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--type", default="T1", choices=["T1", "T2", "T3"])
    a = ap.parse_args()
    qs = [q for q in json.loads((ROOT / "data/questions.json").read_text(encoding="utf-8"))["questions"]
          if q["set"] == "A" and not q.get("anchor", True) and q["type"] == a.type][: a.n]
    gold = {f["fact_id"]: f for f in json.loads((ROOT / "data/gold.json").read_text(encoding="utf-8"))["facts"]}
    ch = bm25.chunks_of(a.size)
    k = base_cfg()["retrieval"]["top_k"]
    print(f"размер {a.size}, k={k}; ранг gold-чанка (меньше = лучше):")
    print(f"{'qid':14s}{'BM25':>6s}{'EMB':>6s}{'RRF':>6s}{'RERANK':>8s}")
    for q in qs:
        g = [set(gold[i]["chunk_ids"]) for i in q["fact_ids"]]
        rows = [
            [ch[i].chunk_id for i in bm25.rank(a.size, q["question"], N_RANK)],
            [ch[i].chunk_id for i in embedding_rag.rank(a.size, q["question"], N_RANK)],
            [c.chunk_id for c in hybrid.retrieve(a.size, q["question"], N_RANK, False)],
            [c.chunk_id for c in hybrid.retrieve(a.size, q["question"], N_RANK, True)],
        ]
        print(f"{q['qid']:14s}" + "".join(f"{gold_rank(r, g):>{w}s}" for r, w in zip(rows, (6, 6, 6, 8))))


if __name__ == "__main__":
    main()
