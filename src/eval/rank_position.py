"""P2: позиция gold-чанка в top-k (PLAN_V2 §2.2, долг V1). Аналог lost-in-the-middle для RAG.

Контекст всегда из k=10 чанков; gold-чанк стоит на ранге r in {1, 5, 10}, остальные 9 — трудные дистракторы (лучшие
по BM25 для этого вопроса, без самого факта и без его связки). Меняется ТОЛЬКО ранг gold-чанка; формат контекста тот же
(format_chunks), промпт тот же. Только T1 (один gold-чанк); T2/T3 требуют двух gold-чанков и в P2 не входят.
Строки результатов: method="p2_bm25", gold_rank, k_label=10; в основные таблицы не попадают (aggregate: summary_p2).
"""
from __future__ import annotations

from ..methods import Context, bm25, format_chunks
from .run import execute, filter_set

K = 10
POOL = 80          # сколько BM25-кандидатов просматриваем, чтобы набрать 9 дистракторов


def build_ranked_context(size: str, q: dict, gold: dict, rank: int, k: int = K) -> Context:
    if len(q["fact_ids"]) != 1:
        raise ValueError(f"P2 только для T1 (один gold-факт): {q['qid']}")
    if not 1 <= rank <= k:
        raise ValueError(f"ранг {rank} вне 1..{k}")
    fact = gold[q["fact_ids"][0]]
    chunks = bm25.chunks_of(size)
    by_id = {c.chunk_id: c for c in chunks}
    gid = next(c for c in fact["chunk_ids"] if c in by_id)
    g = by_id[gid]
    banned = [fact["statement"]] + [gold[i]["statement"] for i in q.get("link_fact_ids", [])]
    distract = [chunks[i] for i in bm25.rank(size, q["question"], POOL)
                if chunks[i].chunk_id != gid and not any(s in chunks[i].text for s in banned)][: k - 1]
    if len(distract) < k - 1:
        raise RuntimeError(f"{q['qid']}: нашлось {len(distract)} дистракторов из {k - 1} (увеличь POOL)")
    ordered = distract[: rank - 1] + [g] + distract[rank - 1:]
    return Context(text=format_chunks(ordered), chunk_ids=[c.chunk_id for c in ordered])


def rank_tasks(spec, gold, qs):
    tasks = []
    for r in spec["runs"]:
        pool = [q for q in filter_set(qs, r.get("question_set")) if q["type"] == "T1"]
        pool = sorted(pool, key=lambda q: q["qid"])[: r.get("n_questions", 99)]
        for size in r["sizes"]:
            for q in pool:
                for rank in r["ranks"]:
                    for rep in range(1, r["repeats"] + 1):
                        tasks.append(dict(size=size, method="p2_bm25", k=K, k_label=K, q=q, repeat=rep, pos=rank,
                                          question_set=r.get("question_set")))

    def fn(t):
        ctx = build_ranked_context(t["size"], t["q"], gold, t["pos"])
        return execute(t, gold, ctx=ctx, extra=dict(gold_rank=t["pos"]))
    return tasks, fn
