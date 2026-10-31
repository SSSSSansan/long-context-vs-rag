"""Смета ДО запуска. Цены и модель берутся из configs/base.yaml (backend=openai).

    python -m scripts.estimate_cost

Подставь реальные размеры контекста RAG после первых smoke-тестов (rag_ctx_tokens).
"""
from __future__ import annotations

from src.config import base_cfg

P = base_cfg()["backends"]["openai"]["price_per_1m_usd"]
OUT_TOK = 150          # типичный JSON-ответ
RAG_CTX = 3_000        # ~top_k * chunk_tokens + промпт
JUDGE_TOK = 1_000


def cost(n_calls: int, in_tok: int, out_tok: int = OUT_TOK, cache_share: float = 0.0) -> float:
    cached = in_tok * cache_share
    return n_calls * ((in_tok - cached) * P["input"] + cached * P["cached_input"] + out_tok * P["output"]) / 1e6


rows = []
# Уровень 1: 60 вопросов x 2 размера x 3 повтора
for size, lc_tok in (("8k", 8_000), ("30k", 30_000)):
    n = 60 * 3
    rows.append((f"L1 LC {size}", n, cost(n, lc_tok)))
    rows.append((f"L1 BM25+EMB {size}", n * 2, cost(n * 2, RAG_CTX)))
# Уровень 2: 100K, 24 вопроса x 2 повтора
rows.append(("L2 LC 100k", 48, cost(48, 100_000)))
rows.append(("L2 LC 100k (с prompt cache ~90%)", 48, cost(48, 100_000, cache_share=0.9)))
# Уровень 3: позиции
rows.append(("L3 P1 LC 30k", 5 * 5 * 3, cost(75, 30_000)))
rows.append(("L3 P1 LC 100k", 15, cost(15, 100_000)))
rows.append(("L3 P2 EMB", 45, cost(45, RAG_CTX)))
# Судья
rows.append(("Judge ~2000", 2000, cost(2000, JUDGE_TOK, out_tok=60)))

total = 0.0
print(f"{'статья':40s}{'вызовов':>9s}{'$':>9s}")
for name, n, c in rows:
    if "prompt cache" in name:
        print(f"{name:40s}{n:9d}{c:9.3f}   (справочно, в итог не входит)")
        continue
    total += c
    print(f"{name:40s}{n:9d}{c:9.3f}")
print(f"{'ИТОГО (без кэша, верхняя оценка)':40s}{'':9s}{total:9.3f}")
