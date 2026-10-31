# Архитектура

## Исследовательский вопрос
При каких объёмах документов и типах вопросов long context превосходит RAG и наоборот? (R04, KBTU X-DAI, Document AI)

## Три инварианта дизайна
1. **Факты внедряются, а не размечаются.** Реальные документы — фон; в них вставляются контролируемые факты с известными `doc_id`/`chunk_id`/позицией. Gold известен по построению → evidence recall и citation correctness считаются **детерминированно**. LLM-судья нужен только для accuracy.
2. **Меняется только `{context}`.** Один промпт, одна модель, одна температура, один судья для LC / BM25 / EMB.
3. **Выравнивание по токенам.** Дополнительно к основному прогону — RAG с таким k, чтобы занять тот же бюджет токенов, что и LC.

## Поток данных

```
corpus/raw/*.txt ──► build_corpus.py (+ plant_facts.py)
                        │  вложенные корпуса 8K ⊂ 30K ⊂ 100K, точный счёт токенов
                        │  вставка фактов → data/gold.json
                        ▼
              chunking.py (стабильные chunk_id)
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   methods/LC      methods/BM25     methods/EMB
   весь корпус     rank_bm25 top-k  Qdrant + bge-m3 top-k
        └───────────────┼───────────────┘
                        ▼
                 answer.py  — ОДИН промпт, различается только context
                        │  (всё через llm_client.complete)
                        ▼
                 runs/<ts>/raw.jsonl — строка на вызов (промпт, ответ, usage, latency)
                        ▼
        eval/exact_metrics.py   evidence recall, citation, abstention  (детерминированно)
        eval/judge.py           accuracy (LLM-судья, слепой к методу)
        eval/aggregate.py       $/correct, latency p50/p95, mean±std по 3 повторам
                        ▼
                 results/*.csv + results/charts/
```

## Слой LLM (src/)

| Модуль | Роль |
|---|---|
| `config.py` | единственное место чтения yaml/.env |
| `llm_client.py` | `complete()` / `embed()`; backend `openai` или `local` (vLLM); возвращает tokens/cached/latency/cost |
| `cache.py` | дисковый кэш ответов по хэшу (backend+model+prompt+params+salt) |
| `cost_guard.py` | жёсткий лимит `BUDGET_USD_LIMIT`, счётчик в `runs/budget.json` |

### Как экономятся токены OpenAI
- `LLM_BACKEND=local` по умолчанию — вся отладка бесплатна; `openai` включается явно только для финальных прогонов.
- Без `BUDGET_USD_LIMIT` платные вызовы заблокированы.
- Кэш ответов: перезапуск после падения не тратит деньги повторно.
- Prompt caching провайдера: `{context}` стоит в начале промпта, вопрос в конце; `cached_tokens` логируется.
- Порядок запуска строго от дешёвого к дорогому: 8K → 30K → позиции 30K → 100K → позиции 100K.
- `scripts/estimate_cost.py` — смета до первого платного вызова.

## Отклонения от структуры в задании
`cost_guard.py` лежит в `src/`, а не в `scripts/`, чтобы `llm_client` мог его импортировать без хаков с путями. Остальное — как в плане (раздел 10).

## Порядок реализации
День 1 — каркас (этот коммит) → корпус + чанкинг. День 2 — факты/gold/вопросы/leakage. День 3 — три метода + `answer.py`. День 4 — eval + уровень 1. День 5 — 100K + позиции + ablation. День 6 — отчёт. День 7 — видео + буфер.
