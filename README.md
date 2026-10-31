# Long Context vs RAG — benchmark

Research-проект R04 (KBTU X-DAI, Document AI): при каких объёмах документов и типах вопросов long context превосходит RAG и наоборот.

**Сравниваем:** LC (весь корпус в промпт) · BM25+LLM · Embedding RAG+LLM
**Корпуса:** 8K / 30K / 100K токенов (вложенные)
**Вопросы:** T1 простой факт · T2 сравнение двух документов · T3 поиск противоречий
**Метрики:** accuracy, evidence recall, citation correctness, latency, $/correct
**Ключевая идея:** факты внедряются в корпус → gold известен по построению → retrieval-метрики точные.

Подробно: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Полный план — в `R04_LONG_CONTEXT_VS_RAG_CONTEXT.md`.

## Статус
- [x] День 1.1 — каркас репозитория, `llm_client`, `cost_guard`, кэш, конфиги
- [ ] День 1.2 — корпус (30–50 документов) и чанкинг
- [ ] День 2 — факты, gold, вопросы, leakage-check
- [ ] День 3 — LC / BM25 / EMB + единый `answer.py`
- [ ] День 4 — метрики, судья, уровень 1
- [ ] День 5 — 100K, позиции, ablation по бюджету
- [ ] День 6–7 — отчёт, видео

## Быстрый старт

```bash
git clone <repo-url> && cd long-context-vs-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # по умолчанию LLM_BACKEND=local

python -m scripts.estimate_cost # смета до запуска
make smoke                      # проверка клиента на локальной модели
pytest -q                       # тесты кэша и бюджета
```

## Backend'ы

| `LLM_BACKEND` | Что | Когда |
|---|---|---|
| `local` | vLLM (OpenAI-совместимый сервер), `bge-m3` для эмбеддингов | отладка и все черновые прогоны — бесплатно |
| `openai` | мини-модель + `text-embedding-3-small` | только финальные зачётные прогоны, под `cost_guard` |

Для `openai` обязательно задать `OPENAI_API_KEY` и `BUDGET_USD_LIMIT` в `.env`.

## Воспроизводимость
Параметры — в `configs/*.yaml`; `seed` фиксирован; версии моделей — полные идентификаторы; каждый вызов логируется в `runs/<ts>/raw.jsonl`; зависимости — `requirements.lock.txt`.
Токены корпусов считаются токенайзером, указанным в `configs/base.yaml` (`tokenizer`).

## Происхождение данных
Заполняется в `corpus/raw/SOURCES.md` (источник и лицензия каждого документа).
