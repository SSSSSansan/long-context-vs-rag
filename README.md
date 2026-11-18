# Long Context vs RAG — benchmark

Research-проект R04 (KBTU X-DAI, Document AI): при каких объёмах документов и типах вопросов long context превосходит RAG и наоборот.

**Сравниваем:** LC (весь корпус в промпт) · BM25+LLM · Embedding RAG+LLM
**Корпуса:** 8K / 30K / 100K токенов (вложенные)
**Вопросы:** T1 простой факт · T2 сравнение двух документов · T3 поиск противоречий
**Метрики:** accuracy, evidence recall, citation correctness, latency, $/correct
**Ключевая идея:** факты внедряются в корпус → gold известен по построению → retrieval-метрики точные.

Подробно: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Полный план — в `R04_LONG_CONTEXT_VS_RAG_CONTEXT.md`.

## Статус V2 (18 ноября): gpt-6-luna, шаги Ш1–Ш7 по PLAN_V2 выполнены
- [x] Наборы вопросов A (с якорем), A-hard (без якоря), B; NO-CONTEXT; HYB-RRF и HYB-rerank; AGENT (инструменты: поиск, чанк, расчёт, запись в Notion-мок)
- [x] Прогоны Ш1–Ш7 на `gpt-6-luna` (расход $1.06 из $3.00); таблица [results/main_table_v2.md](results/main_table_v2.md), парные тесты Макнемара + Холм `results/mcnemar_v2_*.csv`, безубыточность `results/breakeven_*.csv`
- [x] Метрика: судья И `value_in_answer` (T2: число победителя, строгая версия — диагностика); отказ не бывает correct
- [x] Учёт записи в кэш (`cache_write_tokens`, ставка 1.25× input) добавлен в `compute_cost` и смету; расход Ш1–Ш7 ≈ $1.23 вместо $1.06 (оценка, ждёт сверки с дашбордом OpenAI)
- [x] Набор B (42 вопроса с якорем, 100K, 1 повтор): [results/main_table_b.md](results/main_table_b.md); считается отдельно от A и A-hard
- [x] P1 (положение факта в LC) на Luna: 30K (13 вопросов) и 100K (6 вопросов), позиции 0/50/100% — [results/p1_luna.md](results/p1_luna.md); эффекта не обнаружено (малое n, потолок)
- [ ] Не сделано: B-hard (нужна пересборка корпуса), P1 на Luna, другие модели, Ragas, видео. Все числа предварительные (калибровка судьи n=40, не независимая разметка)
- Воспроизведение V2: `LLM_BACKEND=openai python -m src.eval.run --exp configs/experiments/v2_s<N>_*.yaml --workers 2 --out runs/v2_s<N>` (под `cost_guard`), затем `python -m scripts.build_main_table --v2 && python -m scripts.mcnemar_v2`. Локально бесплатно: `LLM_BACKEND=local`, `--repeats 1 --limit 12`.

## Статус V1 (промежуточный, на 10 ноября)
- [x] День 1.1 — каркас репозитория, `llm_client`, `cost_guard`, кэш, конфиги
- [x] День 1.2 — синтетический корпус (48 документов), вложенные 8K/30K/100K, чанкинг
- [x] День 2 — факты, gold, вопросы, leakage-check (0/60 утечек), позиционные корпуса
- [x] День 3 — LC / BM25 / EMB + единый `answer.py`
- [x] День 4 — метрики, судья, уровень 1 (gpt-4o-mini ×3 повтора; gpt-4o ×1)
- [x] День 5, частично — 100K ограниченно (gpt-4o: 9 LC / 3 BM25; gpt-4o-mini: 46 вызовов LC, 91% отказов), позиции P1 (gpt-4o, 30K — 3 факта, 100K — 1 факт), ablation по выровненному бюджету (30K, 6 вопросов)
- [x] Диагностика отказов gpt-4o-mini на T1 — [results/diagnosis_abstain_T1.md](results/diagnosis_abstain_T1.md): гипотеза о механизме, не доказанный факт
- [x] Доверительные интервалы Уилсона 95% по уникальным вопросам во всех `results/*.csv`
- [x] Главная таблица — [results/main_table.md](results/main_table.md) / `.csv`
- [ ] **Не сделано:** P2 (позиция gold-чанка в RAG top-k), HYB (BM25+EMB+reranker), Ragas, роутер по типу вопроса, локальный backend как второе измерение (API vs on-prem), демо-видео
- [x] Калибровка судьи — 40 примеров размечены вслепую от вердиктов судьи (`results/judge_calibration_for_human.csv`): с помощью Claude и проверкой автором; согласие 85% (κ=0.78, по «верно/неверно» 100%), выборка небольшая (n=40)
- [ ] День 6–7 — финальная вычитка, видео

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
