"""Шаг 1б: изолируем причину 100%-отказа gpt-4o-mini на T1 (q_T1_001, q_T1_002).

Варианты A-E меняют ТОЛЬКО {context}, промпт из src/answer.py не трогаем (единый шаблон).
A - одно предложение-факт
B - чанк с фактом (~300 ток., как в реальном BM25/EMB top-k)
C - весь документ с фактом
D - корпус 8K целиком (как в текущем LC-прогоне)
E - BM25 top-8 (как в текущем BM25-прогоне)

Backend берётся из LLM_BACKEND (.env): local (бесплатно, qwen2.5-7b-32k) или openai
(платно, под отдельным счётчиком runs/budget_interim.json, НЕ основным runs/budget.json).

Запуск:
  LLM_BACKEND=local python -m scripts.diagnose_abstain_causes            # бесплатно
  LLM_BACKEND=openai python -m scripts.diagnose_abstain_causes --estimate-only
  LLM_BACKEND=openai python -m scripts.diagnose_abstain_causes --confirm  # платные вызовы
"""
from __future__ import annotations

import argparse
import json
import os

from src import cost_guard
from src.answer import answer_question
from src.chunking import chunk_corpus, split_docs
from src.config import ROOT, backend_name
from src.methods import format_chunks
from src.methods.bm25 import build_context as bm25_build_context
from src.methods.long_context import load_corpus
from src.tokens import count_tokens

INTERIM_BUDGET_PATH = ROOT / "runs" / "budget_interim.json"
INTERIM_LIMIT_USD = 1.5
THIS_DIAGNOSTIC_CAP_USD = 0.05  # лимит именно на этот прогон (в сообщении пользователя)

QUESTIONS = {
    "q_T1_001": {
        "question": "Каков размер ежемесячной выплаты участникам олимпиады «Ксенар-28»?",
        "gold_answer": "114 МРП",
        "statement": "Размер ежемесячной выплаты участникам олимпиады «Ксенар-28» составляет 114 МРП.",
        "doc_id": "doc_002",
        "chunk_id": "doc_002_c01",
    },
    "q_T1_002": {
        "question": "Какова продолжительность стипендии «Нерида-15»?",
        "gold_answer": "10 недель",
        "statement": "Продолжительность стипендии «Нерида-15» составляет 10 недель.",
        "doc_id": "doc_001",
        "chunk_id": "doc_001_c07",
    },
}


def build_variants(qid: str) -> dict[str, str]:
    q = QUESTIONS[qid]
    corpus_text = load_corpus("8k")
    docs = dict(split_docs(corpus_text))
    chunks = {c.chunk_id: c for c in chunk_corpus(corpus_text)}

    a_sentence = q["statement"]
    b_chunk = format_chunks([chunks[q["chunk_id"]]])
    c_doc = "\n".join(docs[q["doc_id"]])
    d_corpus = corpus_text
    e_bm25 = bm25_build_context("8k", q["question"], 8).text

    return {"A_sentence": a_sentence, "B_chunk": b_chunk, "C_document": c_doc,
            "D_corpus_8k": d_corpus, "E_bm25_top8": e_bm25}


def use_interim_budget() -> None:
    """Подменяет путь cost_guard на отдельный файл для этой диагностики,
    НЕ трогая основной runs/budget.json (там уже $6.03 прошлых прогонов)."""
    cost_guard._path = lambda: INTERIM_BUDGET_PATH  # noqa: SLF001 — осознанная подмена для изоляции бюджета
    os.environ["BUDGET_USD_LIMIT"] = str(INTERIM_LIMIT_USD)


def estimate_cost_usd(variants_by_q: dict[str, dict[str, str]], calls_per_variant: int = 2) -> dict:
    # цены gpt-4o-mini-2024-07-18 из configs/base.yaml: input 0.15, output 0.6 за 1M токенов
    PRICE_IN, PRICE_OUT = 0.15, 0.60
    TEMPLATE_OVERHEAD_TOKENS = 120   # правила + "Верни JSON" + вопрос, грубая оценка
    EST_COMPLETION_TOKENS = 150       # по наблюдённым raw_text (30-200 символов) с запасом

    rows = []
    total_prompt_tok = 0
    total_completion_tok = 0
    for qid, variants in variants_by_q.items():
        for name, ctx in variants.items():
            ctx_tok = count_tokens(ctx)
            prompt_tok = ctx_tok + TEMPLATE_OVERHEAD_TOKENS
            calls = calls_per_variant
            total_prompt_tok += prompt_tok * calls
            total_completion_tok += EST_COMPLETION_TOKENS * calls
            rows.append((qid, name, ctx_tok, prompt_tok, calls))

    cost = (total_prompt_tok * PRICE_IN + total_completion_tok * PRICE_OUT) / 1_000_000
    n_calls = sum(r[4] for r in rows)
    return {"rows": rows, "n_calls": n_calls, "total_prompt_tok": total_prompt_tok,
            "total_completion_tok": total_completion_tok, "cost_usd": cost}


def run_variants(variants_by_q: dict[str, dict[str, str]], calls_per_variant: int = 2) -> list[dict]:
    results = []
    for qid, variants in variants_by_q.items():
        q = QUESTIONS[qid]
        for name, ctx in variants.items():
            ctx_tok = count_tokens(ctx)
            for i in range(1, calls_per_variant + 1):
                salt = f"diag-{name}-{qid}-r{i}"
                a = answer_question(ctx, q["question"], salt=salt)
                results.append({
                    "qid": qid, "variant": name, "call": i,
                    "context_tokens": ctx_tok, "abstained": a.abstained,
                    "parsed_ok": a.parsed_ok, "raw_text": a.text,
                    "cost_usd": a.resp.cost_usd, "from_cache": a.resp.from_local_cache,
                })
    return results


def print_table(results: list[dict]) -> None:
    print(f"\n{'qid':<12}{'вариант':<14}{'вызов':<7}{'контекст,ток':<14}{'отказ':<8}сырой ответ")
    for r in results:
        print(f"{r['qid']:<12}{r['variant']:<14}{r['call']:<7}{r['context_tokens']:<14}"
              f"{'ДА' if r['abstained'] else 'нет':<8}{r['raw_text'][:160]!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimate-only", action="store_true", help="только смета, без вызовов (для backend=openai)")
    ap.add_argument("--confirm", action="store_true", help="подтверждение запуска платных вызовов (backend=openai)")
    args = ap.parse_args()

    variants_by_q = {qid: build_variants(qid) for qid in QUESTIONS}

    backend = backend_name()
    if backend == "local":
        print("backend=local (qwen2.5-7b-32k) — бесплатно, запускаю без подтверждения.")
        results = run_variants(variants_by_q)
        print_table(results)
        return

    # backend == openai
    est = estimate_cost_usd(variants_by_q)
    print("=== Смета (gpt-4o-mini-2024-07-18) ===")
    for qid, name, ctx_tok, prompt_tok, calls in est["rows"]:
        print(f"  {qid} / {name}: контекст {ctx_tok} ток, промпт~{prompt_tok} ток x{calls} вызова")
    print(f"Итого вызовов: {est['n_calls']} (лимит сообщения: 30)")
    print(f"Итого промпт-токенов (оценка): {est['total_prompt_tok']}")
    print(f"Итого completion-токенов (оценка, с запасом): {est['total_completion_tok']}")
    print(f"Оценка стоимости: ${est['cost_usd']:.4f} (лимит этого прогона: ${THIS_DIAGNOSTIC_CAP_USD:.2f}, "
          f"общий интерим-бюджет: ${INTERIM_LIMIT_USD:.2f})")

    if est["cost_usd"] > THIS_DIAGNOSTIC_CAP_USD:
        print("СТОП: оценка превышает лимит этого диагностического прогона ($0.05). Не запускаю без явного решения.")
        return

    if args.estimate_only or not args.confirm:
        print("\nЭто только смета. Повторите запуск с --confirm после подтверждения в чате.")
        return

    use_interim_budget()
    print(f"\nБюджет этой диагностики отделён от основного runs/budget.json: {INTERIM_BUDGET_PATH}")
    results = run_variants(variants_by_q)
    print_table(results)
    spent = sum(r["cost_usd"] for r in results)
    print(f"\nПотрачено по факту в этом прогоне: ${spent:.4f}")
    print(f"Интерим-счётчик: {json.loads(INTERIM_BUDGET_PATH.read_text()) if INTERIM_BUDGET_PATH.exists() else 'нет файла'}")


if __name__ == "__main__":
    main()
