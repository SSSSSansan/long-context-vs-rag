"""Разбор низкого quote_grounded (бесплатно, офлайн по runs/*/raw.jsonl; в API ничего не уходит).

Для ответов с correct=True и quote_grounded=False (старое правило) определяет причину:
  (а) нормализация — после нормализации кавычек/ё/тире/ссылок на пункт цитата дословно есть в контексте;
  (в) склейка    — каждый кусок цитаты дословен, но куски разделены только кавычками/«;»;
  (б) перефраз   — после всего этого часть цитаты всё равно не найдена в контексте (модель пересказала);
  (г) другое     — пустая цитата и т.п.
Пересчитывает метрику на логах до/после. Пишет results/quote_grounded_diagnosis.md.   python -m scripts.diagnose_quote_grounded
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import pandas as pd

from src.eval import aggregate as ag
from src.eval import exact_metrics as em

ROOT = Path(__file__).resolve().parent.parent


def legacy_norm(s: str) -> str:
    """Нормализация ДО правки (как в версии от 21 ноября): только «», " и ' удалялись."""
    return re.sub(r"\s+", " ", re.sub(r"[«»\"']", "", s.lower())).strip()


def legacy_grounded(quote: str, ctx: str) -> bool:
    parts = em._DOC_TAG.split(quote or "")
    segs = [p for p in (legacy_norm(p).strip(" .,;:|/") for p in parts) if p]
    return bool(segs) and all(s in legacy_norm(ctx) for s in segs)


def main():
    runs = sorted(glob.glob(str(ROOT / "runs/v2_s[1-7]"))) + [str(ROOT / "runs/v2_b_main")]
    df = ag.load([Path(d) for d in runs])
    df = df[df.model.str.startswith("gpt-6-luna") & df.method.ne("nocontext")].copy()
    df["ctx"] = [em.context_text_of_row(r) for _, r in df.iterrows()]
    df["old"] = [legacy_grounded(q, c) for q, c in zip(df.quote.fillna(""), df.ctx)]
    df["norm_only"] = [em.quote_grounded(q, c, split_quotes=False) for q, c in zip(df.quote.fillna(""), df.ctx)]
    df["new"] = [em.quote_grounded(q, c) for q, c in zip(df.quote.fillna(""), df.ctx)]

    def sentence_level(q: str, c: str) -> bool:
        """Только диагностика (в метрику НЕ принято): каждое ПРЕДЛОЖЕНИЕ цитаты дословно есть в контексте."""
        sents = [em._norm(x).strip(" .,;:|/()") for part in em._DOC_TAG.split(q or "")
                 for x in re.split(r"(?<=[.!?])\s+(?=[А-ЯA-Z«„\d])|[;()]", em._LOCATOR.sub("", part))]
        sents = [x for x in sents if len(x) > 3]
        return bool(sents) and all(x in em._norm(c) for x in sents)

    def cause(r):
        if not (r.correct == 1 and not r.old):
            return None
        if not (r.quote or "").strip():
            return "г: пустая цитата"
        if r.norm_only:
            return "а: нормализация"
        if r.new:
            return "в: склейка кусков"
        if sentence_level(r.quote, r.ctx):
            return "г: склейка предложений без разделителя (не принято в метрику)"
        return "б: перефраз/не найдено"
    df["cause"] = df.apply(cause, axis=1)
    bad = df[df.cause.notna()]
    lines = ["# Разбор quote_grounded (офлайн по сохранённым ответам gpt-6-luna; новых вызовов нет)\n",
             f"Строк: {len(df)} (Ш1–Ш7 + набор B, без NO-CONTEXT); верных ответов {int((df.correct == 1).sum())}; "
             f"верных с `quote_grounded=False` по старому правилу: {len(bad)}.\n",
             "## Причины (среди верных ответов с негодной цитатой по старому правилу)\n", "| Причина | Строк | Доля |", "|---|---|---|"]
    for c, n in bad.cause.value_counts().items():
        lines.append(f"| {c} | {n} | {n / len(bad):.0%} |")
    lines += ["\n## quote_grounded до / после (среднее по всем ответам метода)\n",
              "| Метод | старое | только нормализация | нормализация + склейка (новое) |", "|---|---|---|---|"]
    for m, g in df.groupby("method"):
        lines.append(f"| {m} | {g.old.mean():.2f} | {g.norm_only.mean():.2f} | {g.new.mean():.2f} |")
    lines.append(f"| **все** | {df.old.mean():.2f} | {df.norm_only.mean():.2f} | {df.new.mean():.2f} |")
    ok = df[df.correct == 1]
    lines.append(f"| среди верных | {ok.old.mean():.2f} | {ok.norm_only.mean():.2f} | {ok.new.mean():.2f} |")
    rest = bad[bad.cause.str.startswith("б")]
    lines += [f"\n## Примеры остатка «перефраз / не найдено» ({len(rest)} строк), первые 8\n"]
    for _, r in rest.head(8).iterrows():
        lines.append(f"- `{r.method}` {r.qid}: {str(r.quote)[:220]!r}")
    out = ROOT / "results/quote_grounded_diagnosis.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    # выборка ≥20 для ручного просмотра (разные методы)
    s = pd.concat([g.sample(min(len(g), 4), random_state=0) for _, g in bad.groupby("method")])
    print("\n--- выборка ---")
    for _, r in s.iterrows():
        print(f"{r.cause:24} {r.method:11} {r.qid:10} {str(r.quote)[:150]!r}")


if __name__ == "__main__":
    main()
