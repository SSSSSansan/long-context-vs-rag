"""Агрегация: results/summary_*.csv и графики. mean ± std считается по повторам (std между повторами, не между вопросами).

  python -m src.eval.aggregate [runs/dir1 runs/dir2 ...]     # по умолчанию — все runs/*/raw.jsonl
  RESULTS_DIR=results/local python -m src.eval.aggregate runs/local_s1   # писать таблицы в другую папку
"""
from __future__ import annotations

import json
import math
import os
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..config import ROOT, base_cfg  # noqa: E402
from .exact_metrics import context_text_of_row, quote_grounded as _quote_grounded  # noqa: E402
from .value_check import value_in_answer, value_in_answer_strict  # noqa: E402

RES = Path(os.environ["RESULTS_DIR"]).resolve() if os.environ.get("RESULTS_DIR") else ROOT / "results"   # RESULTS_DIR=results/local — отладка, не затирает V1-таблицы

Z95 = 1.959963984540054  # квантиль N(0,1) для 95% доверительного интервала


def wilson_ci(successes: float, n: float, z: float = Z95) -> tuple[float, float]:
    """Доверительный интервал Уилсона для доли successes/n.

    n здесь — число НЕЗАВИСИМЫХ наблюдений. Для наших данных это число уникальных
    вопросов в ячейке (model x size x method x type), а не число вызовов API: три
    повтора одного и того же вопроса — это не три независимых измерения (тот же
    вопрос, тот же контекст, почти тот же ответ при temperature=0), они только
    показывают разброс ОДНОГО вопроса. Если бы n считали по вызовам, интервал был
    бы искусственно узким — казался бы увереннее, чем позволяют данные.
    `successes` — сумма средних по повторам accuracy для каждого уникального
    вопроса (может быть нецелой), это несмещённая оценка числа "успехов".
    """
    if not n or n <= 0:
        return (np.nan, np.nan)
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _qset(r) -> str:
    """Метка набора вопросов для группировки: "A" | "A-hard" | "B" | "V1" (данные без наборов).
    Группировать надо по qset, а не по set: у A и A-hard одинаковый set="A", различается якорь."""
    if isinstance(r["question_set"], str):
        return r["question_set"]
    if isinstance(r["set"], str):
        hard = pd.notna(r["anchor"]) and not bool(r["anchor"])
        return f"{r['set']}-hard" if hard else r["set"]
    return "V1"


def load(dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for d in dirs:
        p = d / "raw.jsonl"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                r["run"] = d.name
                rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # quote_grounded пересчитывается ТЕКУЩИМ правилом (нормализация кавычек/ссылок на пункт, см. exact_metrics.py) по сохранённому контексту:
    # новых вызовов API не нужно; значение, записанное при прогоне, остаётся в колонке quote_grounded_logged
    if "prompt" in df:
        df["quote_grounded_logged"] = df.get("quote_grounded")
        df["quote_grounded"] = [float(_quote_grounded(q or "", context_text_of_row(r)))
                                for q, (_, r) in zip(df.quote.fillna(""), df.iterrows())]
    # итоговая accuracy = судья == correct И value_in_answer (src/eval/value_check.py); «по судье» оставлена отдельно
    ab = df.abstained.fillna(False).astype(bool) if "abstained" in df else False
    df["correct_judge"] = ((df.verdict == "correct") & ~ab).astype(float)   # отказ -> 0 всегда, даже если судья ошибся
    df["value_ok"] = [value_in_answer(g, a, q) for g, a, q in zip(df.gold_answer.fillna(""), df.answer.fillna(""), df.quote.fillna(""))]
    df["correct"] = df.correct_judge * df.value_ok.astype(float)
    # диагностика: строгое правило «оба числа» для T2 (основное — winner-value-only, см. value_check.py)
    df["value_ok_strict"] = [value_in_answer_strict(g, a, q) for g, a, q in zip(df.gold_answer.fillna(""), df.answer.fillna(""), df.quote.fillna(""))]
    df["correct_strict"] = df.correct_judge * df.value_ok_strict.astype(float)
    for c in ("set", "anchor", "question_set"):      # у данных V1 этих полей нет
        if c not in df:
            df[c] = None
    df["qset"] = df.apply(_qset, axis=1)
    df["lenient"] = df.verdict.isin(["correct", "partial"]).astype(float)
    df["system"] = df.model.str.replace(r"-\d{4}-\d{2}-\d{2}$", "", regex=True) + ":" + df.method + df.k_label.map(lambda k: "" if k is None or pd.isna(k) else f"@k={k}")
    mp = base_cfg()["model_prices"]
    pin = df.model.map(lambda m: mp.get(m, {"input": 0.0})["input"])
    pw = df.model.map(lambda m: mp.get(m, {"input": 0.0}).get("cache_write_input", mp.get(m, {"input": 0.0})["input"]))
    pout = df.model.map(lambda m: mp.get(m, {"output": 0.0})["output"])
    # «холодная» цена вызова (кэш не сработал): весь промпт >=1024 токенов записывается в кэш по cache_write_input (1.25x input)
    pin = pin.where(df.prompt_tokens < 1024, pw)
    df["cost_nocache_usd"] = (df.prompt_tokens * pin + df.completion_tokens * pout) / 1e6
    return df


def summarize(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    # сначала среднее по повторам внутри каждого повтора -> потом mean ± std по повторам
    per_rep = df.groupby(by + ["repeat"]).agg(
        accuracy=("correct", "mean"), accuracy_judge=("correct_judge", "mean"), value_ok=("value_ok", "mean"), lenient=("lenient", "mean"), evidence_recall=("evidence_recall", "mean"),
        citation=("citation_correct", "mean"), quote_grounded=("quote_grounded", "mean"),
        abstain=("abstained", "mean")).reset_index()
    g = per_rep.groupby(by)
    out = g.mean(numeric_only=True).drop(columns="repeat")
    out["accuracy_std"] = g.accuracy.std(ddof=0)
    tot = df.groupby(by).agg(
        n_calls=("qid", "size"), n_questions=("qid", "nunique"), repeats=("repeat", "nunique"),
        prompt_tokens=("prompt_tokens", "mean"), cached_share=("cached_tokens", lambda s: s.sum() / max(1, df.loc[s.index, "prompt_tokens"].sum())),
        lat_p50_ms=("generation_ms", lambda s: s.quantile(0.5)), lat_p95_ms=("generation_ms", lambda s: s.quantile(0.95)),
        retr_p50_ms=("retrieval_ms", lambda s: s.quantile(0.5)),
        cost_usd=("cost_usd", "sum"), cost_nocache_usd=("cost_nocache_usd", "sum"), n_correct=("correct", "sum"))
    out = out.join(tot)
    agent_cols = [c for c in ("n_tool_calls", "tool_call_accuracy", "agent_give_up", "search_recall") if c in df and df[c].notna().any()]
    if agent_cols:       # метрики AGENT: среднее по эпизодам (пустые значения у не-агентных методов пропускаются)
        num = df[agent_cols].apply(lambda col: col.map(lambda v: float(v) if v is not None and v == v else np.nan))
        out = out.join(num.groupby([df[b] for b in by]).mean())
    # $/correct: стоимость ответов / число верных (стоимость judge не входит; цена из первого, не из кэш-хитов)
    out["usd_per_correct"] = out.cost_nocache_usd / out.n_correct.replace(0, np.nan)
    out["usd_per_correct_cached"] = out.cost_usd / out.n_correct.replace(0, np.nan)

    # Доверительные интервалы Уилсона 95% — по уникальным вопросам, не по вызовам (см. wilson_ci).
    # Сначала схлопываем повторы одного вопроса в одно число (среднее по repeat для каждого qid),
    # затем n = число уникальных вопросов в ячейке, successes = сумма этих средних по вопросам.
    per_q = df.groupby(by + ["qid"]).agg(
        acc_q=("correct", "mean"), evr_q=("evidence_recall", "mean"), cit_q=("citation_correct", "mean")
    ).reset_index()
    ci = per_q.groupby(by).agg(
        n_q=("qid", "nunique"), acc_k=("acc_q", "sum"), evr_k=("evr_q", "sum"), cit_k=("cit_q", "sum"))
    for col, k_col in [("acc", "acc_k"), ("evidence_recall", "evr_k"), ("citation", "cit_k")]:
        lo_hi = ci.apply(lambda row, kc=k_col: wilson_ci(row[kc], row["n_q"]), axis=1)
        ci[f"{col}_ci_low"] = lo_hi.map(lambda t: t[0])
        ci[f"{col}_ci_high"] = lo_hi.map(lambda t: t[1])
    out = out.join(ci[[c for c in ci.columns if c.endswith("_ci_low") or c.endswith("_ci_high")]])
    return out.reset_index()


MODEL_ORDER = ["gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18"]
SIZE_ORDER = ["8k", "30k", "100k"]
METHOD_ORDER = ["lc", "bm25", "emb"]
METHOD_LABEL = {"lc": "LC", "bm25": "BM25", "emb": "EMB"}
METHOD_COLOR = {"lc": "#4C72B0", "bm25": "#DD8452", "emb": "#55A868"}
ZERO_MARK_H = 0.02   # высота видимого столбика-маркера для accuracy == 0 (иначе bar высотой 0 не виден)


def _models_present(df: pd.DataFrame) -> list[str]:
    have = set(df["model"].unique())
    return [m for m in MODEL_ORDER if m in have] + sorted(have - set(MODEL_ORDER))


def _bar_panel(ax, sub: pd.DataFrame, value_col: str, sizes: list[str], methods: list[str],
               log: bool = False, no_data_y: float | None = None) -> None:
    """Сгруппированные столбцы method x size с явной отметкой нулей и отсутствующих данных.

    На линейной шкале accuracy=0 рисуется видимым коротким столбиком с подписью «0.00»
    (столбик высотой 0 иначе не виден и неотличим от «нет данных»). Если ячейки method x size
    вообще нет в данных ИЛИ значение NaN (например $/correct при n_correct=0) — подпись
    «нет данных» вместо столбика.
    """
    n_m = len(methods)
    width = 0.8 / n_m
    for mi, method in enumerate(methods):
        for si, size in enumerate(sizes):
            xpos = si + (mi - (n_m - 1) / 2) * width
            row = sub[(sub["size"] == size) & (sub["method"] == method)]
            val = row[value_col].iloc[0] if not row.empty else np.nan
            if row.empty or pd.isna(val):
                y = no_data_y if no_data_y is not None else (0.03 if not log else ax.get_ylim()[0])
                ax.text(xpos, y, "нет данных", ha="center", va="bottom", fontsize=6.5,
                        rotation=90, color="#888888")
                continue
            if not log and val == 0:
                ax.bar(xpos, ZERO_MARK_H, width=width * 0.9, color=METHOD_COLOR[method])
                ax.text(xpos, ZERO_MARK_H + 0.015, "0.00", ha="center", va="bottom", fontsize=7)
            else:
                ax.bar(xpos, val, width=width * 0.9, color=METHOD_COLOR[method])
    ax.set_xlim(-0.5, len(sizes) - 0.5)   # фиксируем ширину панели независимо от того, что отрисовано
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([s.upper() for s in sizes])


def _method_legend(fig) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, color=METHOD_COLOR[m]) for m in METHOD_ORDER]
    fig.legend(handles, [METHOD_LABEL[m] for m in METHOD_ORDER], loc="upper center",
               ncol=len(METHOD_ORDER), bbox_to_anchor=(0.5, 1.02), frameon=False)


def charts(df: pd.DataFrame):
    (RES / "charts").mkdir(parents=True, exist_ok=True)
    main_ = df[(df.get("target_pct").isna() if "target_pct" in df else True) & df.k_label.isna() | df.method.ne("lc") & df.k_label.eq(8)]
    main_ = main_[main_.get("target_pct", pd.Series(index=main_.index, dtype=float)).isna()] if "target_pct" in main_ else main_
    if main_.empty:
        return
    models = _models_present(main_)

    # Accuracy по размеру корпуса и типу вопроса: строка на модель, столбец на тип вопроса.
    s = summarize(main_, ["model", "size", "method", "type"])
    sizes = [x for x in SIZE_ORDER if x in set(s["size"])]
    fig, ax = plt.subplots(len(models), 3, figsize=(13, 3.8 * len(models)), sharey=True, squeeze=False)
    for ri, model in enumerate(models):
        for ci, t in enumerate(["T1", "T2", "T3"]):
            a = ax[ri][ci]
            sub = s[(s.model == model) & (s.type == t)]
            _bar_panel(a, sub, "accuracy", sizes, METHOD_ORDER)
            a.set_title(f"{model} — {t}", fontsize=10)
            a.set_ylim(0, 1.08)
        ax[ri][0].set_ylabel("accuracy")
    fig.suptitle("Accuracy по размеру корпуса, типу вопроса и модели", y=1.08 if len(models) == 1 else 1.04)
    _method_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.92 if len(models) == 1 else 0.95))
    fig.savefig(RES / "charts/accuracy_by_type.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # $/correct по размеру корпуса: панель на модель.
    s2 = summarize(main_, ["model", "size", "method"])
    fig, ax2 = plt.subplots(1, len(models), figsize=(6.5 * len(models), 3.8), squeeze=False)
    for ci, model in enumerate(models):
        a = ax2[0][ci]
        sub = s2[s2.model == model]
        a.set_yscale("log")
        _bar_panel(a, sub, "usd_per_correct", sizes, METHOD_ORDER, log=True, no_data_y=sub["usd_per_correct"].min(skipna=True) or 1e-4)
        a.set_title(f"{model}", fontsize=10)
        a.set_ylabel("$ на один верный ответ (лог.)" if ci == 0 else "")
    fig.suptitle("$/correct по размеру корпуса и модели", y=1.12)
    _method_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig(RES / "charts/usd_per_correct.png", dpi=140, bbox_inches="tight"); plt.close(fig)


def position_chart(df: pd.DataFrame):
    p = df[df.get("target_pct", pd.Series(index=df.index, dtype=float)).notna()] if "target_pct" in df else df.iloc[0:0]
    if p.empty:
        return
    s = summarize(p, ["model", "size", "target_pct"])
    s.to_csv(RES / "summary_position.csv", index=False)

    models = _models_present(s)
    offsets = {"30k": -2.0, "100k": 2.0}         # сдвиг по x (в п.п.), чтобы точки 30K/100K не сливались
    markers = {"30k": "o", "100k": "s"}
    for model in models:
        sm = s[s.model == model]
        if sm.empty:
            continue
        fig, a = plt.subplots(figsize=(7.5, 4.3))
        for size, g in sm.groupby("size"):
            g = g.sort_values("target_pct")
            x = g.target_pct + offsets.get(size, 0.0)
            yerr = np.vstack([
                (g.accuracy - g.acc_ci_low).clip(lower=0).to_numpy(),
                (g.acc_ci_high - g.accuracy).clip(lower=0).to_numpy(),
            ])
            a.errorbar(x, g.accuracy, yerr=yerr, marker=markers.get(size, "o"), linestyle="-",
                       label=f"{size.upper()} (95% CI Уилсона)", capsize=3)
            # n вопросов меняется от точки к точке (не один факт на каждой позиции) —
            # поэтому подписываем n у каждой точки отдельно, а не одним числом в легенде.
            for xi, yi, ni in zip(x, g.accuracy, g.n_questions):
                a.annotate(f"n={int(ni)}", (xi, yi), textcoords="offset points", xytext=(0, 8),
                           ha="center", fontsize=7, color="#555555")
        a.set_xlabel("позиция факта в корпусе, % (точки 30K/100K разнесены по x для читаемости)")
        a.set_ylabel("accuracy (LC)"); a.set_ylim(-0.05, 1.15); a.legend(fontsize=8, loc="lower center")
        a.set_title(f"Lost in the Middle: LC, {model}"); fig.tight_layout()
        suffix = "" if model == models[0] else f"_{model.split('-')[0]}{'mini' if 'mini' in model else ''}"
        fig.savefig(RES / f"charts/position_curve{suffix}.png", dpi=140, bbox_inches="tight"); plt.close(fig)


METHOD_PAIR_ORDER = ["nocontext", "lc", "bm25", "emb", "hyb_rrf", "hyb_rerank", "agent_bm25", "agent_emb", "agent_hyb"]
MCNEMAR_QSETS = ("A", "A-hard")   # парный дизайн есть только там: один и тот же вопрос во всех условиях


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Точный двусторонний тест Макнемара (биномиальный, p=0.5 на несогласных парах).

    only_a — вопросов, где верно только условие A; only_b — только условие B.
    Совпавшие пары (оба верно / оба неверно) на результат не влияют. При n=18 вопросах
    точный тест надёжнее хи-квадрат приближения. Если несогласных пар нет, p=1.0.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def holm_adjust(pvals: list[float]) -> list[float]:
    """Поправка Холма на множественные сравнения (возвращает скорректированные p в исходном порядке)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def question_outcomes(df: pd.DataFrame) -> pd.Series:
    """Один бинарный исход на вопрос: верно, если верны БОЛЬШЕ половины повторов (mean(correct) > 0.5).
    Повторы одного вопроса — не независимые наблюдения, поэтому в тест идёт один исход на вопрос."""
    return df.groupby(["model", "qset", "size", "method", "qid"]).correct.mean().gt(0.5)


def _pair_row(family: str, model: str, qset: str, context: str, a: str, b: str, wide: pd.DataFrame) -> dict | None:
    pair = wide[[a, b]].dropna().astype(bool)
    if pair.empty:
        return None
    xa, xb = pair[a], pair[b]
    only_a, only_b = int((xa & ~xb).sum()), int((~xa & xb).sum())
    return {"family": family, "model": model, "qset": qset, "context": context, "a": a, "b": b,
            "n_pairs": len(pair), "both_correct": int((xa & xb).sum()), "only_a": only_a, "only_b": only_b,
            "both_wrong": int((~xa & ~xb).sum()), "acc_a": float(xa.mean()), "acc_b": float(xb.mean()),
            "p_value": mcnemar_exact(only_a, only_b)}


def mcnemar_tables(df: pd.DataFrame, qsets: tuple[str, ...] = MCNEMAR_QSETS) -> pd.DataFrame:
    """Парные тесты Макнемара внутри наборов A и A-hard.

    family="method": пары методов на каждом размере (context = размер).
    family="size":   пары размеров для каждого метода (context = метод); сравниваются только вопросы,
                     которые есть на обоих размерах.
    p_holm — поправка Холма внутри одной таблицы пар (family, model, qset, context).
    """
    d = df[df.qset.isin(qsets)]
    if d.empty:
        return pd.DataFrame()
    out = question_outcomes(d)
    rows: list[dict] = []
    for (model, qset), g in out.groupby(level=["model", "qset"]):
        g = g.droplevel(["model", "qset"])
        for size in [x for x in SIZE_ORDER if x in set(g.index.get_level_values("size"))]:
            wide = g.xs(size, level="size").unstack("method")
            methods = sorted(wide.columns, key=lambda m: METHOD_PAIR_ORDER.index(m) if m in METHOD_PAIR_ORDER else 99)
            for a, b in combinations(methods, 2):
                r = _pair_row("method", model, qset, size, a, b, wide)
                if r:
                    rows.append(r)
        for method in sorted(set(g.index.get_level_values("method"))):
            wide = g.xs(method, level="method").unstack("size")
            sizes = [x for x in SIZE_ORDER if x in wide.columns]
            for a, b in combinations(sizes, 2):
                r = _pair_row("size", model, qset, method, a, b, wide)
                if r:
                    rows.append(r)
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["p_holm"] = np.nan
    for _, idx in res.groupby(["family", "model", "qset", "context"]).groups.items():
        res.loc[idx, "p_holm"] = holm_adjust(res.loc[idx, "p_value"].tolist())
    return res


def default_pair_key() -> dict[str, str]:
    """qid -> ключ пары «вопрос с якорем / тот же вопрос без якоря»: множество fact_ids (у A и A-hard оно совпадает)."""
    qs = json.loads((ROOT / "data/questions.json").read_text(encoding="utf-8"))["questions"]
    return {q["qid"]: "|".join(sorted(q["fact_ids"])) for q in qs}


def anchor_tables(df: pd.DataFrame, pair_key: dict[str, str] | None = None,
                  methods_skip: tuple[str, ...] = ("nocontext",)) -> pd.DataFrame:
    """Семейство «anchor»: парный McNemar exact A (с якорем) против A-hard (без якоря) для каждого метода и размера.

    Пара = один и тот же факт (совпадающие fact_ids) в двух формулировках вопроса. a = вариант A, b = вариант A-hard:
    only_a — верно только с якорем, only_b — верно только без якоря. Холм — по всей семье модель × (метод × размер),
    то есть строже, чем отдельно на каждый размер. NO-CONTEXT исключён (там 0 и в A, и в A-hard: проверять нечего).
    Методы без пары (агенты есть только в A-hard) пропускаются."""
    d = df[df.qset.isin(["A", "A-hard"]) & ~df.method.isin(methods_skip)]
    if d.empty:
        return pd.DataFrame()
    key = pair_key if pair_key is not None else default_pair_key()
    out = question_outcomes(d).rename("ok").reset_index()
    out["pair"] = out.qid.map(lambda q: key.get(q, q.rstrip("h") if q.endswith("h") else q))
    rows: list[dict] = []
    for (model, size, method), g in out.groupby(["model", "size", "method"]):
        wide = g.pivot(index="pair", columns="qset", values="ok")
        if not {"A", "A-hard"} <= set(wide.columns):
            continue
        pair = wide[["A", "A-hard"]].dropna().astype(bool)
        if pair.empty:
            continue
        xa, xh = pair["A"], pair["A-hard"]
        oa, ob = int((xa & ~xh).sum()), int((~xa & xh).sum())
        rows.append({"family": "anchor", "model": model, "size": size, "method": method, "n_pairs": len(pair),
                     "both_correct": int((xa & xh).sum()), "only_a": oa, "only_b": ob, "both_wrong": int((~xa & ~xh).sum()),
                     "acc_a": float(xa.mean()), "acc_b": float(xh.mean()), "p_value": mcnemar_exact(oa, ob)})
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["p_holm"] = np.nan
    for _, idx in res.groupby("model").groups.items():
        res.loc[idx, "p_holm"] = holm_adjust(res.loc[idx, "p_value"].tolist())
    res["size"] = pd.Categorical(res["size"], SIZE_ORDER, ordered=True)
    res["method"] = pd.Categorical(res["method"], METHOD_PAIR_ORDER, ordered=True)
    return res.sort_values(["model", "method", "size"]).reset_index(drop=True).astype({"size": str, "method": str})


def main():
    dirs = [Path(a) for a in sys.argv[1:]] or sorted((ROOT / "runs").glob("*/"))
    df = load([d if d.is_absolute() else ROOT / d for d in dirs])
    if df.empty:
        raise SystemExit("нет данных в runs/*/raw.jsonl")
    RES.mkdir(exist_ok=True)
    has_pos = "target_pct" in df
    p2 = df[df.gold_rank.notna()] if "gold_rank" in df else df.iloc[0:0]            # P2: ранг gold-чанка в top-10
    df = df.drop(p2.index)
    base = df[df.target_pct.isna()] if has_pos else df
    main_ = base[base.k_label.isna() | base.k_label.eq(base_cfg()["retrieval"]["top_k"])]
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    for name, by, d in [("summary_by_type", ["model", "qset", "size", "method", "type"], main_),
                        ("summary_overall", ["model", "qset", "size", "method"], main_),
                        ("summary_ablation_k", ["model", "qset", "size", "system", "type"], base[~base.method.isin(["lc", "nocontext"]) & ~base.method.str.startswith("agent_")])]:
        if d.empty:
            continue
        s = summarize(d, by)
        s.to_csv(RES / f"{name}.csv", index=False)
        print(f"\n== {name} ==\n", s.round(3).to_string(index=False))
    mc = mcnemar_tables(main_)
    if not mc.empty:
        mc.to_csv(RES / "mcnemar.csv", index=False)
        sig = mc[mc.p_value < 0.05]
        print(f"\n== mcnemar (парные тесты A / A-hard; значимых при p<0.05: {len(sig)} из {len(mc)}; полная таблица results/mcnemar.csv) ==\n",
              sig.round(4).to_string(index=False) if len(sig) else "значимых различий нет")
    if not p2.empty:
        s2 = summarize(p2, ["model", "qset", "size", "gold_rank"])
        s2.to_csv(RES / "summary_p2.csv", index=False)
        print("\n== summary_p2 (accuracy по рангу gold-чанка, k=10) ==\n", s2.round(3).to_string(index=False))
    if has_pos:
        position_chart(df)
    base_ = df[df.target_pct.isna()] if has_pos else df
    charts_df = base_[base_.qset.isin(["V1", "A"])]   # столбчатые графики V1-вида: A-hard и B не смешиваем с A
    try:
        charts_df = charts_df.assign(k_label=charts_df.k_label.where(~charts_df.method.isin(["lc", "nocontext"]), np.nan))
        charts(charts_df)  # обе модели — отдельные панели/подписи внутри charts()
    except Exception as e:  # графики не должны ломать таблицы
        print("charts:", e)


if __name__ == "__main__":
    main()
