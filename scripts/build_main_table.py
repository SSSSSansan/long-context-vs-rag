"""Шаг 4: главная таблица на одном экране — results/main_table.md + .csv.

Строки — методы (LC, BM25, EMB), столбцы — размеры корпуса (8K/30K/100K), в ячейке
accuracy [95% CI Уилсона], $/correct, n вопросов. Отдельно блок по T1/T2/T3.
Две отдельные таблицы — gpt-4o (1 повтор) и gpt-4o-mini (3 повтора), с явной пометкой модели.
Пропуски (нет данных в results/*.csv) — «—», не 0.

Источники: results/summary_overall.csv, results/summary_by_type.csv (уже пересчитаны
с доверительными интервалами в scripts шага 2). Без новых вызовов API — чистое чтение CSV.

Запуск: python -m scripts.build_main_table
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

MODELS = [
    ("gpt-4o-2024-08-06", "gpt-4o (1 повтор)"),
    ("gpt-4o-mini-2024-07-18", "gpt-4o-mini (3 повтора)"),
]
SIZES = ["8k", "30k", "100k"]
METHODS = [("lc", "LC"), ("bm25", "BM25"), ("emb", "EMB")]
TYPES = ["T1", "T2", "T3"]


def load_rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def cell(rows: list[dict], model: str, size: str, method: str, type_: str | None = None) -> str:
    for r in rows:
        if r["model"] == model and r["size"] == size and r["method"] == method and (type_ is None or r.get("type") == type_):
            acc = float(r["accuracy"])
            lo, hi = r["acc_ci_low"], r["acc_ci_high"]
            n = r["n_questions"]
            upc = r["usd_per_correct"]
            upc_s = "—" if upc in ("", None) else f"${float(upc):.4f}"
            ci_s = f"[{float(lo):.2f}–{float(hi):.2f}]" if lo not in ("", None) else "—"
            aj = r.get("accuracy_judge")
            judge_s = f" (судья {float(aj):.2f})" if aj not in ("", None) else ""
            return f"{acc:.2f}{judge_s} {ci_s}, {upc_s}, n={n}"
        if r["model"] == model and r["size"] == size and r["method"] == method and type_ is not None and r.get("type") != type_:
            continue
    return "—"


def table_block(rows: list[dict], model: str, type_: str | None = None) -> str:
    header = "| Метод | " + " | ".join(s.upper() for s in SIZES) + " |"
    sep = "|---" * (len(SIZES) + 1) + "|"
    lines = [header, sep]
    for m_key, m_label in METHODS:
        cells = [cell(rows, model, size, m_key, type_) for size in SIZES]
        lines.append(f"| {m_label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_markdown() -> str:
    overall = load_rows(RES / "summary_overall.csv")
    by_type = load_rows(RES / "summary_by_type.csv")

    parts = ["# Главная таблица: Long Context vs RAG",
             "",
             "Ячейка: accuracy = судья correct И value_in_answer (судья отдельно в скобках, если есть) [95% CI Уилсона по уникальным вопросам] · $/correct (без кэша) · n вопросов.",
             "Прочерк «—» значит данных нет (прогон не делался или бюджет не позволил), не 0.",
             ""]
    for model, model_label in MODELS:
        parts.append(f"## {model_label} — все типы вопросов")
        parts.append("")
        parts.append(table_block(overall, model))
        parts.append("")
        parts.append(f"### {model_label} — по типу вопроса (T1 простой факт / T2 сравнение / T3 противоречие)")
        parts.append("")
        for t in TYPES:
            parts.append(f"**{t}**")
            parts.append("")
            parts.append(table_block(by_type, model, t))
            parts.append("")
    return "\n".join(parts)


def build_csv() -> list[dict]:
    overall = load_rows(RES / "summary_overall.csv")
    by_type = load_rows(RES / "summary_by_type.csv")
    out = []
    for model, model_label in MODELS:
        for m_key, m_label in METHODS:
            for size in SIZES:
                row = {"model": model_label, "method": m_label, "size": size, "type": "ALL"}
                row["cell"] = cell(overall, model, size, m_key, None)
                out.append(row)
                for t in TYPES:
                    row_t = {"model": model_label, "method": m_label, "size": size, "type": t}
                    row_t["cell"] = cell(by_type, model, size, m_key, t)
                    out.append(row_t)
    return out


# ---------------------------------------------------------------------------------------------------------------
# V2 (gpt-6-luna, runs/v2_s1..s7): два варианта итоговой метрики + судья.   python -m scripts.build_main_table --v2
# (а) основная: судья correct И value_in_answer (T2: число победителя);  (б) строгая: T2 требует оба числа (диагностика).
# ---------------------------------------------------------------------------------------------------------------
V2_METHODS = [("nocontext", "NO-CONTEXT"), ("lc", "LC"), ("bm25", "BM25"), ("emb", "EMB"), ("hyb_rrf", "HYB-RRF"),
              ("hyb_rerank", "HYB-rerank"), ("agent_bm25", "AGENT(bm25)"), ("agent_emb", "AGENT(emb)"), ("agent_hyb", "AGENT(hyb)")]
V2_SETS = [("A", "Набор A (с лексическим якорем)"), ("A-hard", "Набор A-hard (без якоря)")]


def build_v2() -> None:
    import glob
    import pandas as pd
    from src.eval import aggregate as ag

    dirs = [Path(d) for d in sorted(glob.glob(str(ROOT / "runs/v2_s[1-7]")))]
    df = ag.load(dirs)
    df = df[df.model.str.startswith("gpt-6-luna")].copy()
    strict = df.copy()
    strict["correct"] = strict["correct_strict"]
    by = ["question_set", "size", "method"]
    main_s, strict_s = ag.summarize(df, by), ag.summarize(strict, by)

    def cell_v2(sm: pd.DataFrame, qs: str, size: str, m: str, show: str) -> str:
        r = sm[(sm.question_set == qs) & (sm["size"] == size) & (sm.method == m)]
        if r.empty:
            return "—"
        r = r.iloc[0]
        return f"{r.accuracy:.2f} [{r.acc_ci_low:.2f}–{r.acc_ci_high:.2f}]" if show == "ci" else f"{r[show]:.2f}"

    def block(qs: str, sub: str) -> list[str]:
        out = [f"### {sub}", "", "Ячейка: основная метрика [CI Уилсона] / строгая / судья. «—» = не прогонялось. n = 18 вопросов × 3 повтора = 54 вызова.", "",
               "| Метод | " + " | ".join(s.upper() for s in SIZES) + " |", "|---" * (len(SIZES) + 1) + "|"]
        for m, label in V2_METHODS:
            cells = []
            for size in SIZES:
                a = cell_v2(main_s, qs, size, m, "ci")
                cells.append("—" if a == "—" else f"{a} / {cell_v2(strict_s, qs, size, m, 'accuracy')} / {cell_v2(main_s, qs, size, m, 'accuracy_judge')}")
            out.append(f"| {label} | " + " | ".join(cells) + " |")
        return out + [""]

    parts = ["# Главная таблица V2: gpt-6-luna", "",
             "Метрики: **основная** = судья `correct` И `value_in_answer` (T2: число победителя, правило winner-value-only, выбрано постфактум — см. REPORT.md); "
             "**строгая** = то же, но T2 требует оба числа (диагностика); **судья** = accuracy по вердикту судьи (отказ = 0). "
             "Числа предварительные до независимой калибровки. Без кэша и судьи в $.", ""]
    for qs, sub in V2_SETS:
        parts += block(qs, sub)
    # T2 отдельно: именно здесь правило меняет цифры
    t2 = df[df.type == "T2"]
    t2s = t2.copy(); t2s["correct"] = t2s["correct_strict"]
    sm_m, sm_s = ag.summarize(t2, by), ag.summarize(t2s, by)
    parts += ["### Только T2 («что из двух»), n = 6 вопросов × 3 повтора на ячейку: основная / строгая / судья", "",
              "| Набор | Метод | " + " | ".join(s.upper() for s in SIZES) + " |", "|---" * (len(SIZES) + 2) + "|"]
    for qs, _ in V2_SETS:
        for m, label in V2_METHODS:
            cs = [f"{cell_v2(sm_m, qs, z, m, 'accuracy')} / {cell_v2(sm_s, qs, z, m, 'accuracy')} / {cell_v2(sm_m, qs, z, m, 'accuracy_judge')}"
                  if cell_v2(sm_m, qs, z, m, "accuracy") != "—" else "—" for z in SIZES]
            if any(c != "—" for c in cs):
                parts.append(f"| {qs} | {label} | " + " | ".join(cs) + " |")
    parts.append("")
    # $/correct (основная метрика)
    parts += ["### $/correct (основная метрика; без кэша, без судьи)", "", "| Набор | Метод | " + " | ".join(s.upper() for s in SIZES) + " |",
              "|---" * (len(SIZES) + 2) + "|"]
    for qs, _ in V2_SETS:
        for m, label in V2_METHODS:
            cs = []
            for z in SIZES:
                r = main_s[(main_s.question_set == qs) & (main_s["size"] == z) & (main_s.method == m)]
                cs.append("—" if r.empty or pd.isna(r.iloc[0].usd_per_correct) else f"${r.iloc[0].usd_per_correct:.5f}")
            if any(c != "—" for c in cs):
                parts.append(f"| {qs} | {label} | " + " | ".join(cs) + " |")
    (RES / "main_table_v2.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    main_s.assign(variant="main").pipe(lambda a: pd.concat([a, strict_s.assign(variant="strict")])).to_csv(RES / "main_table_v2.csv", index=False)
    print(f"записано: {RES / 'main_table_v2.md'}, {RES / 'main_table_v2.csv'}")


def main() -> None:
    import sys
    if "--v2" in sys.argv:
        return build_v2()
    md = build_markdown()
    (RES / "main_table.md").write_text(md, encoding="utf-8")
    rows = build_csv()
    with (RES / "main_table.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "method", "size", "type", "cell"])
        w.writeheader()
        w.writerows(rows)
    print(f"записано: {RES / 'main_table.md'}, {RES / 'main_table.csv'}")


if __name__ == "__main__":
    main()
