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
            return f"{acc:.2f} {ci_s}, {upc_s}, n={n}"
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
             "Ячейка: accuracy [95% CI Уилсона по уникальным вопросам] · $/correct (без кэша) · n вопросов.",
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


def main() -> None:
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
