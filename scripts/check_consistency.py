"""Сверка ключевых чисел демо с источниками (results/*.csv, results/*.md, REPORT.md, README.md). Источник истины — results/.
Без сети и API. Запуск в окружении демо (нужен streamlit):   python scripts/check_consistency.py
Код возврата 1, если есть расхождения."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "demo/data"
R = ROOT / "results"
fails: list[str] = []
n_ok = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global n_ok
    if ok:
        n_ok += 1
    else:
        fails.append(f"{name}: {detail}")
        print(f"РАСХОЖДЕНИЕ  {name}: {detail}")


def close(a, b, tol=0.006) -> bool:
    return abs(float(a) - float(b)) <= tol


# 1. файлы демо = файлы results/ (демо их только копирует)
for f in ["mcnemar_v2_main.csv", "mcnemar_v2_strict.csv", "mcnemar_b_main.csv", "mcnemar_b_strict.csv", "mcnemar_v2_anchor_main.csv", "mcnemar_v2_anchor_strict.csv",
          "routing_sim.csv", "bonus_bge.csv", "bonus_bge_pairs.csv", "bonus_oprag.csv", "bonus_oprag_pairs.csv", "bonus_terra.csv", "bonus_terra_pairs.csv",
          "breakeven_gpt-6-luna.csv", "breakeven_gpt-6.1-terra.csv"]:
    check(f"копия {f}", (D / f).exists() and (R / f).exists() and (D / f).read_bytes() == (R / f).read_bytes(), "нет файла или содержимое отличается от results/")

# 2. расход
meta = json.loads((D / "meta.json").read_text(encoding="utf-8"))
bf = ROOT / "runs/budget_v2.json"
if bf.exists():
    spent = json.loads(bf.read_text())["spent_usd"]
    check("расход: meta.json = счётчику", close(meta["spent_usd"], spent, 1e-6), f"{meta['spent_usd']} против {spent}")
else:
    spent = meta["spent_usd"]
    print("(runs/budget_v2.json нет в этой копии: расход сверяется только с meta.json)")
txt = {n: (ROOT / n).read_text(encoding="utf-8") for n in ("README.md", "REPORT.md", "results/spend_summary.md")}
for n, t in txt.items():
    check(f"расход ${spent:.4f} в {n}", f"{spent:.4f}" in t, "числа нет в тексте")

# 3. демо: главный экран против results/main_table_v2.csv
mt = pd.read_csv(R / "main_table_v2.csv")
mt = mt[mt.variant == "main"]
style = json.loads((D / "style.json").read_text(encoding="utf-8"))["methods"]
by_label = {v["label"]: k for k, v in style.items()}
at = AppTest.from_file(str(ROOT / "demo/app.py"), default_timeout=120).run()
PAGES = list(at.sidebar.radio[0].options)


def goto(i: int):
    at.sidebar.radio[0].set_value(PAGES[i]).run()


def radio(label):
    return next(r for r in at.radio if r.label == label)


goto(0)
cards = " ".join(m.value for m in at.markdown)
for qs in ("A-hard", "A"):
    for size in ("8k", "30k", "100k"):
        radio("Набор вопросов").set_value(qs).run()
        radio("Размер корпуса").set_value(size).run()
        df = at.dataframe[0].value
        for _, r in df.iterrows():
            m = by_label.get(re.sub(r"^[^\w]+\s*", "", r["Метод"]))
            src = mt[(mt.question_set == qs) & (mt["size"] == size) & (mt.method == m)]
            if src.empty:
                check(f"таблица {qs}/{size}/{m}", False, "нет строки в main_table_v2.csv")
                continue
            s = src.iloc[0]
            check(f"accuracy {qs}/{size}/{m}", close(r["Accuracy"], s.accuracy, 0.0011), f"демо {r['Accuracy']}, results {s.accuracy:.3f}")
            check(f"evidence recall {qs}/{size}/{m}", close(r["Evidence recall"], s.evidence_recall, 0.011), f"демо {r['Evidence recall']}, results {s.evidence_recall:.3f}")
            check(f"citation {qs}/{size}/{m}", close(r["Citation"], s.citation, 0.011), f"демо {r['Citation']}, results {s.citation:.3f}")
            check(f"отказы {qs}/{size}/{m}", close(r["Отказы"], s.abstain, 0.011), f"демо {r['Отказы']}, results {s.abstain:.3f}")
            check(f"$ за верный {qs}/{size}/{m}", abs(float(r["$/верный"]) - s.usd_per_correct) <= 0.00001, f"демо {r['$/верный']}, results {s.usd_per_correct:.5f}")
radio("Набор вопросов").set_value("A-hard").run(); radio("Размер корпуса").set_value("100k").run()
cards = " ".join(m.value for m in at.markdown)
e = mt[(mt["size"] == "100k") & (mt.method == "emb")].set_index("question_set").accuracy
check("карточка EMB A → A-hard", f"{e['A']:.2f} → {e['A-hard']:.2f}" in cards, f"results: {e['A']:.2f} → {e['A-hard']:.2f}")
nc = mt[mt.method == "nocontext"]
check("карточка NO-CONTEXT: 0 утечек", nc.accuracy.max() == 0 and f"0 из {int(nc.n_calls.sum())}" in cards, f"results: accuracy {nc.accuracy.max()}, вызовов {int(nc.n_calls.sum())}")
m_ratio = re.search(r"×(\d+)</div>", cards)
rep_ratio = re.search(r"×(\d+) \(100K\)", txt["REPORT.md"])
check("карточка «LC дороже RAG» = REPORT", bool(m_ratio and rep_ratio and m_ratio.group(1) == rep_ratio.group(1)), f"демо ×{m_ratio and m_ratio.group(1)}, REPORT ×{rep_ratio and rep_ratio.group(1)}")

# 4. quote_grounded: среднее в демо = число в main_table_v2.md и README/REPORT
rows = pd.read_json(D / "rows.jsonl", lines=True)
qg = rows[rows.qset.isin(["A", "A-hard"]) & (rows.method != "nocontext")].quote_grounded.mean()
md = (R / "main_table_v2.md").read_text(encoding="utf-8")
mm = re.search(r"без NO-CONTEXT ([0-9.]+)", md)
check("quote_grounded: демо = main_table_v2.md", bool(mm) and close(qg, mm.group(1), 0.0051), f"демо {qg:.3f}, таблица {mm.group(1) if mm else '—'}")
check("quote_grounded ≈ 0.92 в README и REPORT", "0.92" in txt["README.md"] and "0.92" in txt["REPORT.md"] and close(qg, 0.92, 0.0051), f"демо {qg:.3f}")

# 5. якорь: таблица демо = results, p-value в тексте
goto(2)
an = pd.read_csv(R / "mcnemar_v2_anchor_main.csv")
shown = next(d.value for d in at.dataframe if "Пар" in d.value.columns)
check("якорь: число строк", len(shown) == len(an), f"демо {len(shown)}, results {len(an)}")
emb = an[(an.method == "emb") & (an["size"] == "100k")].iloc[0]
row = shown[(shown["Метод"] == "EMB") & (shown["Размер"] == "100K")].iloc[0]
check("якорь EMB 100K: p и p(Холм)", close(row["p"], emb.p_value, 5e-5) and close(row["p (Холм)"], emb.p_holm, 5e-5), f"демо {row['p']}/{row['p (Холм)']}, results {emb.p_value:.4f}/{emb.p_holm:.4f}")
check("якорь EMB 100K: p в REPORT", f"{emb.p_value:.4f}" in txt["REPORT.md"] and f"{emb.p_holm:.3f}" in txt["REPORT.md"], "в REPORT нет чисел из results")
check("якорь EMB 100K: только A верно / только A-hard", (emb.only_a, emb.only_b) == (9, 0), f"results {emb.only_a}/{emb.only_b}")

# 6. бонус-проверки: вкладки демо = results; ключевые числа REPORT = results
goto(6)
for name, claims in (("bge", {("100k", "EMB bge-m3 (локально)"): 0.59, ("100k", "EMB text-embedding-3-small"): 0.44, ("100k", "BM25"): 0.93}),
                     ("terra", {("100k", "Terra · BM25"): 0.94, ("100k", "Luna · EMB"): 0.44}),
                     ("oprag", {("100k", "BM25 k=16"): 1.00, ("100k", "BM25 k=8"): 0.93, ("100k", "HYB-rerank порядок документа k=8"): 0.91})):
    b = pd.read_csv(R / f"bonus_{name}.csv")
    for (size, cond), v in claims.items():
        r = b[(b["size"] == size) & (b.cond == cond)]
        check(f"бонус {name}: «{cond}» {size}", len(r) == 1 and close(r.accuracy.iloc[0], v, 0.0051), f"results {r.accuracy.iloc[0] if len(r) else '—'}, ожидалось {v}")
shown_tables = [d.value for d in at.dataframe]
for name in ("bge", "terra", "oprag"):
    b = pd.read_csv(R / f"bonus_{name}.csv")
    ok = any(len(t) == len(b) and "Условие" in t.columns and set(t["Условие"]) == set(b.cond) for t in shown_tables)
    check(f"бонус {name}: таблица в демо = results", ok, "строки условий не совпали")
sim = pd.read_csv(R / "routing_sim.csv")
check("Self-Route EMB A-hard 100K: 0.83", close(sim[(sim.qset == "A-hard") & (sim["size"] == "100k") & (sim.rag_method == "emb") & (sim.strategy == "self_route")].accuracy.iloc[0], 0.83, 0.0051), "")

# 7. калькулятор: ежемесячные суммы против REPORT (10 000 вопросов, 100K, Luna, кэш 0); судья ($0.00005 за вызов) в демо не входит
goto(3)
card_txt = " ".join(m.value for m in at.markdown)
vals = {}
for name in ("LC", "RAG (BM25)", "AGENT", "Self-Route"):
    m = re.search(rf"{re.escape(name)}: в месяц</div><div class='big'[^>]*>\$([\d ]+(?:\.\d+)?)</div>", card_txt)
    vals[name] = float(m.group(1).replace(" ", "")) if m else float("nan")
judge = 0.00005 * 10_000
for name, rep in (("LC", 127), ("RAG (BM25)", 4.5), ("AGENT", 8)):
    got = vals[name] + judge
    check(f"калькулятор {name}: $/мес. (+судья) против REPORT ≈ ${rep}", abs(got - rep) / rep <= 0.06, f"демо {vals[name]:.1f} + судья {judge:.1f} = {got:.1f}")

print(f"\nпроверено: {n_ok + len(fails)}; совпало: {n_ok}; расхождений: {len(fails)}")
sys.exit(1 if fails else 0)
