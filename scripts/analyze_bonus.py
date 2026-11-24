"""Анализ бонус-проверок (постфактум, 23 ноября 2026). Без API.   python -m scripts.analyze_bonus {bge|oprag|terra|all}

Сравнивает условия парным McNemar exact (один бинарный исход на вопрос: верно на большинстве повторов; Терра — 1 повтор) с поправкой Холма
внутри эксперимента. Пишет results/bonus_<имя>.csv и results/bonus_<имя>.md. Отдельно от главной таблицы.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

from src.eval import aggregate as ag

ROOT = Path(__file__).resolve().parent.parent


def base_rows() -> pd.DataFrame:
    df = ag.load([Path(d) for d in sorted(glob.glob(str(ROOT / "runs/v2_s[1-7]")))] + [ROOT / "runs/v2_emb_k16_check"])
    return df[df.model.str.startswith("gpt-6-luna") & df.qset.eq("A-hard")]


def cond_df(df, label_fn) -> pd.DataFrame:
    d = df.copy()
    d["cond"] = [label_fn(r) for _, r in d.iterrows()]
    return d[d.cond.notna()]


def per_cond_table(d: pd.DataFrame, by=("size", "cond")) -> pd.DataFrame:
    s = ag.summarize(d.assign(k_label=0), list(by))
    return s


def mcnemar_pairs(d: pd.DataFrame, pairs: list[tuple[str, str]], by_size=True, outcome_repeat=None) -> pd.DataFrame:
    """pairs: [(условие_a, условие_b)]; пары вопросов внутри размера. Холм по всем парам таблицы."""
    rows = []
    for size, g in d.groupby("size"):
        if outcome_repeat is not None:
            g = g[g.repeat == outcome_repeat]
        out = g.groupby(["cond", "qid"]).correct.mean().gt(0.5).unstack("cond")
        for a, b in pairs:
            if a in out and b in out:
                pr = out[[a, b]].dropna().astype(bool)
                if pr.empty:
                    continue
                oa, ob = int((pr[a] & ~pr[b]).sum()), int((~pr[a] & pr[b]).sum())
                rows.append({"size": size, "a": a, "b": b, "n_pairs": len(pr), "acc_a": pr[a].mean(), "acc_b": pr[b].mean(),
                             "only_a": oa, "only_b": ob, "p_value": ag.mcnemar_exact(oa, ob)})
    res = pd.DataFrame(rows)
    if not res.empty:
        res["p_holm"] = ag.holm_adjust(res.p_value.tolist())
    return res


def md_table(s: pd.DataFrame) -> list[str]:
    out = ["| Размер | Условие | accuracy [CI] | evidence recall | citation | отказы | $/correct | вызовов |", "|---|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        out.append(f"| {str(r['size']).upper()} | {r.cond} | {r.accuracy:.2f} [{r.acc_ci_low:.2f}–{r.acc_ci_high:.2f}] | {r.evidence_recall:.2f} | {r.citation:.2f} | "
                   f"{r.abstain:.2f} | ${r.usd_per_correct:.5f} | {int(r.n_calls)} |")
    return out


def md_pairs(p: pd.DataFrame) -> list[str]:
    out = ["| Размер | A | B | пар | acc A | acc B | только A верно | только B верно | p | p_Холм |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in p.iterrows():
        out.append(f"| {str(r['size']).upper()} | {r.a} | {r.b} | {int(r.n_pairs)} | {r.acc_a:.2f} | {r.acc_b:.2f} | {int(r.only_a)} | {int(r.only_b)} | {r.p_value:.4f} | {r.p_holm:.3f} |")
    return out


def write(name: str, title: str, s: pd.DataFrame, p: pd.DataFrame, note: str):
    s.to_csv(ROOT / f"results/bonus_{name}.csv", index=False)
    p.to_csv(ROOT / f"results/bonus_{name}_pairs.csv", index=False)
    md = [f"# {title}", "", note, "", "## Условия", ""] + md_table(s) + ["", "## Парные тесты (McNemar exact, Холм внутри эксперимента)", ""] + md_pairs(p)
    (ROOT / f"results/bonus_{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def bge():
    b = base_rows()
    b = b[b.method.isin(["emb", "bm25"]) & (b.k_label.isna() | b.k_label.eq(8))]
    n = ag.load([ROOT / "runs/v2_emb_bge_check"])
    d = pd.concat([b, n])
    lab = {"emb": "EMB text-embedding-3-small", "bm25": "BM25", "emb_bge": "EMB bge-m3 (локально)"}
    d = cond_df(d, lambda r: lab.get(r.method))
    d["cond"] = d["cond"].astype(str)
    s = per_cond_table(d).sort_values(["size", "cond"], key=lambda c: c.map({"8k": 0, "30k": 1, "100k": 2}) if c.name == "size" else c)
    p = mcnemar_pairs(d, [(lab["emb_bge"], lab["emb"]), (lab["emb_bge"], lab["bm25"])])
    write("bge", "Бонус: EMB на bge-m3 вместо text-embedding-3-small (A-hard, k=8, gpt-6-luna; постфактум, отдельно от главной таблицы)", s, p,
          "Те же 18 вопросов и те же ответы-модели; меняется только эмбеддер поиска. Эмбеддинги bge-m3 считались локально.")


def oprag():
    b = base_rows()
    b = b[b["size"].eq("100k") & b.method.isin(["bm25", "hyb_rerank"])]
    n = ag.load([ROOT / "runs/v2_oprag_check"])
    d = pd.concat([b, n])
    d = d[d.k_label.isin([8, 16])]
    L = {"bm25": "BM25", "bm25_doc": "BM25 порядок документа", "hyb_rerank": "HYB-rerank", "hyb_rerank_doc": "HYB-rerank порядок документа"}
    d = cond_df(d, lambda r: f"{L[r.method]} k={int(r.k_label)}" if r.method in L else None)
    s = per_cond_table(d).sort_values("cond")
    pairs = [(f"{L[m + '_doc']} k={k}", f"{L[m]} k={k}") for m in ("bm25", "hyb_rerank") for k in (8, 16)]
    p = mcnemar_pairs(d, pairs)
    write("oprag", "Бонус: OP-RAG — порядок чанков как в документе (A-hard, 100K, gpt-6-luna; постфактум, отдельно от главной таблицы)", s, p,
          "Состав найденных чанков тот же, меняется только порядок в промпте. k=8 в обычном порядке — из Ш5, k=16 в обычном порядке — контрольный прогон этого эксперимента.")


def terra():
    b = base_rows()
    b = b[b["size"].eq("100k") & b.method.isin(["bm25", "emb"]) & (b.k_label.isna() | b.k_label.eq(8)) & b.repeat.eq(1)]
    n = ag.load([ROOT / "runs/v2_terra_reader"])
    n = n[n.model.str.startswith("gpt-6.1-terra")]
    d = pd.concat([b, n])
    lab = lambda r: f"{'Terra' if r.model.startswith('gpt-6.1-terra') else 'Luna'} · {'BM25' if r.method == 'bm25' else 'EMB'}"
    d = cond_df(d, lab)
    s = per_cond_table(d).sort_values("cond")
    p = mcnemar_pairs(d, [("Terra · BM25", "Luna · BM25"), ("Terra · EMB", "Luna · EMB"), ("Terra · BM25", "Terra · EMB")], outcome_repeat=1)
    write("terra", "Бонус: более сильный читатель (gpt-6.1-terra) на тех же контекстах (A-hard, 100K, 1 повтор; постфактум, отдельно от главной таблицы)", s, p,
          "Поиск тот же (BM25 и EMB на text-embedding-3-small, k=8), читатель другой; судья остаётся gpt-6-luna. Luna — тот же 1-й повтор из Ш5. Тонкий срез «читателя», не полная ось моделей.")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("bge", bge), ("oprag", oprag), ("terra", terra)):
        if which in (name, "all") and (ROOT / f"runs/v2_{ {'bge':'emb_bge_check','oprag':'oprag_check','terra':'terra_reader'}[name]}/raw.jsonl").exists():
            fn()
