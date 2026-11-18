"""Демо R04 «Long Context vs RAG»: показывает сохранённые результаты (demo/data/). Никаких вызовов API и пересчёта моделей.
Запуск из корня репозитория:  streamlit run demo/app.py
Данные: python -m scripts.export_demo_data (в основном окружении репозитория)."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).parent / "data"
ACCENT = "#4F46E5"
INK, MUTED, LINE, CARD = "#0F172A", "#64748B", "#E2E8F0", "#FFFFFF"
OK, BAD, SKIP = "#15803D", "#B91C1C", "#B45309"
METHODS = {  # ключ: (название, цвет, штрих)
    "lc": ("Long Context", ACCENT, "solid"), "bm25": ("BM25 + LLM", "#0EA5E9", "solid"), "emb": ("Embedding RAG + LLM", "#F59E0B", "solid"),
    "hyb_rrf": ("Гибрид (RRF)", "#94A3B8", "dash"), "hyb_rerank": ("Гибрид + реранкер", "#10B981", "dash"), "agent_bm25": ("Агент (BM25)", "#EC4899", "dot"),
}
BASE3 = ["lc", "bm25", "emb"]
SIZES = ["8k", "30k", "100k"]
VERDICT = {"correct": ("✓", "Верно", OK), "incorrect": ("✗", "Неверно", BAD), "partial": ("✗", "Неверно (частично)", BAD), "abstain": ("⊘", "Отказ", SKIP)}

st.set_page_config(page_title="R04: Long Context vs RAG", page_icon="📊", layout="wide")
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; color:{INK}; }}
.banner {{ background:{ACCENT}; color:white; padding:.55rem 1rem; border-radius:10px; font-weight:500; font-size:.95rem; margin-bottom:1rem; }}
.card {{ background:{CARD}; border:1px solid {LINE}; border-radius:16px; padding:1.1rem 1.2rem; box-shadow:0 4px 14px rgba(15,23,42,.06); height:100%; }}
.big {{ font-size:2.2rem; font-weight:700; line-height:1.1; color:{INK}; }}
.cap {{ color:{MUTED}; font-size:.85rem; }}
.badge {{ display:inline-block; padding:.2rem .7rem; border-radius:999px; font-weight:600; font-size:.9rem; }}
.mark {{ background:#FEF08A; padding:0 2px; border-radius:3px; }}
.frag {{ border-left:3px solid {LINE}; padding:.3rem .7rem; margin:.4rem 0; font-size:.85rem; color:#334155; }}
.frag.gold {{ border-left-color:{ACCENT}; background:#EEF2FF; }}
.metric-row {{ display:flex; gap:1.2rem; flex-wrap:wrap; margin:.6rem 0; }}
.metric-row div {{ min-width:80px; }}
.warn {{ background:#FFFBEB; border:1px solid #FDE68A; color:#78350F; padding:.7rem 1rem; border-radius:12px; font-size:.9rem; }}
h1, h2, h3 {{ letter-spacing:-.01em; }}
</style>
<div class="banner">Демо на сохранённых результатах, платных вызовов нет</div>
""", unsafe_allow_html=True)


@st.cache_data
def load():
    rows = pd.read_json(DATA / "rows.jsonl", lines=True)
    return (rows, json.loads((DATA / "questions.json").read_text(encoding="utf-8")), json.loads((DATA / "chunks.json").read_text(encoding="utf-8")),
            pd.DataFrame(json.loads((DATA / "p1.json").read_text(encoding="utf-8"))), pd.DataFrame(json.loads((DATA / "p2.json").read_text(encoding="utf-8"))),
            json.loads((DATA / "meta.json").read_text(encoding="utf-8")))


rows, QS, CH, P1, P2, META = load()


def wilson(k: float, n: int, z: float = 1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - m), min(1.0, c + m)


def acc_table(df: pd.DataFrame, col: str, by: list[str]) -> pd.DataFrame:
    """Точность по ячейкам: среднее по повторам внутри вопроса, затем по уникальным вопросам; CI Уилсона по числу вопросов."""
    q = df.groupby(by + ["qid"])[col].mean().reset_index()
    g = q.groupby(by)[col].agg(["mean", "sum", "count"]).reset_index()
    g[["lo", "hi"]] = [wilson(s, int(n)) for s, n in zip(g["sum"], g["count"])]
    return g


def fmt_usd(x: float) -> str:
    return f"${x:.5f}" if x < 0.01 else f"${x:.4f}"


def style(fig, h=420):
    fig.update_layout(height=h, paper_bgcolor="white", plot_bgcolor="white", font=dict(family="Inter, system-ui, sans-serif", color=INK),
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.2), hoverlabel=dict(bgcolor="white"))
    fig.update_xaxes(gridcolor="#F1F5F9", zeroline=False)
    fig.update_yaxes(gridcolor="#F1F5F9", zeroline=False)
    return fig


page = st.sidebar.radio("Экран", ["1 · Вопрос", "2 · Точность и размер", "3 · Деньги", "4 · Положение факта", "5 · Выводы"], label_visibility="collapsed")
st.sidebar.caption(f"gpt-6-luna · {META['rows']} сохранённых вызовов · расход проекта ${META['spent_usd']:.2f} из $3.00")

# ---------------------------------------------------------------------------------------------------------------- 1
if page.startswith("1"):
    st.title("Один вопрос — разные методы")
    c1, c2, c3 = st.columns(3)
    qset = c1.selectbox("Набор", ["A", "A-hard", "B"], help="A — вопрос с названием сущности; A-hard — без названия, по описанию; B — ещё 42 вопроса с названием (только 100K)")
    sizes = ["100k"] if qset == "B" else SIZES
    size = c2.selectbox("Размер корпуса", sizes, format_func=lambda s: s.upper(), index=len(sizes) - 1)
    qtype = c3.selectbox("Тип вопроса", ["T1", "T2", "T3"], format_func=lambda t: {"T1": "T1 · факт", "T2": "T2 · сравнение", "T3": "T3 · противоречие"}[t])
    sub = rows[(rows.qset == qset) & (rows["size"] == size) & (rows.type == qtype)]
    qids = sorted(sub.qid.unique())
    if not qids:
        st.info("Для этого сочетания нет сохранённых результатов.")
        st.stop()
    qid = st.selectbox("Вопрос", qids, format_func=lambda i: f"{i} — {QS[i]['question'][:110]}")
    q = QS[qid]
    st.markdown(f"<div class='card'><div class='cap'>Вопрос</div><div style='font-size:1.1rem;font-weight:600'>{html.escape(q['question'])}</div>"
                f"<div class='cap' style='margin-top:.6rem'>Эталонный ответ</div><div>{html.escape(q['gold_answer'])}</div></div>", unsafe_allow_html=True)
    extra = st.toggle("Показать гибрид и агента", value=False)
    methods = BASE3 + (["hyb_rrf", "hyb_rerank", "agent_bm25"] if extra else [])
    cols = st.columns(len(methods))
    for col, m in zip(cols, methods):
        name, color, _ = METHODS[m]
        r = sub[(sub.qid == qid) & (sub.method == m)].sort_values("repeat")
        with col:
            if r.empty:
                col.markdown(f"<div class='card'><b style='color:{color}'>{name}</b><p class='cap'>не прогонялось на этом размере</p></div>", unsafe_allow_html=True)
                continue
            r = r.iloc[0]
            icon, word, vcol = VERDICT.get(r.verdict, ("?", r.verdict, MUTED))
            if r.verdict == "correct" and not r.main_ok:
                icon, word, vcol = "✗", "Неверно (нет значения в ответе)", BAD
            parts = [f"<div class='card'><b style='color:{color};font-size:1.05rem'>{name}</b><br>",
                     f"<span class='badge' style='background:{vcol}18;color:{vcol}'>{icon} {word}</span>",
                     f"<p style='margin:.7rem 0'>{html.escape(r.answer) if r.answer else '<i>—</i>'}</p>",
                     "<div class='metric-row'>",
                     f"<div><div class='big' style='font-size:1.4rem'>{(r.gen_ms or 0)/1000:.1f} с</div><div class='cap'>ответ модели</div></div>",
                     f"<div><div class='big' style='font-size:1.4rem'>{fmt_usd(r.cost_cold)}</div><div class='cap'>цена вызова</div></div>"]
            if r.evidence_recall is not None and not pd.isna(r.evidence_recall):
                parts.append(f"<div><div class='big' style='font-size:1.4rem'>{float(r.evidence_recall):.0%}</div><div class='cap'>evidence recall</div></div>")
            parts.append("</div>")
            if m == "lc":
                pos = f", факт на {r.fact_pos:.0f}% длины" if r.fact_pos is not None and not pd.isna(r.fact_pos) else ""
                ntok = f"{int(r.context_tokens or r.prompt_tokens):,}".replace(",", " ")
                parts.append(f"<div class='cap'>весь корпус в контексте: {ntok} токенов{pos}</div>")
            else:
                if m == "agent_bm25":
                    parts.append(f"<div class='cap'>шагов: {int(r.n_steps or 0)}, вызовов инструментов: {int(r.n_tool_calls or 0)}, прочитано фрагментов: {len(r.chunk_ids)}</div>")
                shown = r.chunk_ids[:6]
                parts.append(f"<div class='cap' style='margin-top:.5rem'>найденные фрагменты ({len(r.chunk_ids)}; золотой отмечен ★):</div>")
                gold_ids = set(q["gold_chunk_ids"])
                for cid in shown:
                    raw = CH.get(size, {}).get(cid, "(текст не выгружен)")
                    cores = []
                    for f in q["facts"]:
                        core = f.split(". ", 1)[-1] if f[:1].isdigit() else f
                        if core in raw:
                            cores.append(core)
                    cut = ""
                    if cores:      # окно вокруг золотого факта, чтобы подсветка была видна
                        i0 = raw.find(cores[0])
                        a, b = max(0, i0 - 110), min(len(raw), i0 + len(cores[0]) + 110)
                        raw, cut = raw[a:b], ("…" if a > 0 else "")
                    else:
                        raw = raw[:300]
                    txt = html.escape(raw)
                    for core in cores:
                        ce = html.escape(core)
                        txt = txt.replace(ce, f"<span class='mark'>{ce}</span>")
                    star = "★ " if cid in gold_ids else ""
                    parts.append(f"<div class='frag{' gold' if cid in gold_ids else ''}'><b>{star}{cid}</b><br>{cut}{txt}…</div>")
                if not any(c in gold_ids for c in r.chunk_ids):
                    parts.append(f"<div class='cap' style='color:{BAD}'>золотой фрагмент не найден</div>")
            parts.append("</div>")
            st.markdown("".join(parts), unsafe_allow_html=True)
    st.caption("Цена вызова — «холодная» (без попадания в кэш провайдера). Для A и A-hard на каждый вопрос приходится 3 повтора, здесь показан первый.")

# ---------------------------------------------------------------------------------------------------------------- 2
elif page.startswith("2"):
    st.title("Точность и размер корпуса")
    c1, c2 = st.columns([1, 2])
    qset = c1.radio("Набор", ["A-hard", "A", "B"], horizontal=True)
    metric = c2.radio("Метрика", ["Основная", "Строгая", "По судье"], horizontal=True,
                      help="Основная: судья верно И в ответе есть значение (T2 — число победителя). Строгая: T2 требует оба числа. По судье: только вердикт судьи.")
    col = {"Основная": "main_ok", "Строгая": "strict_ok", "По судье": "judge_ok"}[metric]
    df = rows[(rows.qset == qset) & (rows.method != "nocontext")]
    show_extra = st.toggle("Показать гибрид и агента", value=True)
    keys = list(METHODS) if show_extra else BASE3
    t = acc_table(df, col, ["size", "method"])
    fig = go.Figure()
    if qset == "B":
        for m in keys:
            r = t[t.method == m]
            if r.empty:
                continue
            fig.add_trace(go.Bar(x=[METHODS[m][0]], y=r["mean"], marker_color=METHODS[m][1], showlegend=False,
                                 error_y=dict(type="data", symmetric=False, array=r["hi"] - r["mean"], arrayminus=r["mean"] - r["lo"], color="#475569"),
                                 text=[f"{v:.2f}" for v in r["mean"]], textposition="outside"))
        fig.update_yaxes(range=[0, 1.12], title="точность")
        st.caption("Набор B прогонялся только на 100K, 1 повтор, 42 вопроса на метод.")
    else:
        shown = [m for m in keys if not t[t.method == m].empty]
        for j, m in enumerate(shown):
            r = t[t.method == m].set_index("size").reindex(SIZES)
            off = (j - (len(shown) - 1) / 2) * 0.045          # сдвиг по X, чтобы интервалы разных методов не накладывались
            fig.add_trace(go.Scatter(x=[i + off for i in range(3)], y=r["mean"], name=METHODS[m][0], mode="lines+markers",
                                     line=dict(color=METHODS[m][1], dash=METHODS[m][2], width=3), marker=dict(size=9),
                                     error_y=dict(type="data", symmetric=False, array=(r["hi"] - r["mean"]).fillna(0), arrayminus=(r["mean"] - r["lo"]).fillna(0), thickness=1.2, width=4),
                                     connectgaps=False))
        fig.update_yaxes(range=[0, 1.05], title="точность", tickformat=".0%")
        fig.update_xaxes(title="размер корпуса", tickmode="array", tickvals=[0, 1, 2], ticktext=[x.upper() for x in SIZES], range=[-0.4, 2.4])
    st.plotly_chart(style(fig), width="stretch")
    best = t.sort_values("mean", ascending=False).iloc[0]
    n_q = int(t["count"].max())
    a, b, c = st.columns(3)
    a.markdown(f"<div class='card'><div class='cap'>лучшая ячейка</div><div class='big'>{best['mean']:.0%}</div><div class='cap'>{METHODS[best['method']][0]}, {best['size'].upper()}</div></div>", unsafe_allow_html=True)
    b.markdown(f"<div class='card'><div class='cap'>вопросов на ячейку</div><div class='big'>{n_q}</div><div class='cap'>{'1 повтор' if qset == 'B' else '3 повтора'}, полосы — доверительный интервал Уилсона 95%</div></div>", unsafe_allow_html=True)
    mc = pd.read_csv(DATA / ("mcnemar_b_main.csv" if qset == "B" else "mcnemar_v2_main.csv"))
    mc = mc[(mc.family == "method") & (mc.qset == qset) & (mc.a != "nocontext") & (mc.b != "nocontext")] if qset != "B" else mc
    sig = int((mc.p_holm < 0.05).sum())
    c.markdown(f"<div class='card'><div class='cap'>значимых различий между методами</div><div class='big'>{sig}</div><div class='cap'>из {len(mc)} парных тестов (McNemar, поправка Холма). «Не значимо» ≠ «нет различий»</div></div>", unsafe_allow_html=True)
    if qset == "A":
        st.info("Набор A (с названием сущности): все методы 0.93–1.00, потолок — различий не видно.")

# ---------------------------------------------------------------------------------------------------------------- 3
elif page.startswith("3"):
    st.title("Деньги")
    st.markdown("<div class='warn'><b>Важно:</b> кэш провайдера для Long Context между РАЗНЫМИ вопросами в наших замерах не сработал (cached_tokens = 0). "
                "Поэтому цены ниже «холодные», а кривая «LC с кэшем» — теоретическая, не измеренная.</div>", unsafe_allow_html=True)
    st.markdown("")
    qset = st.radio("Набор", ["A-hard", "A"], horizontal=True)
    df = rows[(rows.qset == qset) & (~rows.method.isin(["nocontext"]))]
    g = df.groupby(["size", "method"]).agg(cost=("cost_cold", "sum"), ok=("main_ok", "sum"), n=("qid", "size"), per_call=("cost_cold", "mean")).reset_index()
    g["per_correct"] = g.cost / g.ok.replace(0, pd.NA)
    fig = go.Figure()
    for m, (name, color, _) in METHODS.items():
        r = g[g.method == m].set_index("size").reindex(SIZES)
        if r.per_correct.isna().all():
            continue
        fig.add_trace(go.Bar(x=[s.upper() for s in SIZES], y=r.per_correct.astype(float), name=name, marker_color=color))
    fig.update_layout(barmode="group", bargap=0.2)
    fig.update_yaxes(type="log", title="$ за верный ответ (логарифмическая шкала)")
    st.plotly_chart(style(fig, 400), width="stretch")
    both = rows[rows.qset.isin(["A", "A-hard"]) & rows.method.isin(["lc", "bm25", "emb", "hyb_rrf", "hyb_rerank"])]
    pc = both.groupby(["size", "method"]).cost_cold.mean().unstack()
    ratio = {z: pc.loc[z, "lc"] / pc.loc[z, ["bm25", "emb", "hyb_rrf", "hyb_rerank"]].mean() for z in SIZES}
    lc100 = pc.loc["100k", "lc"]
    bm100 = pc.loc["100k", ["bm25", "emb", "hyb_rrf", "hyb_rerank"]].mean()
    k1, k2, k3 = st.columns(3)
    k1.markdown(f"<div class='card'><div class='cap'>вызов LC, 100K</div><div class='big'>{fmt_usd(lc100)}</div></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='card'><div class='cap'>вызов RAG (среднее по 4 методам), 100K</div><div class='big'>{fmt_usd(bm100)}</div></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='card'><div class='cap'>LC дороже RAG на 100K</div><div class='big'>×{ratio['100k']:.0f}</div><div class='cap'>на 8K ×{ratio['8k']:.1f}, на 30K ×{ratio['30k']:.1f} (A и A-hard вместе)</div></div>", unsafe_allow_html=True)
    st.subheader("Когда LC может окупиться (по формуле)")
    be = pd.read_csv(DATA / "breakeven_gpt-6-luna.csv")
    size = st.radio("Размер", [s.upper() for s in SIZES], horizontal=True, index=0).lower()
    b = be[be["size"] == size].iloc[0]
    xs = list(range(1, 501))
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=xs, y=[b.C_full * n for n in xs], name="LC без кэша (как в замерах)", line=dict(color=ACCENT, width=3)))
    fig2.add_trace(go.Scatter(x=xs, y=[b.C_full + (n - 1) * b.C_cached for n in xs], name="LC с кэшем (теория)", line=dict(color=ACCENT, width=3, dash="dash")))
    fig2.add_trace(go.Scatter(x=xs, y=[b.C_rag * n for n in xs], name="RAG", line=dict(color="#0EA5E9", width=3)))
    fig2.update_xaxes(type="log", title="число вопросов к одному корпусу")
    fig2.update_yaxes(type="log", title="суммарная стоимость, $")
    st.plotly_chart(style(fig2, 380), width="stretch")
    nstar = b.N_star
    st.caption(f"Точка безубыточности LC с кэшем против RAG: {'после ≈' + str(nstar) + ' вопросов' if str(nstar) != 'нет' else 'не наступает'} (допущение: кэш срабатывает между вопросами — в наших замерах нет).")

# ---------------------------------------------------------------------------------------------------------------- 4
elif page.startswith("4"):
    st.title("Положение факта")
    st.markdown("Один и тот же факт переносится в другое место контекста, вопрос остаётся тем же. Это проверка «потери в середине».")
    c1, c2 = st.columns(2)

    def bar(df, xcol, xs, xlabels, title):
        t = acc_table(df, "main_ok", [xcol, "size"]) if "size" in df else acc_table(df, "main_ok", [xcol])
        fig = go.Figure()
        for size in (df["size"].unique() if "size" in df else [None]):
            r = t[t["size"] == size].set_index(xcol).reindex(xs) if size else t.set_index(xcol).reindex(xs)
            fig.add_trace(go.Bar(x=xlabels, y=r["mean"], name=(size.upper() if size else ""), text=[f"{v:.2f}" for v in r["mean"]], textposition="inside", insidetextanchor="end", textfont=dict(color="white", size=12), textangle=0,
                                 error_y=dict(type="data", symmetric=False, array=(r["hi"] - r["mean"]).fillna(0), arrayminus=(r["mean"] - r["lo"]).fillna(0), color="#475569"),
                                 marker_color=ACCENT if size in (None, "30k") else "#6366F1"))
        fig.update_yaxes(range=[0, 1.15], title="точность", tickformat=".0%")
        fig.update_layout(barmode="group", bargap=0.2)
        return style(fig, 380)

    with c1:
        st.subheader("P1 · Long Context")
        st.caption("Позиция факта в корпусе: T1, 13 вопросов на 30K и 6 на 100K")
        st.plotly_chart(bar(P1, "target_pct", [0, 50, 100], ["начало", "середина", "конец"], ""), width="stretch")
    with c2:
        st.subheader("P2 · RAG")
        st.caption("Ранг золотого фрагмента среди 10 найденных: T1, 30K, 6 вопросов")
        t2 = acc_table(P2, "main_ok", ["gold_rank", "qset"])
        fig = go.Figure()
        for qs, color in (("A", ACCENT), ("A-hard", "#A5B4FC")):
            r = t2[t2.qset == qs].set_index("gold_rank").reindex([1, 5, 10])
            fig.add_trace(go.Bar(x=["ранг 1", "ранг 5", "ранг 10"], y=r["mean"], name=qs, marker_color=ACCENT if qs == "A" else "#6366F1", text=[f"{v:.2f}" for v in r["mean"]], textposition="inside", insidetextanchor="end", textfont=dict(color="white", size=12), textangle=0,
                                 error_y=dict(type="data", symmetric=False, array=(r["hi"] - r["mean"]).fillna(0), arrayminus=(r["mean"] - r["lo"]).fillna(0), color="#475569")))
        fig.update_yaxes(range=[0, 1.15], tickformat=".0%", title="точность")
        fig.update_layout(barmode="group", bargap=0.2)
        st.plotly_chart(style(fig, 380), width="stretch")
    st.markdown("<div class='warn'><b>Как читать:</b> эффекта положения не обнаружено, но это потолок при малом числе вопросов (6–13), а не доказательство, что положение не влияет. "
                "«Потеря в середине» на A-hard 100K не проверялась.</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------------------------------------------- 5
else:
    st.title("Выводы")
    st.caption("Только то, что подтверждено нашими данными (gpt-6-luna, синтетические русские регламенты).")
    cards = [
        ("Когда Long Context", "Корпус до ~8K токенов: точность 0.93–1.00, цена вызова всего ×2.6 от RAG. Простой путь без индекса.", ACCENT),
        ("Когда RAG", "От 30K: BM25 или гибрид с реранкером. Точность сопоставима с LC (различия не доказаны), цена вызова ×8.5 (30K) и ×28 (100K) ниже.", "#0EA5E9"),
        ("Когда агент", "Сравнение двух документов и противоречия без названия сущности: агент с BM25 дал 0.98 на A-hard 100K и 1.00 на B при цене ×1.8 от RAG и задержке ~5 с.", "#EC4899"),
        ("Что хуже всех", "Чистые эмбеддинги (EMB): A-hard 100K — 0.44, на B — 0.71. Значимо хуже реранкера и агента на B (p_Холм 0.009 и 0.005).", "#F59E0B"),
        ("Что не доказано", "Различий между LC, BM25, реранкером и агентом нет (потолок, n = 18 и 42). LC против EMB на B: p = 0.013, но p_Холм = 0.104 — тенденция.", MUTED),
        ("Ограничения", "Синтетический корпус; малое n; кэш LC между вопросами не сработал; правило метрики T2 выбрано постфактум; реальных данных KOZ AI нет.", BAD),
    ]
    for i in range(0, len(cards), 3):
        cs = st.columns(3)
        for c, (t, txt, color) in zip(cs, cards[i:i + 3]):
            c.markdown(f"<div class='card' style='border-top:4px solid {color}'><div style='font-weight:700;font-size:1.05rem'>{t}</div><p style='margin-top:.5rem'>{txt}</p></div>", unsafe_allow_html=True)
        st.markdown("")
    st.subheader("Значимые различия (McNemar, поправка Холма, p_Холм < 0.1)")
    out = []
    for fn, label in (("mcnemar_b_main.csv", "B, 100K"), ("mcnemar_v2_main.csv", "A / A-hard")):
        d = pd.read_csv(DATA / fn)
        d = d[(d.family == "method") & (d.p_holm < 0.1) & (d.a != "nocontext") & (d.b != "nocontext")]
        for _, r in d.iterrows():
            out.append({"набор": label if label == "B, 100K" else r.qset, "размер": r.get("context", "100k"), "пара": f"{METHODS.get(r.a, (r.a,))[0]} против {METHODS.get(r.b, (r.b,))[0]}",
                        "только первый верен": int(r.only_a), "только второй верен": int(r.only_b), "p": round(r.p_value, 4), "p (Холм)": round(r.p_holm, 3),
                        "вывод": "значимо" if r.p_holm < 0.05 else "тенденция, не значимо"})
    st.dataframe(pd.DataFrame(out), width="stretch", hide_index=True)
