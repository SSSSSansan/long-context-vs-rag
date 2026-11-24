"""Демо R04 «Long Context vs RAG»: показывает сохранённые результаты (demo/data/). Никаких вызовов API и пересчёта моделей.
Запуск из корня репозитория:  streamlit run demo/app.py
Данные: python -m scripts.export_demo_data (в основном окружении репозитория). Стиль методов — demo/data/style.json (из configs/style.yaml)."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DATA = Path(__file__).parent / "data"
INK, MUTED, LINE, CARD, ACCENT = "#0F172A", "#475569", "#E2E8F0", "#FFFFFF", "#1F4E79"
OK, BAD, WARN = "#0F766E", "#B91C1C", "#B45309"
SIZES = ["8k", "30k", "100k"]
SYMBOL = {"o": "circle", "s": "square", "^": "triangle-up", "D": "diamond", "v": "triangle-down", "P": "cross", "X": "x", "*": "star", "x": "x-thin"}
VERDICT = {"correct": ("✓", "Верно", OK), "incorrect": ("✗", "Неверно", BAD), "partial": ("✗", "Неверно (частично)", BAD), "abstain": ("⊘", "Отказ", WARN)}
POST = "постфактум, после основных прогонов, отдельно от главной таблицы"

st.set_page_config(page_title="R04: Long Context vs RAG", page_icon="📊", layout="wide")
st.markdown(f"""
<style>
.card {{ background:{CARD}; border:1px solid {LINE}; border-radius:14px; padding:1rem 1.1rem; height:100%; }}
.big {{ font-size:2rem; font-weight:700; line-height:1.1; color:{INK}; }}
.cap {{ color:{MUTED}; font-size:.85rem; }}
.badge {{ display:inline-block; padding:.15rem .65rem; border-radius:999px; font-weight:600; font-size:.88rem; }}
.post {{ display:inline-block; background:#FEF3C7; color:#78350F; border:1px solid #FCD34D; padding:.1rem .6rem; border-radius:999px; font-size:.8rem; font-weight:600; }}
.mark {{ background:#FEF08A; padding:0 2px; border-radius:3px; }}
.frag {{ border-left:3px solid {LINE}; padding:.3rem .7rem; margin:.4rem 0; font-size:.85rem; color:#334155; }}
.frag.gold {{ border-left-color:{ACCENT}; background:#EFF6FF; }}
.warn {{ background:#FFFBEB; border:1px solid #FDE68A; color:#78350F; padding:.7rem 1rem; border-radius:12px; font-size:.9rem; }}
.step {{ border-left:4px solid {LINE}; padding:.4rem .8rem; margin:.5rem 0; background:#F8FAFC; border-radius:0 10px 10px 0; font-size:.9rem; }}
.metric-row {{ display:flex; gap:1.2rem; flex-wrap:wrap; margin:.6rem 0; }}
@media (max-width: 800px) {{ .big {{ font-size:1.5rem; }} }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------------ данные (с запасными вариантами)
def read_json(name, default):
    try:
        return json.loads((DATA / name).read_text(encoding="utf-8"))
    except Exception:
        return default


def read_csv(name) -> pd.DataFrame:
    try:
        return pd.read_csv(DATA / name)
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load():
    try:
        rows = pd.read_json(DATA / "rows.jsonl", lines=True)
    except Exception:
        rows = pd.DataFrame()
    return (rows, read_json("questions.json", {}), read_json("chunks.json", {}), pd.DataFrame(read_json("p1.json", [])), pd.DataFrame(read_json("p2.json", [])),
            read_json("meta.json", {}), read_json("style.json", {"methods": {}, "anchor": {}}), read_json("agent_traces.json", []))


rows, QS, CH, P1, P2, META, STYLE, TRACES = load()
if rows.empty:
    st.error("Нет данных демо (demo/data/rows.jsonl). Сначала выполните: python -m scripts.export_demo_data")
    st.stop()
M = STYLE.get("methods", {})
BASE3 = ["lc", "bm25", "emb"]
RAG5 = ["lc", "bm25", "emb", "hyb_rrf", "hyb_rerank"]
ALL = RAG5 + ["agent_bm25", "agent_emb", "agent_hyb"]


def mname(m):  # название с маркером: цвет — не единственный носитель смысла
    return M.get(m, {}).get("label", m)


def mcolor(m):
    return M.get(m, {}).get("color", "#64748B")


def msym(m):
    return SYMBOL.get(M.get(m, {}).get("marker", "o"), "circle")


def wilson(k, n, z=1.96):
    if not n:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def acc_table(df, col, by):
    """Среднее по повторам внутри вопроса, затем по уникальным вопросам; CI Уилсона по числу вопросов."""
    if df.empty:
        return pd.DataFrame(columns=by + ["mean", "sum", "count", "lo", "hi"])
    q = df.groupby(by + ["qid"])[col].mean().reset_index()
    g = q.groupby(by)[col].agg(["mean", "sum", "count"]).reset_index()
    g[["lo", "hi"]] = [wilson(s, int(n)) for s, n in zip(g["sum"], g["count"])]
    return g


def fmt_usd(x):
    return "—" if x is None or pd.isna(x) else (f"${x:.5f}" if x < 0.01 else f"${x:.4f}")


def card(title, big, cap, color=INK):
    return f"<div class='card'><div class='cap'>{title}</div><div class='big' style='color:{color}'>{big}</div><div class='cap'>{cap}</div></div>"


def style(fig, h=420):
    fig.update_layout(height=h, paper_bgcolor="white", plot_bgcolor="white", font=dict(color=INK), margin=dict(l=10, r=10, t=50, b=70),
                      legend=dict(orientation="h", y=-0.32, yanchor="top"), hoverlabel=dict(bgcolor="white"))
    fig.update_xaxes(gridcolor="#F1F5F9", zeroline=False)
    fig.update_yaxes(gridcolor="#F1F5F9", zeroline=False)
    return fig


def download(df, name, label="Скачать CSV"):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=name, mime="text/csv")


def post_badge():
    st.markdown(f"<span class='post'>{POST}</span>", unsafe_allow_html=True)


GLOSSARY = [
    ("Accuracy (основная)", "Доля верных ответов. Верным считается ответ, который судья признал правильным И в котором есть нужное число или имя из эталона."),
    ("Evidence recall", "Попал ли нужный фрагмент документа в то, что получила модель. У long context всегда 1.0 (видит всё). Если у RAG низкий — ломается поиск, а не модель."),
    ("Citation", "Назвал ли ответ нужный документ. Считается точным совпадением, без судьи."),
    ("quote_grounded", "Найдена ли цитата из ответа дословно в том, что получила модель. Кавычки и ссылки на пункты не мешают; пересказ или изменённое число — не засчитываются."),
    ("CI 95% (Уилсон)", "Диапазон, в котором, скорее всего, лежит настоящая accuracy. Считается по числу вопросов, а не вызовов (повторы одного вопроса — не независимые наблюдения)."),
    ("McNemar", "Парный тест: сравнивает два условия на ОДНИХ И ТЕХ ЖЕ вопросах и смотрит только на вопросы, где результаты разошлись. Если разошлись 9 против 0, p маленькое."),
    ("p и p (Холм)", "p — вероятность увидеть такое различие случайно. p (Холм) — то же с поправкой на то, что мы сделали много сравнений; значимо, только если p (Холм) < 0.05. «Не значимо» ≠ «различий нет»: при 18 вопросах тест видит только очень большие различия."),
    ("Холодная цена", "Цена вызова, когда кэш провайдера не сработал. В наших замерах кэш между разными вопросами не работал, поэтому цены холодные."),
    ("A / A-hard / B", "A — вопрос содержит название сущности (якорь). A-hard — тот же факт без названия, по описанию. B — ещё 42 вопроса с названием, только 100K."),
    ("Постфактум", "Проверка придумана после того, как результат главной таблицы уже был виден. Показывается отдельно и не меняет заранее записанные гипотезы."),
]


def glossary():
    with st.expander("Что значат термины"):
        for t, d in GLOSSARY:
            st.markdown(f"**{t}.** {d}")


PAGES = ["Ответ на исследовательский вопрос", "Один вопрос — разные методы", "Якорь: с именем и без", "Деньги", "Агент", "Положение факта", "Бонус-проверки", "Ограничения и честность"]
page = st.sidebar.radio("Экран", PAGES)
st.sidebar.caption(f"gpt-6-luna · {META.get('rows', len(rows))} сохранённых вызовов · расход проекта ${META.get('spent_usd', 0):.2f} из $3.00")
st.sidebar.info("Демо на сохранённых результатах: платных вызовов нет.")
with st.sidebar.expander("Словарик терминов"):
    for _t, _d in GLOSSARY:
        st.markdown(f"**{_t}.** {_d}")
A_ = rows[rows.qset.isin(["A", "A-hard"]) & (rows.method != "nocontext")]


try:
    # ================================================================================================================ 1
    if page == PAGES[0]:
        st.title("Ответ на исследовательский вопрос")
        st.caption("Когда long context выигрывает у RAG, а когда проигрывает? Данные: синтетические русские регламенты, gpt-6-luna, 18 вопросов × 3 повтора на ячейку.")
        a100 = A_[A_["size"] == "100k"]
        _h = a100[a100.qset == "A-hard"].groupby("method").main_ok.mean()
        _cc = a100.groupby("method").cost_cold.mean()
        _ratio = _cc.get("lc", float("nan")) / _cc[[m for m in ("bm25", "emb", "hyb_rrf", "hyb_rerank") if m in _cc.index]].mean()
        st.markdown(f"<div class='card' style='border-left:6px solid {ACCENT}'><b>Ответ за 10 секунд.</b> Доказанного победителя нет: после поправки Холма ни одна пара методов не различается значимо (по 18 вопросов). "
                    f"Решает цена: на 100K RAG (BM25, поиск с реранкером) примерно в ×{_ratio:.0f} дешевле long context за вызов при сопоставимой точности "
                    f"(A-hard: {_h.get('bm25', float('nan')):.2f} и {_h.get('hyb_rerank', float('nan')):.2f} против {_h.get('lc', float('nan')):.2f} у LC). "
                    f"Слабое место — чистые эмбеддинги: без названия сущности в вопросе они теряют факт ({_h.get('emb', float('nan')):.2f}), и виноват поиск, а не читающая модель.</div>", unsafe_allow_html=True)
        st.markdown("")
        per = a100.groupby("method").agg(cost=("cost_nojudge", "sum"), ok=("main_ok", "sum"), cc=("cost_cold", "mean"), acc=("main_ok", "mean")).reset_index()
        per["usd_pc"] = per.cost / per.ok.replace(0, pd.NA)
        good = per[(per.acc >= 0.9) & per.method.isin(RAG5)].sort_values("usd_pc")
        best = good.iloc[0] if len(good) else None
        lc_c = per[per.method == "lc"].cc.mean()
        rag_c = per[per.method.isin(["bm25", "emb", "hyb_rrf", "hyb_rerank"])].cc.mean()
        emb = A_[(A_.method == "emb") & (A_["size"] == "100k")].groupby("qset").main_ok.mean()
        nc = rows[rows.method == "nocontext"]
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(card("Лучше всех по цене и качеству (100K)", mname(best.method) if best is not None else "—",
                         f"accuracy {best.acc:.2f}, {fmt_usd(best.usd_pc)} за верный ответ без судьи (среди методов с accuracy ≥ 0.9)" if best is not None else "", ACCENT), unsafe_allow_html=True)
        c2.markdown(card("LC дороже RAG за вызов (100K)", f"×{lc_c / rag_c:.0f}", f"{fmt_usd(lc_c)} против {fmt_usd(rag_c)}; холодная цена, кэш между вопросами не сработал", WARN), unsafe_allow_html=True)
        c3.markdown(card("EMB без якоря, 100K", f"{emb.get('A', float('nan')):.2f} → {emb.get('A-hard', float('nan')):.2f}", "accuracy с якорем → без якоря; после поправки Холма p = 0.059, не значимо", BAD), unsafe_allow_html=True)
        c4.markdown(card("NO-CONTEXT: утечек", f"{int(nc.main_ok.sum())} из {len(nc)}", "без контекста модель отказывается, а не отвечает из общих знаний", OK), unsafe_allow_html=True)
        st.markdown("")
        cc1, cc2 = st.columns([1, 1])
        size = cc1.radio("Размер корпуса", SIZES, format_func=str.upper, horizontal=True, index=2)
        qset = cc2.radio("Набор вопросов", ["A-hard", "A"], horizontal=True, help="A — вопрос содержит название сущности; A-hard — то же без названия, по описанию")
        sub = rows[(rows.qset == qset) & (rows["size"] == size) & (rows.method != "nocontext")]
        t = acc_table(sub, "main_ok", ["method"]).set_index("method")
        ex = sub.groupby("method").agg(er=("evidence_recall", "mean"), cit=("citation", "mean"), ab=("abstained", "mean"), cost=("cost_nojudge", "sum"), ok=("main_ok", "sum"), cc=("cost_cold", "mean"),
                                       p50=("gen_ms", lambda s: s.median() / 1000))
        tab = t.join(ex)
        tab["usd_pc"] = tab.cost / tab.ok.replace(0, pd.NA)
        tab = tab.reindex([m for m in ALL if m in tab.index])
        out = pd.DataFrame({"Метод": [("● " if m == "lc" else "▲ " if m == "emb" else "■ ") + mname(m) for m in tab.index], "Accuracy": tab["mean"].round(3).values,
                            "CI 95%": [f"{lo:.2f}–{hi:.2f}" for lo, hi in zip(tab.lo, tab.hi)], "Вопросов": tab["count"].astype(int).values,
                            "Evidence recall": tab.er.round(2).values, "Citation": tab.cit.round(2).values, "Отказы": tab.ab.round(2).values,
                            "$/вызов": tab.cc.round(5).values, "$/верный": tab.usd_pc.astype(float).round(5).values, "Задержка p50, с": tab.p50.round(1).values})
        st.dataframe(out, hide_index=True, width="stretch", column_config={
            "Accuracy": st.column_config.ProgressColumn("Accuracy", min_value=0, max_value=1, format="%.2f"),
            "Evidence recall": st.column_config.ProgressColumn("Evidence recall", min_value=0, max_value=1, format="%.2f")})
        st.caption("$/вызов — холодная цена одного вызова вместе с судьёй; $/верный — расход на один верный ответ без судьи (как в results/main_table_v2.md). Evidence recall, Citation, Отказы — доли от 0 до 1.")
        download(out, f"table_{qset}_{size}.csv")
        fig = go.Figure()
        for m in [m for m in ALL if m in set(A_[A_.qset == qset].method)]:
            r = acc_table(A_[(A_.qset == qset) & (A_.method == m)], "main_ok", ["size"]).set_index("size").reindex(SIZES)
            cost = A_[(A_.qset == qset) & (A_.method == m)].groupby("size").cost_cold.mean().reindex(SIZES)
            fig.add_trace(go.Scatter(x=[s.upper() for s in SIZES], y=r["mean"], name=mname(m), mode="lines+markers", line=dict(color=mcolor(m), width=2.5),
                                     marker=dict(symbol=msym(m), size=10, color=mcolor(m)),
                                     error_y=dict(type="data", symmetric=False, array=(r.hi - r["mean"]).fillna(0), arrayminus=(r["mean"] - r.lo).fillna(0), thickness=1, width=3),
                                     customdata=pd.DataFrame({"n": r["count"], "lo": r.lo, "hi": r.hi, "c": cost}).values,
                                     hovertemplate="%{fullData.name}<br>accuracy %{y:.2f} (CI %{customdata[1]:.2f}–%{customdata[2]:.2f})<br>вопросов %{customdata[0]:.0f}<br>цена вызова $%{customdata[3]:.5f}<extra></extra>"))
        fig.update_yaxes(range=[0, 1.05], title="accuracy (доля верных ответов)", tickformat=".0%")
        fig.update_xaxes(title="размер корпуса, токенов")
        fig.update_layout(title=f"Accuracy по размеру корпуса, набор {qset}")
        st.plotly_chart(style(fig, 420), width="stretch")
        st.markdown("<div class='warn'><b>Как читать:</b> после поправки Холма ни одна пара методов не различается значимо при n = 18; различия — тенденции. "
                    "Решение определяет цена (RAG на порядок дешевле LC на 100K) и поведение EMB без лексического якоря.</div>", unsafe_allow_html=True)
        glossary()

    # ================================================================================================================ 2
    elif page == PAGES[1]:
        st.title("Один вопрос — разные методы")
        c1, c2, c3 = st.columns(3)
        qset = c1.selectbox("Набор", ["A", "A-hard", "B"], help="A — вопрос с названием сущности; A-hard — без названия; B — ещё 42 вопроса с названием (только 100K)")
        sizes = ["100k"] if qset == "B" else SIZES
        size = c2.selectbox("Размер корпуса", sizes, format_func=str.upper, index=len(sizes) - 1)
        qtype = c3.selectbox("Тип вопроса", ["T1", "T2", "T3"], format_func=lambda t: {"T1": "T1 · факт", "T2": "T2 · сравнение", "T3": "T3 · противоречие"}[t])
        sub = rows[(rows.qset == qset) & (rows["size"] == size) & (rows.type == qtype)]
        qids = sorted(sub.qid.unique())
        if not qids:
            st.info("Для этого сочетания нет сохранённых результатов.")
            st.stop()
        qid = st.selectbox("Вопрос", qids, format_func=lambda i: f"{i} — {QS.get(i, {}).get('question', '')[:110]}")
        q = QS.get(qid, {"question": qid, "gold_answer": "", "facts": [], "gold_chunk_ids": []})
        st.markdown(f"<div class='card'><div class='cap'>Вопрос</div><div style='font-size:1.1rem;font-weight:600'>{html.escape(q['question'])}</div>"
                    f"<div class='cap' style='margin-top:.6rem'>Эталонный ответ</div><div>{html.escape(q['gold_answer'])}</div></div>", unsafe_allow_html=True)
        extra = st.toggle("Показать гибрид и агента", value=False)
        methods = BASE3 + (["hyb_rrf", "hyb_rerank", "agent_bm25"] if extra else [])
        for col, m in zip(st.columns(len(methods)), methods):
            r = sub[(sub.qid == qid) & (sub.method == m)].sort_values("repeat")
            with col:
                if r.empty:
                    col.markdown(f"<div class='card'><b style='color:{mcolor(m)}'>{mname(m)}</b><p class='cap'>не прогонялось на этом размере</p></div>", unsafe_allow_html=True)
                    continue
                r = r.iloc[0]
                icon, word, vcol = VERDICT.get(r.verdict, ("?", r.verdict, MUTED))
                if r.verdict == "correct" and not r.main_ok:
                    icon, word, vcol = "✗", "Неверно (нет значения в ответе)", BAD
                parts = [f"<div class='card'><b style='color:{mcolor(m)};font-size:1.05rem'>{mname(m)}</b><br>",
                         f"<span class='badge' style='background:{vcol}18;color:{vcol}'>{icon} {word}</span>",
                         f"<p style='margin:.7rem 0'>{html.escape(r.answer) if r.answer else '<i>—</i>'}</p>", "<div class='metric-row'>",
                         f"<div><div class='big' style='font-size:1.4rem'>{(r.gen_ms or 0) / 1000:.1f} с</div><div class='cap'>ответ модели</div></div>",
                         f"<div><div class='big' style='font-size:1.4rem'>{fmt_usd(r.cost_cold)}</div><div class='cap'>цена вызова</div></div>"]
                if r.evidence_recall is not None and not pd.isna(r.evidence_recall):
                    parts.append(f"<div><div class='big' style='font-size:1.4rem'>{float(r.evidence_recall):.0%}</div><div class='cap'>evidence recall</div></div>")
                parts.append("</div>")
                if m == "lc":
                    pos = f", факт на {r.fact_pos:.0f}% длины" if r.fact_pos is not None and not pd.isna(r.fact_pos) else ""
                    parts.append(f"<div class='cap'>весь корпус в контексте: {int(r.context_tokens or r.prompt_tokens):,} токенов{pos}</div>".replace(",", " "))
                else:
                    if m == "agent_bm25":
                        parts.append(f"<div class='cap'>шагов: {int(r.n_steps or 0)}, вызовов инструментов: {int(r.n_tool_calls or 0)}, прочитано фрагментов: {len(r.chunk_ids)}</div>")
                    parts.append(f"<div class='cap' style='margin-top:.5rem'>найденные фрагменты ({len(r.chunk_ids)}; золотой отмечен ★):</div>")
                    gold_ids = set(q.get("gold_chunk_ids", []))
                    for cid in list(r.chunk_ids)[:6]:
                        raw = CH.get(size, {}).get(cid, "(текст не выгружен)")
                        cores = [f.split(". ", 1)[-1] if f[:1].isdigit() else f for f in q.get("facts", [])]
                        cores = [c for c in cores if c in raw]
                        cut = ""
                        if cores:
                            i0 = raw.find(cores[0])
                            a, b = max(0, i0 - 110), min(len(raw), i0 + len(cores[0]) + 110)
                            raw, cut = raw[a:b], ("…" if a > 0 else "")
                        else:
                            raw = raw[:300]
                        txt = html.escape(raw)
                        for core in cores:
                            txt = txt.replace(html.escape(core), f"<span class='mark'>{html.escape(core)}</span>")
                        parts.append(f"<div class='frag{' gold' if cid in gold_ids else ''}'><b>{'★ ' if cid in gold_ids else ''}{cid}</b><br>{cut}{txt}…</div>")
                    if not any(c in gold_ids for c in r.chunk_ids):
                        parts.append(f"<div class='cap' style='color:{BAD}'>золотой фрагмент не найден</div>")
                parts.append("</div>")
                st.markdown("".join(parts), unsafe_allow_html=True)
        st.caption("Цена вызова — «холодная» (без попадания в кэш). Для A и A-hard на каждый вопрос 3 повтора, здесь показан первый.")

    # ================================================================================================================ 3
    elif page == PAGES[2]:
        st.title("Якорь: с именем и без")
        st.markdown("Один и тот же факт спрашивается дважды: **A** — с названием сущности (лексический «якорь»), **A-hard** — только по описанию. Лексический поиск и длинный контекст якорь не нужен; поиск по эмбеддингам без него теряет факт.")
        t = acc_table(A_[A_.method.isin(RAG5)], "main_ok", ["qset", "size", "method"])
        fig = make_subplots(rows=1, cols=3, shared_yaxes=True, subplot_titles=[f"Корпус {s.upper()}" for s in SIZES])
        for ci, z in enumerate(SIZES, start=1):
            for m in RAG5:
                for qs, dx, sym_, filled in (("A", -0.14, "circle", True), ("A-hard", 0.14, "diamond", False)):
                    g = t[(t["size"] == z) & (t.method == m) & (t.qset == qs)]
                    if g.empty:
                        continue
                    r = g.iloc[0]
                    fig.add_trace(go.Scatter(x=[RAG5.index(m) + dx], y=[r["mean"]], mode="markers", showlegend=False,
                                             marker=dict(symbol=sym_ if filled else sym_ + "-open", size=11, color=mcolor(m), line=dict(width=2, color=mcolor(m))),
                                             error_y=dict(type="data", symmetric=False, array=[r.hi - r["mean"]], arrayminus=[r["mean"] - r.lo], color=mcolor(m), thickness=1.2, width=4),
                                             hovertemplate=f"{mname(m)} · {qs}<br>accuracy {r['mean']:.2f} (CI {r.lo:.2f}–{r.hi:.2f}), n = {int(r['count'])}<extra></extra>"), row=1, col=ci)
            fig.update_xaxes(tickmode="array", tickvals=list(range(len(RAG5))), ticktext=[mname(m) for m in RAG5], row=1, col=ci)
        fig.update_yaxes(range=[0.2, 1.08], title="accuracy (доля верных)", row=1, col=1)
        fig.update_xaxes(title="метод", row=1, col=2)
        st.plotly_chart(style(fig, 430), width="stretch")
        st.caption("Закрашенный круг — A (с якорем), пустой ромб — A-hard (без якоря); цвет и форма маркера соответствуют методу; усы — 95% CI Уилсона, n = 18 вопросов × 3 повтора.")
        st.subheader("Парный тест: A против A-hard (McNemar exact, поправка Холма по 15 тестам)")
        glossary()
        variant = st.radio("Метрика", ["Основная", "Строгая"], horizontal=True)
        an = read_csv("mcnemar_v2_anchor_main.csv" if variant == "Основная" else "mcnemar_v2_anchor_strict.csv")
        if an.empty:
            st.info("Таблица парного теста не выгружена (python -m scripts.mcnemar_v2 и scripts.export_demo_data).")
        else:
            an = an.copy()
            an["вывод"] = ["значимо" if p < 0.05 else "тенденция" if raw < 0.05 else "нет различий" for p, raw in zip(an.p_holm, an.p_value)]
            show = an.assign(method=an.method.map(mname), size=an["size"].str.upper())[["method", "size", "n_pairs", "acc_a", "acc_b", "only_a", "only_b", "p_value", "p_holm", "вывод"]]
            show.columns = ["Метод", "Размер", "Пар", "acc A", "acc A-hard", "только A верно", "только A-hard верно", "p", "p (Холм)", "Вывод"]
            st.dataframe(show.round(4), hide_index=True, width="stretch")
            download(show, "anchor_paired_test.csv")
            st.markdown("<div class='warn'>Единственная ячейка с заметным эффектом — EMB на 100K: 9 вопросов верны только с якорем, 0 — только без; p = 0.0039, но <b>p (Холм) = 0.059</b> — не значимо. "
                        "При n = 18 тест способен подтвердить только очень большую разницу (≥ 0.56 после поправки): «не значимо» здесь — предел чувствительности.</div>", unsafe_allow_html=True)
        st.subheader("Пары вопросов рядом")
        c1, c2 = st.columns(2)
        qtype = c1.selectbox("Тип", ["T1", "T2", "T3"], key="pt")
        pairs = sorted(q for q, v in QS.items() if v.get("qset") == "A-hard" and v.get("type") == qtype and v.get("parent_qid") in QS)
        if pairs:
            hq = c2.selectbox("Пара", pairs, format_func=lambda i: f"{QS[i]['parent_qid']} ↔ {i}")
            pq = QS[hq]["parent_qid"]
            ca, cb = st.columns(2)
            ca.markdown(f"<div class='card'><span class='badge' style='background:#DBEAFE;color:#1E3A8A'>A · с якорем</span><p style='margin-top:.5rem'>{html.escape(QS[pq]['question'])}</p><div class='cap'>эталон: {html.escape(QS[pq]['gold_answer'])}</div></div>", unsafe_allow_html=True)
            cb.markdown(f"<div class='card'><span class='badge' style='background:#FCE7F3;color:#831843'>A-hard · без якоря</span><p style='margin-top:.5rem'>{html.escape(QS[hq]['question'])}</p><div class='cap'>эталон: {html.escape(QS[hq]['gold_answer'])}</div></div>", unsafe_allow_html=True)
            z = st.radio("Размер", SIZES, index=2, format_func=str.upper, horizontal=True, key="pz")
            res = []
            for m in RAG5:
                ra = rows[(rows.qid == pq) & (rows.method == m) & (rows["size"] == z)].main_ok
                rh = rows[(rows.qid == hq) & (rows.method == m) & (rows["size"] == z)].main_ok
                if len(ra) and len(rh):
                    res.append({"Метод": mname(m), "A: верных повторов": f"{int(ra.sum())} из {len(ra)}", "A-hard: верных повторов": f"{int(rh.sum())} из {len(rh)}"})
            if res:
                st.dataframe(pd.DataFrame(res), hide_index=True, width="stretch")

    # ================================================================================================================ 4
    elif page == PAGES[3]:
        st.title("Деньги")
        st.markdown("<div style='background:#FEE2E2;border:2px solid #B91C1C;color:#7F1D1D;padding:.9rem 1.1rem;border-radius:12px;font-size:1.05rem'>"
                    "<b>⚠ Кэш между РАЗНЫМИ вопросами в наших замерах НЕ сработал</b> (cached_tokens = 0): кэш был только при повторах одного и того же вопроса. "
                    "Все цены ниже «холодные». Ползунок «доля кэша» — сценарий «что если у провайдера в проде кэш сработает», а не измерение.</div>", unsafe_allow_html=True)
        st.markdown("")
        prices, tok = META.get("prices", {}), META.get("tokens", {})
        if not prices or not tok:
            st.info("Нет цен и токенов в meta.json: выполните scripts.export_demo_data.")
            st.stop()
        c1, c2, c3 = st.columns(3)
        n_month = c1.slider("Вопросов в месяц", 0, 1_000_000, 10_000, step=100)
        model = c2.radio("Модель", ["gpt-6-luna", "gpt-6.1-terra"], format_func=lambda x: {"gpt-6-luna": "Luna", "gpt-6.1-terra": "Terra"}[x], horizontal=True)
        cache = c3.slider("Доля кэша LC, %", 0, 90, 0, step=5) / 100
        c4, c5 = st.columns(2)
        size = c4.radio("Размер корпуса", SIZES, index=2, format_func=str.upper, horizontal=True)
        fb = c5.slider("Доля отказов RAG → вызов LC (Self-Route), %", 0, 50, int(round(META.get("fallback_bm25", 0.03) * 100))) / 100
        p = prices[model]
        cw, cin, cout = p.get("cache_write_input", p["input"]), p["cached_input"], p["output"]

        def call_cost(prompt_tokens, out_tokens, share=0.0):
            return (prompt_tokens * (share * cin + (1 - share) * cw) + out_tokens * cout) / 1e6

        t_lc = tok[f"lc_{size}"]
        lc_cold, lc_cached = call_cost(t_lc, tok["lc_completion"], 0.0), call_cost(t_lc, tok["lc_completion"], cache)
        rag = call_cost(tok["rag"], tok["rag_completion"])
        agent = call_cost(tok["agent_prompt"], tok["agent_completion"])
        sr = rag + fb * lc_cold
        if n_month == 0:
            st.info("0 вопросов в месяц — стоимость 0 $. Сдвиньте ползунок «Вопросов в месяц», чтобы увидеть сравнение.")
        costs = {"LC": lc_cached, "RAG (BM25)": rag, "AGENT": agent, "Self-Route": sr}
        cols = st.columns(4)
        for col, (name, c) in zip(cols, costs.items()):
            col.markdown(card(f"{name}: в месяц", f"${c * n_month:,.0f}".replace(",", " ") if c * n_month >= 10 else f"${c * n_month:.2f}", f"{fmt_usd(c)} за вызов"), unsafe_allow_html=True)
        fig = go.Figure(go.Bar(x=list(costs), y=[max(c * n_month, 1e-9) for c in costs.values()], marker_color=[mcolor("lc"), mcolor("bm25"), mcolor("agent_bm25"), "#6B7280"],
                               marker_pattern_shape=["", "/", "+", "x"], text=[f"${c * n_month:,.2f}" if c * n_month < 100 else f"${c * n_month:,.0f}" for c in costs.values()], textposition="outside"))
        fig.update_yaxes(title="стоимость в месяц, $ (лог. шкала)", type="log", dtick=1, tickprefix="$", tickformat=",.0f")
        fig.update_xaxes(title="стратегия")
        fig.update_layout(title=f"Корпус {size.upper()}, " + f"{n_month:,}".replace(",", " ") + " вопросов в месяц")
        st.plotly_chart(style(fig, 380), width="stretch")
        # безубыточность: N вопросов к одному корпусу; LC: первый вызов холодный, дальше с долей кэша
        ns = list(range(1, 1001))
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=ns, y=[lc_cold + (n - 1) * lc_cached for n in ns], name="LC", line=dict(color=mcolor("lc"), width=3)))
        fig2.add_trace(go.Scatter(x=ns, y=[rag * n for n in ns], name="RAG (BM25)", line=dict(color=mcolor("bm25"), width=3, dash="dash")))
        fig2.add_trace(go.Scatter(x=ns, y=[agent * n for n in ns], name="AGENT", line=dict(color=mcolor("agent_bm25"), width=3, dash="dot")))
        fig2.add_trace(go.Scatter(x=ns, y=[sr * n for n in ns], name="Self-Route", line=dict(color="#6B7280", width=3, dash="dashdot")))
        fig2.update_xaxes(type="log", title="число вопросов к одному корпусу")
        fig2.update_yaxes(type="log", title="накопленная стоимость, $ (лог. шкала)", dtick=1, tickprefix="$", tickformat=",.4f")
        st.plotly_chart(style(fig2, 380), width="stretch")
        cross = next((n for n in ns if lc_cold + (n - 1) * lc_cached <= rag * n), None)
        if lc_cold <= rag:
            st.success("При этих параметрах LC дешевле RAG уже на первом вопросе.")
        elif cross:
            st.info(f"Точка безубыточности LC против RAG: после ≈ {cross} вопросов к одному корпусу (при доле кэша {cache:.0%}).")
        else:
            st.warning(f"При доле кэша {cache:.0%} LC не окупается против RAG на {size.upper()} ни при каком разумном числе вопросов (до 1000).")
        st.caption(f"Токены на вызов измерены в прогонах: LC {t_lc:,.0f}, RAG {tok['rag']:,.0f}, агент {tok['agent_prompt']:,.0f} (сумма по шагам); ответ ≈ {tok['rag_completion']:.0f} токенов. "
                   "Цена — холодная: запись в кэш 1.25× входа. Судья и поиск (эмбеддинги, реранкер) не входят.".replace(",", " "))

    # ================================================================================================================ 5
    elif page == PAGES[4]:
        st.title("Агент: пошаговая трассировка")
        if not TRACES:
            st.info("Трассы агента не выгружены (scripts.export_demo_data).")
            st.stop()
        tr = pd.DataFrame([{k: v for k, v in t.items() if k != "steps"} for t in TRACES])
        c1, c2, c3 = st.columns(3)
        meth = c1.selectbox("Вариант агента", ["agent_bm25", "agent_hyb", "agent_emb"], format_func=mname)
        qt = c2.selectbox("Тип", ["T1", "T2", "T3"])
        cand = [t for t in TRACES if t["method"] == meth and t["type"] == qt]
        ep = c3.selectbox("Эпизод", range(len(cand)), format_func=lambda i: f"{cand[i]['qid']} — {cand[i]['question'][:70]}")
        t = cand[ep]
        icon, word, vcol = VERDICT.get(t["verdict"], ("?", t["verdict"], MUTED))
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(card("вызовов инструментов", str(t["n_tool_calls"]), f"шагов цикла: {t['n_steps']}"), unsafe_allow_html=True)
        k2.markdown(card("tool_call_accuracy", f"{(t['tool_call_accuracy'] if t['tool_call_accuracy'] is not None else float('nan')):.2f}", "доля структурно корректных вызовов"), unsafe_allow_html=True)
        k3.markdown(card("цена эпизода", fmt_usd(t["cost_cold"]), "холодная, с судьёй"), unsafe_allow_html=True)
        k4.markdown(card("вердикт", f"{icon} {word}", f"search_recall {t['search_recall']:.2f}" if t["search_recall"] is not None else "", vcol), unsafe_allow_html=True)
        st.markdown(f"<div class='step' style='border-left-color:{ACCENT}'><b>1 · Вопрос</b><br>{html.escape(t['question'])}<br><span class='cap'>эталон: {html.escape(t['gold'])}</span></div>", unsafe_allow_html=True)
        for i, s in enumerate(t["steps"], start=2):
            if s["kind"] == "call":
                try:
                    a = json.loads(s["args"]); a_txt = ", ".join(f"{k}={v!r}" for k, v in a.items())
                except Exception:
                    a_txt = s["args"]
                st.markdown(f"<div class='step' style='border-left-color:#0072B2'><b>{i} · вызов {html.escape(s['name'])}</b><br><code>{html.escape(a_txt)}</code></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='step'><b>{i} · результат {html.escape(s['name'])}</b><br><span class='cap'>{html.escape(s['content'][:500])}{'…' if len(s['content']) > 500 else ''}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='step' style='border-left-color:{vcol}'><b>Ответ</b><br>{html.escape(t['answer'] or '—')}<br><span class='cap'>цитата: {html.escape(t['quote'] or '—')} · источники: {', '.join(t['doc_ids']) or '—'}</span></div>", unsafe_allow_html=True)
        page_md = f"# Ответ на вопрос\n\n{t['answer']}\n\n## Источники\n" + "\n".join(f"- {d}" for d in t["doc_ids"])
        with st.expander("Страница в Notion-моке (предпросмотр)"):
            st.markdown(page_md)
            st.caption("В зачётных прогонах запись в Notion была выключена (notion_instruction: false); инструмент write_to_notion — мок без сети, он пишет такой .md в runs/notion_mock. Здесь показано, что был бы записан для этого ответа.")

    # ================================================================================================================ 6
    elif page == PAGES[5]:
        st.title("Положение факта")
        st.markdown("Один и тот же факт переносится в другое место контекста, вопрос остаётся тем же. Это проверка «потери в середине».")
        c1, c2 = st.columns(2)

        def bars(t, xs, labels, color_by, names):
            fig = go.Figure()
            for key in names:
                r = t[t[color_by] == key].set_index(t.columns[0]).reindex(xs) if color_by in t else t.set_index(t.columns[0]).reindex(xs)
                fig.add_trace(go.Bar(x=labels, y=r["mean"], name=str(key).upper(), text=[f"{v:.2f}" for v in r["mean"]], textposition="inside", insidetextanchor="start", textfont=dict(color="white" if z == "30k" else "black"),
                                     error_y=dict(type="data", symmetric=False, array=(r.hi - r["mean"]).fillna(0), arrayminus=(r["mean"] - r.lo).fillna(0), color="#475569"),
                                     customdata=r["count"], hovertemplate="accuracy %{y:.2f}, вопросов %{customdata}<extra></extra>"))
            fig.update_yaxes(range=[0, 1.15], title="accuracy", tickformat=".0%")
            fig.update_layout(barmode="group", bargap=0.2)
            return style(fig, 380)

        with c1:
            st.subheader("P1 · Long Context")
            st.caption("Позиция факта в корпусе: T1, 13 вопросов на 30K и 6 на 100K")
            if P1.empty:
                st.info("нет данных")
            else:
                t1 = acc_table(P1, "main_ok", ["target_pct", "size"])
                fig = go.Figure()
                for z, pat in (("30k", ""), ("100k", "/")):
                    r = t1[t1["size"] == z].set_index("target_pct").reindex([0, 50, 100])
                    fig.add_trace(go.Bar(x=["начало", "середина", "конец"], y=r["mean"], name=z.upper(), marker_pattern_shape=pat, marker_color=mcolor("lc") if z == "30k" else "#56B4E9",
                                         text=[f"{v:.2f}" for v in r["mean"]], textposition="inside", textfont=dict(color="white"),
                                         error_y=dict(type="data", symmetric=False, array=(r.hi - r["mean"]).fillna(0), arrayminus=(r["mean"] - r.lo).fillna(0), color="#475569"),
                                         customdata=r["count"], hovertemplate="accuracy %{y:.2f}, вопросов %{customdata}<extra></extra>"))
                fig.update_yaxes(range=[0, 1.15], title="accuracy (доля верных)", tickformat=".0%"); fig.update_xaxes(title="положение факта в корпусе"); fig.update_layout(barmode="group", bargap=0.2)
                st.plotly_chart(style(fig, 380), width="stretch")
        with c2:
            st.subheader("P2 · RAG")
            st.caption("Ранг золотого фрагмента среди 10 найденных: T1, 30K, 6 вопросов")
            if P2.empty:
                st.info("нет данных")
            else:
                t2 = acc_table(P2, "main_ok", ["gold_rank", "qset"])
                fig = go.Figure()
                for qs, pat in (("A", ""), ("A-hard", "/")):
                    r = t2[t2.qset == qs].set_index("gold_rank").reindex([1, 5, 10])
                    fig.add_trace(go.Bar(x=["ранг 1", "ранг 5", "ранг 10"], y=r["mean"], name=qs, marker_pattern_shape=pat, marker_color=mcolor("bm25") if qs == "A" else "#CC79A7",
                                         text=[f"{v:.2f}" for v in r["mean"]], textposition="inside", insidetextanchor="start", textfont=dict(color="black"),
                                         error_y=dict(type="data", symmetric=False, array=(r.hi - r["mean"]).fillna(0), arrayminus=(r["mean"] - r.lo).fillna(0), color="#475569")))
                fig.update_yaxes(range=[0, 1.15], tickformat=".0%", title="accuracy (доля верных)"); fig.update_xaxes(title="ранг золотого фрагмента в выдаче"); fig.update_layout(barmode="group", bargap=0.2)
                st.plotly_chart(style(fig, 380), width="stretch")
        st.markdown("<div class='warn'><b>Как читать:</b> эффекта положения не обнаружено, но это потолок при малом числе вопросов (6–13), а не доказательство, что положение не влияет. «Потеря в середине» на A-hard 100K не проверялась.</div>", unsafe_allow_html=True)

    # ================================================================================================================ 7
    elif page == PAGES[6]:
        st.title("Бонус-проверки")
        post_badge()
        st.caption("Четыре точечные проверки, сделанные после того, как главная таблица уже была готова. Они НЕ входят в main_table_v2 и не меняют гипотезы, записанные до прогонов. Любой исход — результат.")
        glossary()
        tabs = st.tabs(["EMB на bge-m3", "OP-RAG: порядок чанков", "Читатель Terra", "Self-Route и роутер"])

        def block(name, note):
            s, p = read_csv(f"bonus_{name}.csv"), read_csv(f"bonus_{name}_pairs.csv")
            if s.empty:
                st.info("Эксперимент не запускался или данные не выгружены.")
                return
            show = s[["size", "cond", "accuracy", "acc_ci_low", "acc_ci_high", "evidence_recall", "citation", "abstain", "usd_per_correct", "n_calls"]].copy()
            show["size"] = show["size"].str.upper()
            show["CI 95%"] = [f"{a:.2f}–{b:.2f}" for a, b in zip(show.acc_ci_low, show.acc_ci_high)]
            show = show[["size", "cond", "accuracy", "CI 95%", "evidence_recall", "citation", "abstain", "usd_per_correct", "n_calls"]]
            show.columns = ["Размер", "Условие", "Accuracy", "CI 95%", "Evidence recall", "Citation", "Отказы", "$ за верный", "Вызовов"]
            show = show.round({"Accuracy": 2, "Evidence recall": 2, "Citation": 2, "Отказы": 2, "$ за верный": 5})
            st.dataframe(show, hide_index=True, width="stretch", column_config={"Accuracy": st.column_config.ProgressColumn("Accuracy", min_value=0, max_value=1, format="%.2f")})
            download(show, f"bonus_{name}.csv")
            if not p.empty:
                pp = p[["size", "a", "b", "n_pairs", "only_a", "only_b", "p_value", "p_holm"]].copy()
                pp["size"] = pp["size"].str.upper()
                pp["вывод"] = ["значимо" if h < 0.05 else "не значимо" for h in pp.p_holm]
                pp.columns = ["Размер", "A", "B", "Пар", "только A верно", "только B верно", "p", "p (Холм)", "Вывод"]
                st.markdown("**Парные тесты** (McNemar exact, Холм внутри эксперимента)")
                st.dataframe(pp.round({"p": 4, "p (Холм)": 3}), hide_index=True, width="stretch")
            st.markdown(f"<div class='warn'>{note}</div>", unsafe_allow_html=True)

        notes = read_json("bonus_notes.json", {})
        with tabs[0]:
            block("bge", notes.get("bge", "Выводы — в REPORT.md."))
        with tabs[1]:
            block("oprag", notes.get("oprag", "Выводы — в REPORT.md."))
        with tabs[2]:
            block("terra", notes.get("terra", "Выводы — в REPORT.md."))
        with tabs[3]:
            sim = read_csv("routing_sim.csv")
            if sim.empty:
                st.info("Симуляция не выгружена.")
            else:
                st.markdown("**Симуляция по сохранённым ответам, не отдельный прогон ($0).** RAG отвечает; при отказе берётся ответ LC на тот же вопрос и повтор. Роутер по типу: T1 → RAG, T2/T3 → LC (oracle).")
                c1, c2, c3 = st.columns(3)
                qs = c1.selectbox("Набор", sorted(sim.qset.unique()), index=sorted(sim.qset.unique()).index("A-hard") if "A-hard" in set(sim.qset) else 0)
                sizes_ = [z for z in SIZES if z in set(sim[sim.qset == qs]["size"])]
                z = c2.selectbox("Размер", sizes_, index=len(sizes_) - 1, format_func=str.upper)
                rm = c3.selectbox("RAG-метод", sorted(sim.rag_method.unique()), format_func=mname)
                g = sim[(sim.qset == qs) & (sim["size"] == z) & (sim.rag_method == rm)].copy()
                lab = {"rag": "чистый RAG", "lc": "чистый LC", "self_route": "Self-Route", "router": "роутер по типу"}
                g["Стратегия"] = g.strategy.map(lab)
                fig = go.Figure(go.Scatter(x=g.cost_per_call, y=g.accuracy, mode="markers+text", text=g["Стратегия"], textposition="top center",
                                           marker=dict(size=16, color=["#E69F00", "#0072B2", "#6B7280", "#009E73"][: len(g)], symbol=["square", "circle", "diamond", "triangle-up"][: len(g)]),
                                           customdata=g[["acc_ci_low", "acc_ci_high", "n_questions"]].values,
                                           hovertemplate="%{text}<br>accuracy %{y:.2f} (CI %{customdata[0]:.2f}–%{customdata[1]:.2f}), n = %{customdata[2]:.0f}<br>$ за вызов %{x:.5f}<extra></extra>"))
                fig.update_xaxes(type="log", title="$ за вызов (лог. шкала)"); fig.update_yaxes(title="accuracy", range=[0, 1.1])
                st.plotly_chart(style(fig, 400), width="stretch")
                fbs = g[g.strategy == "self_route"].fallback_share.iloc[0]
                st.caption(f"Доля отказов RAG (= доля вызовов LC в Self-Route): {fbs:.0%}. Если отказов почти нет, Self-Route совпадает с чистым RAG.")
                show = g[["Стратегия", "accuracy", "cost_per_call", "usd_per_correct"]].round(5)
                st.dataframe(show, hide_index=True, width="stretch")
                download(sim, "routing_sim.csv", "Скачать всю симуляцию (CSV)")
                st.markdown(f"<div class='warn'>{notes.get('routing', 'Выводы — в REPORT.md.')}</div>", unsafe_allow_html=True)

    # ================================================================================================================ 8
    else:
        st.title("Ограничения и честность")
        st.caption("Что стоит помнить, читая любые цифры этого демо.")
        items = [
            ("Мало вопросов", "На ячейку всего 18 вопросов (повторы одного вопроса — не независимые наблюдения). Парный тест при n = 18 способен подтвердить только очень большую разницу: «не значимо» значит «не хватило чувствительности», а не «различий нет»."),
            ("Синтетические данные", "Корпус — 48 выдуманных регламентов вымышленного университета. Реальные документы KOZ AI устроены иначе; цифры нельзя переносить напрямую."),
            ("Судья проверен на 40 примерах", "Калибровка судьи: метки предзаполнены автоматически и проверены одним человеком, согласие 0.85, κ = 0.752. Это не независимая двойная разметка. Правило для T2 выбрано после просмотра ошибок."),
            ("Кэш между вопросами не сработал", "Цены «холодные». Кривые «с кэшем» — сценарий, не измерение; для LC это оптимистичное допущение."),
            ("Одна модель", "Основные результаты — только gpt-6-luna; Terra проверена на тонком срезе читателя (18 вопросов, 1 повтор), другие модели не запускались."),
            ("Точечные проверки — постфактум", "Бонус-проверки выбраны после того, как провал EMB был виден в таблице. Они отделены от главной таблицы и не меняют заранее записанные гипотезы."),
            ("Метрика цитаты исправлена задним числом", "quote_grounded была 0.1–0.3 из-за строгого сравнения строк; после нормализации формы (кавычки, пункты) — около 0.9. Слова и числа по-прежнему должны совпадать дословно."),
            ("Агент с ограничениями", "У агента reasoning_effort = none (требование API для function tools), Notion — мок; сравнение «агент ≈ LC» нужно читать с этой поправкой."),
        ]
        for i in range(0, len(items), 2):
            cs = st.columns(2)
            for c, (t, txt) in zip(cs, items[i:i + 2]):
                c.markdown(f"<div class='card' style='border-top:4px solid {WARN}'><div style='font-weight:700'>{t}</div><p style='margin-top:.4rem'>{txt}</p></div>", unsafe_allow_html=True)
            st.markdown("")
        st.markdown("Полный список — в разделе «Ограничения» файла REPORT.md.")
except Exception as _e:   # понятное сообщение вместо traceback, если не хватает файла или поля данных
    st.warning(f"Этот экран не удалось построить: не хватает данных или файла ({type(_e).__name__}: {_e}). "
               "Выгрузите данные заново: python -m scripts.export_demo_data.")
