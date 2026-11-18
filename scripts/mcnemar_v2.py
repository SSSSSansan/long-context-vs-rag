"""McNemar exact + Холм для V2 (gpt-6-luna) в двух вариантах метрики, отдельно внутри A и A-hard.
Пишет results/mcnemar_v2_main.csv и results/mcnemar_v2_strict.csv. Без API.   python -m scripts.mcnemar_v2
Один бинарный исход на вопрос (верно, если верна >половины из 3 повторов); пары — один и тот же вопрос в обоих условиях."""
import glob
from pathlib import Path

import pandas as pd

from src.eval import aggregate as ag

ROOT = Path(__file__).resolve().parent.parent
df = ag.load([Path(d) for d in sorted(glob.glob(str(ROOT / "runs/v2_s[1-7]")))])
df = df[df.model.str.startswith("gpt-6-luna")]
pd.set_option("display.width", 220)
for variant, col in (("main", "correct"), ("strict", "correct_strict")):
    d = df.copy(); d["correct"] = d[col]
    t = ag.mcnemar_tables(d)
    t.to_csv(ROOT / f"results/mcnemar_v2_{variant}.csv", index=False)
    m = t[t.family == "method"]
    print(f"\n=== {variant}: пар {len(t)} (метод {len(m)}); p_holm<0.05: {int((t.p_holm < 0.05).sum())}; p<0.05 без поправки: {int((t.p_value < 0.05).sum())}")
    keep = m[(m.p_value < 0.1) & ~m.a.eq("nocontext") & ~m.b.eq("nocontext")]
    print(keep[["qset", "context", "a", "b", "n_pairs", "only_a", "only_b", "acc_a", "acc_b", "p_value", "p_holm"]].round(3).to_string(index=False))
    s = t[(t.family == "size") & (t.p_value < 0.2)]
    print("размеры (p<0.2):\n", s[["qset", "context", "a", "b", "only_a", "only_b", "p_value", "p_holm"]].round(3).to_string(index=False) if len(s) else "  нет")
    nc = m[m.a.eq("nocontext") | m.b.eq("nocontext")]
    print("NO-CONTEXT против остальных: max p_holm =", round(nc.p_holm.max(), 4), "в", len(nc), "парах")
