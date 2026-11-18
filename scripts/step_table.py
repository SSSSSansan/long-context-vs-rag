"""Быстрая сводка по шагам V2: python -m scripts.step_table runs/v2_s2 [runs/v2_s1 ...]
accuracy_judge = вердикт судьи (отказ -> 0); accuracy = судья И value_in_answer. Стоимость = ответ + судья."""
import json, sys
import pandas as pd
from src.eval.value_check import value_in_answer

rows = [json.loads(l) for d in sys.argv[1:] for l in open(f"{d}/raw.jsonl")]
df = pd.DataFrame(rows)
df["judge"] = ((df.verdict == "correct") & ~df.abstained).astype(float)
df["value"] = [value_in_answer(g, a or "", q or "") for g, a, q in zip(df.gold_answer, df.answer, df.quote)]
df["acc"] = df.judge * df.value.astype(float)
df["cost"] = df.cost_usd + df.judge_cost_usd.fillna(0)
df["m"] = df.method
pd.set_option("display.width", 200)
def show(by):
    g = df.groupby(by).agg(n=("acc", "size"), acc_judge=("judge", "mean"), acc=("acc", "mean"), abstain=("abstained", "mean"),
                           recall=("evidence_recall", "mean"), cost=("cost", "sum"), cost_per_call=("cost", "mean"))
    print(g.round(3).to_string(), "\n")
show(["question_set", "size", "m"])
show(["question_set", "m", "type"])
print("ИТОГО стоимость, $:", round(df.cost.sum(), 4), " вызовов:", len(df), " cached_share:", round(df.cached_tokens.sum() / df.prompt_tokens.sum(), 3))
