"""Проверка без обращений к API: задачи пилота (runs/v2_pilot_b) должны целиком отвечаться из дискового кэша.
Клиенты OpenAI (чат и эмбеддинги) заблокированы: любое обращение к сети = AssertionError. Бюджет до/после сравнивается."""
import json, os, sys
os.environ["LLM_BACKEND"] = "openai"
from pathlib import Path
import yaml
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
from src import llm_client
from src.config import base_cfg, ROOT
from src.eval import run as R

def blocked(*a, **k):
    raise AssertionError("ОБРАЩЕНИЕ К API (кэш не сработал)")
llm_client._get_client = blocked
llm_client._get_openai_embed_client = blocked

pilot = {(r["qid"], r["method"]) for r in map(json.loads, open(ROOT / "runs/v2_pilot_b/raw.jsonl"))}
spec = yaml.safe_load(open(ROOT / "configs/experiments/v2_b_main.yaml"))
gold, qs = R.load_data()
tasks = [t for t in R.expand_tasks(spec, qs, base_cfg()["seed"]) if (t["q"]["qid"], t["method"]) in pilot]
b0 = json.load(open(ROOT / "runs/budget_v2.json"))["spent_usd"]
print("задач пилота найдено в новом конфиге:", len(tasks), "из", len(pilot))
ok = 0
for t in tasks:
    row = R.execute(t, gold)
    ok += 1
    print(f"  {t['method']:4s} {t['q']['qid']}  from_cache={row['from_cache']}  cost=${row['cost_usd']:.5f}  verdict={row['verdict']}")
b1 = json.load(open(ROOT / "runs/budget_v2.json"))["spent_usd"]
print(f"выполнено без сети: {ok}/{len(pilot)}; бюджет до {b0} после {b1} (изменение {b1-b0:+.6f})")
