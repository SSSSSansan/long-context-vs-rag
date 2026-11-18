"""Микро-проверка: что возвращает API в usage (поля про запись в кэш) на gpt-6-luna. Два одинаковых вызова ~1500 токенов.
python -m scripts.probe_usage   (~$0.0003, пишет в cost_guard)"""
import json, os, random, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
from openai import OpenAI
from src import cost_guard
from src.config import base_cfg
from src.llm_client import compute_cost

cost_guard.check()
cfg = base_cfg()["backends"]["openai"]
c = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0)
rnd = random.Random(20261118)
words = "альфа бета гамма дельта сигма омега кварк лептон мезон фотон".split()
prompt = " ".join(rnd.choice(words) + str(rnd.randint(100, 999)) for _ in range(900)) + "\n\nОтветь одним словом: ок."
tot = 0.0
for i in (1, 2):
    r = c.chat.completions.create(model=cfg["chat_model"], messages=[{"role": "user", "content": prompt}], max_completion_tokens=200)
    u = r.usage
    print(f"--- вызов {i}: model={r.model}")
    print(json.dumps(u.model_dump(), ensure_ascii=False))
    print("model_extra:", getattr(u, "model_extra", None), "| ответ:", repr(r.choices[0].message.content))
    d = u.prompt_tokens_details
    cached = (d.cached_tokens if d else 0) or 0
    cost = compute_cost(u.prompt_tokens, cached, u.completion_tokens, cfg["price_per_1m_usd"]); tot += cost
    print("наша формула, $:", round(cost, 7))
    time.sleep(4)
cost_guard.add(tot)
print("всего $", round(tot, 7))
