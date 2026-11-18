"""Ragas faithfulness (вторичная проверка) на gpt-6-luna: BM25, набор B, 100K, первые N не-отказных ответов по qid.

Запуск ТОЛЬКО из окружения, где Ragas импортируется (чистый venv; в основном .venv нет langchain_community):
  python -m scripts.ragas_faithfulness --dry            # без API: подменная модель, считает токены промптов Ragas (смета)
  python -m scripts.ragas_faithfulness --live --n 20    # платно, через cost_guard; только после явного «да»

Совместимость с gpt-6-luna: temperature не передаётся (LangchainLLMWrapper(bypass_temperature=True), ChatOpenAI без temperature),
лимит вывода через max_completion_tokens, function tools не используются (Ragas разбирает JSON из текста)."""
from __future__ import annotations

import argparse, json, os, re, statistics as st
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
import tiktoken
enc = tiktoken.get_encoding("o200k_base")


def load_samples(n: int):
    rows = [json.loads(l) for l in open(ROOT / "runs/v2_b_main/raw.jsonl")]
    rows = sorted((r for r in rows if r["method"] == "bm25" and not r["abstained"] and (r["answer"] or "").strip()), key=lambda r: r["qid"])[:n]
    qs = {q["qid"]: q["question"] for q in json.load(open(ROOT / "data/questions.json"))["questions"]}
    out = []
    for r in rows:
        ctx = r["prompt"].split("ДОКУМЕНТЫ:", 1)[1].rsplit("ВОПРОС:", 1)[0].strip()
        out.append(dict(qid=r["qid"], user_input=qs[r["qid"]], response=r["answer"], retrieved_contexts=[ctx]))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); ap.add_argument("--live", action="store_true"); ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from ragas import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness
    import asyncio

    stats = []
    if a.dry:
        class Fake(BaseChatModel):
            @property
            def _llm_type(self): return "fake"
            def _generate(self, messages, stop=None, run_manager=None, **kw):
                text = "\n".join(m.content for m in messages if isinstance(m.content, str))
                if "Break down each sentence" in text:
                    out = json.dumps({"statements": ["Первое утверждение об ответе.", "Второе утверждение об ответе."]}, ensure_ascii=False)
                else:
                    out = json.dumps({"statements": [{"statement": "Первое утверждение об ответе.", "reason": "есть в контексте", "verdict": 1},
                                                      {"statement": "Второе утверждение об ответе.", "reason": "есть в контексте", "verdict": 1}]}, ensure_ascii=False)
                stats.append((len(enc.encode(text)), len(enc.encode(out))))
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=out))])
        llm = LangchainLLMWrapper(Fake(), bypass_temperature=True)
    else:
        assert a.live, "укажи --dry или --live"
        from langchain_openai import ChatOpenAI
        from src import cost_guard
        cost_guard.check()
        llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-6-luna", api_key=os.environ["OPENAI_API_KEY"], max_completion_tokens=800, max_retries=0), bypass_temperature=True)

    metric = Faithfulness(llm=llm)
    samples = load_samples(a.n)
    out_path = ROOT / "results/ragas_faithfulness_bm25_b100k.json"
    done = {}
    if a.live and out_path.exists():            # докатываем: уже оценённые вопросы не повторяем (экономия денег)
        done = {x["qid"]: x["faithfulness"] for x in json.load(open(out_path)) if x["faithfulness"] is not None}

    async def run_all():
        # ОДИН event loop на все вопросы: async-клиент OpenAI внутри Ragas привязан к циклу, на asyncio.run() на вопрос рвалось соединение
        res = []
        for s in samples:
            if s["qid"] in done:
                res.append(dict(qid=s["qid"], faithfulness=done[s["qid"]])); continue
            sc = await metric.single_turn_ascore(SingleTurnSample(user_input=s["user_input"], response=s["response"], retrieved_contexts=s["retrieved_contexts"]))
            res.append(dict(qid=s["qid"], faithfulness=sc))
            if a.live:
                json.dump(res, open(out_path, "w"), ensure_ascii=False, indent=1)   # сохраняем после каждого вопроса
        return res
    res = asyncio.run(run_all())
    scores = [x["faithfulness"] for x in res]
    if a.dry:
        p = [x[0] for x in stats]
        print(f"вопросов {len(samples)}; вызовов LLM {len(stats)} (≈{len(stats)/len(samples):.1f} на вопрос); токенов промпта ср. {st.mean(p):.0f}, макс {max(p)}; всего вход {sum(p)}; выход (фейк) {sum(x[1] for x in stats)}")
        json.dump(dict(calls=len(stats), in_tokens=sum(p), n=len(samples)), open(ROOT / "runs/ragas_dry.json", "w"))
    else:
        json.dump(res, open(out_path, "w"), ensure_ascii=False, indent=1)
        print("faithfulness ср.:", round(st.mean(x for x in scores if x is not None), 3), "n =", len(scores), res[:3])


if __name__ == "__main__":
    main()
