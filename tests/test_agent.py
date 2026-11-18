"""AGENT: инструменты, валидация вызовов (BFCL), цикл с лимитом, учёт токенов, complete_tools (всё на моках, без сети)."""
import json
from types import SimpleNamespace

import pytest

from src import llm_client
from src.chunking import Chunk
from src.llm_client import LLMResponse, ToolResponse
from src.methods import agent


def _chunk(doc, i, text):
    return Chunk(chunk_id=f"{doc}_c{i:02d}", doc_id=doc, idx=i, text=text, n_tokens=20)


CHUNKS = {c.chunk_id: c for c in [
    _chunk("doc_001", 0, "1.1. Срок рассмотрения заявления составляет 14 рабочих дней. Подробности ниже."),
    _chunk("doc_001", 1, "2.1. Заявка подаётся через канцелярию."),
    _chunk("doc_002", 0, "1.1. Срок рассмотрения заявления составляет 7 рабочих дней."),
]}


def _resp(text="", p=100, c=10, cached=0, cost=0.001, from_cache=False):
    return LLMResponse(text=text, prompt_tokens=p, completion_tokens=c, cached_tokens=cached, latency_ms=5.0,
                       cost_usd=cost, model="m", backend="local", from_local_cache=from_cache)


def _call(name, args, cid="x1"):
    return {"id": cid, "name": name, "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)}


class ScriptedLLM:
    """Отдаёт заранее заданные шаги; запоминает tool_choice каждого вызова."""
    def __init__(self, steps):
        self.steps, self.choices, self.n = steps, [], 0

    def __call__(self, messages, tools, *, tool_choice="auto", max_tokens=None, salt=""):
        self.choices.append(tool_choice)
        step = self.steps[min(self.n, len(self.steps) - 1)]
        self.n += 1
        return step


def fake_search(index, size, query, k):
    return [CHUNKS["doc_001_c00"], CHUNKS["doc_002_c00"]][:k]


FINAL = '{"answer": "14 и 7 рабочих дней", "doc_ids": ["doc_001", "doc_002"], "quote": "14 рабочих дней", "contradiction_found": true}'


def _episode(steps, tmp_path, **kw):
    llm = ScriptedLLM(steps)
    ep = agent.run_episode("8k", "Срок рассмотрения?", salt="r1", chunks=CHUNKS, search_fn=fake_search, llm=llm,
                           notion_dir=tmp_path, **kw)
    return ep, llm


def test_safe_eval():
    assert agent.safe_eval("2+3*4") == "14"
    assert agent.safe_eval("(20-8)/4") == "3"
    assert agent.safe_eval("7/2") == "3.5"
    for bad in ["__import__('os').system('ls')", "2**1000", "1/0", "abc", "(", "x+1", "True+1"]:
        with pytest.raises(ValueError):
            agent.safe_eval(bad)


def test_notion_mock_is_deterministic(tmp_path):
    u1 = agent.write_notion_mock("Т", "тело", ["doc_001"], tmp_path)
    u2 = agent.write_notion_mock("Т", "тело", ["doc_001"], tmp_path)
    assert u1 == u2 and u1.startswith("mock://notion/")
    f = tmp_path / u1.split("/")[-1]
    assert "# Т" in f.read_text(encoding="utf-8") and "- doc_001" in f.read_text(encoding="utf-8")
    assert len(list(tmp_path.iterdir())) == 1


def test_validate_bfcl_rules(tmp_path):
    tb = agent.Toolbox("8k", "bm25", CHUNKS, fake_search, tmp_path)
    ok = lambda n, a: tb.validate(n, a)[0]
    assert ok("search_corpus", {"query": "срок"}) and ok("search_corpus", {"query": "срок", "k": 3})
    assert not ok("search_corpus", {"k": 3})                          # нет обязательного
    assert not ok("search_corpus", {"query": "срок", "k": 99})        # значение вне диапазона
    assert not ok("search_corpus", {"query": "срок", "k": "3"})       # тип
    assert not ok("search_corpus", {"query": "срок", "k": True})      # bool не int
    assert not ok("search_corpus", {"query": "  "})                   # пустой запрос
    assert not ok("search_corpus", {"query": "срок", "limit": 3})     # лишний параметр
    assert not ok("web_search", {"query": "срок"})                    # выдуманная функция
    assert not ok("get_chunk", {"chunk_id": "doc_999_c00"})           # нет в корпусе
    assert ok("get_chunk", {"chunk_id": "doc_001_c00"})
    assert ok("compute", {"expression": "14-7"}) and not ok("compute", {"expression": "14-"})
    assert not ok("write_to_notion", {"title": "т", "body": "b", "sources": ["doc_zzz"]})
    assert ok("write_to_notion", {"title": "т", "body": "b", "sources": ["doc_001", "doc_002_c00"]})
    assert not tb.validate("get_chunk", None)[0] and not tb.validate("get_chunk", "строка")[0]


def test_full_episode_metrics(tmp_path):
    steps = [
        ToolResponse(_resp(p=100, cost=0.001), [_call("search_corpus", {"query": "срок", "k": 2}, "a")]),
        ToolResponse(_resp(p=200, cost=0.002), [_call("get_chunk", {"chunk_id": "doc_001_c00"}, "b"),
                                                _call("get_chunk", {"chunk_id": "doc_002_c00"}, "c")]),
        ToolResponse(_resp(p=300, cost=0.003), [_call("compute", {"expression": "14-7"}, "d")]),
        ToolResponse(_resp(FINAL, p=400, c=50, cached=100, cost=0.004), []),
    ]
    ep, llm = _episode(steps, tmp_path)
    assert ep.extra["n_tool_calls"] == 4 and ep.extra["tool_call_accuracy"] == 1.0
    assert ep.extra["n_steps"] == 4 and llm.choices == ["auto"] * 4
    assert not ep.extra["hit_call_cap"] and not ep.extra["agent_give_up"]
    assert ep.answer.parsed_ok and ep.answer.doc_ids == ["doc_001", "doc_002"] and ep.answer.contradiction_found
    r = ep.answer.resp                                                # суммы по шагам
    assert (r.prompt_tokens, r.completion_tokens, r.cached_tokens) == (1000, 10 + 10 + 10 + 50, 100)
    assert r.cost_usd == pytest.approx(0.01) and r.latency_ms == pytest.approx(20.0)
    assert ep.context.chunk_ids == ["doc_001_c00", "doc_002_c00"]     # evidence = прочитанные
    assert ep.extra["search_chunk_ids"] == ["doc_001_c00", "doc_002_c00"]
    assert "14 рабочих дней" in ep.context.text and "7 рабочих дней" in ep.context.text
    assert ep.extra["prompt"].rstrip().endswith("ВОПРОС: Срок рассмотрения?") and "ДОКУМЕНТЫ:\n(документы не загружены" in ep.extra["prompt"]


def test_invalid_calls_lower_accuracy_and_do_not_run(tmp_path):
    steps = [
        ToolResponse(_resp(), [_call("web_search", {"query": "q"}, "a"), _call("get_chunk", {"chunk_id": "nope"}, "b"),
                               _call("search_corpus", "{не json", "c"), _call("search_corpus", {"query": "срок"}, "d")]),
        ToolResponse(_resp(FINAL), []),
    ]
    ep, _ = _episode(steps, tmp_path)
    assert ep.extra["n_tool_calls"] == 4 and ep.extra["tool_call_accuracy"] == 0.25
    assert [c["valid"] for c in ep.extra["tool_calls"]] == [False, False, False, True]
    assert ep.context.chunk_ids == []                                 # неверные вызовы ничего не открыли


def test_no_calls_gives_none_accuracy(tmp_path):
    ep, _ = _episode([ToolResponse(_resp(FINAL), [])], tmp_path)
    assert ep.extra["n_tool_calls"] == 0 and ep.extra["tool_call_accuracy"] is None


def test_call_cap_forces_final_answer(tmp_path):
    loop = ToolResponse(_resp(), [_call("search_corpus", {"query": "срок"})])
    final = ToolResponse(_resp(FINAL), [])
    steps = [loop, loop, loop, final]                                 # лимит 2: после двух вызовов tool_choice=none
    ep, llm = _episode(steps, tmp_path, max_calls=2)
    assert ep.extra["n_tool_calls"] == 2 and llm.choices[:3] == ["auto", "auto", "none"]
    assert ep.answer.parsed_ok and not ep.extra["hit_call_cap"]


def test_model_that_never_stops_is_cut_off(tmp_path):
    loop = ToolResponse(_resp(), [_call("search_corpus", {"query": "срок"}), _call("search_corpus", {"query": "ещё"})])
    ep, llm = _episode([loop], tmp_path, max_calls=2)                 # игнорирует tool_choice=none
    assert llm.n <= 5 and ep.extra["hit_call_cap"] and ep.extra["agent_give_up"]
    assert ep.extra["n_tool_calls"] == 2 and ep.answer.text == "" and not ep.answer.parsed_ok


def test_abstention_counts_as_give_up(tmp_path):
    ab = '{"answer": "В предоставленных документах ответа нет.", "doc_ids": [], "quote": "", "contradiction_found": false}'
    ep, _ = _episode([ToolResponse(_resp(ab), [])], tmp_path)
    assert ep.answer.abstained and ep.extra["agent_give_up"]


def test_notion_call_returns_mock_url(tmp_path):
    steps = [ToolResponse(_resp(), [_call("write_to_notion", {"title": "Итог", "body": "14 и 7", "sources": ["doc_001"]})]),
             ToolResponse(_resp(FINAL), [])]
    ep, _ = _episode(steps, tmp_path)
    assert ep.extra["notion_url"].startswith("mock://notion/") and len(list(tmp_path.iterdir())) == 1


# ---------- complete_tools ----------
class FakeClient:
    def __init__(self):
        self.calls = 0
        tc = SimpleNamespace(id="t1", function=SimpleNamespace(name="get_chunk", arguments='{"chunk_id": "doc_001_c00"}'))
        msg = SimpleNamespace(content=None, tool_calls=[tc])
        usage = SimpleNamespace(prompt_tokens=120, completion_tokens=15, prompt_tokens_details=SimpleNamespace(cached_tokens=20))
        self.r = SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage, model="m-2026")
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls += 1
        assert kw["tools"] and kw["tool_choice"] == "auto"
        return self.r


def test_complete_tools_parses_calls_and_caches(monkeypatch):
    store = {}
    monkeypatch.setenv("LLM_BACKEND", "local")
    monkeypatch.setattr(llm_client.cache, "get", lambda k: store.get(k))
    monkeypatch.setattr(llm_client.cache, "put", lambda k, v: store.__setitem__(k, v))
    fc = FakeClient()
    monkeypatch.setattr(llm_client, "_get_client", lambda: fc)
    msgs = [{"role": "user", "content": "привет"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "z", "type": "function", "function": {"name": "compute", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "z", "content": "3"}]
    a = llm_client.complete_tools(msgs, agent.TOOLS, salt="r1:s0")      # content=None в assistant не ломает оценку токенов
    assert a.tool_calls == [{"id": "t1", "name": "get_chunk", "arguments": '{"chunk_id": "doc_001_c00"}'}]
    assert (a.resp.prompt_tokens, a.resp.cached_tokens, a.resp.model_api) == (120, 20, "m-2026") and a.resp.text == ""
    b = llm_client.complete_tools(msgs, agent.TOOLS, salt="r1:s0")      # второй раз — из кэша
    assert fc.calls == 1 and b.resp.from_local_cache and b.resp.cost_usd == 0.0 and b.tool_calls == a.tool_calls
    llm_client.complete_tools(msgs, agent.TOOLS, salt="r2:s0")          # другой повтор — новый вызов
    assert fc.calls == 2


def test_api_5xx_ends_episode_as_give_up():
    import httpx, openai
    from src.methods.agent import run_episode

    def boom(*a, **k):
        req = httpx.Request("POST", "http://x")
        raise openai.InternalServerError("repeat limit", response=httpx.Response(500, request=req), body=None)

    ep = run_episode("8k", "q?", chunks={}, search_fn=lambda *a, **k: [], llm=boom)
    assert ep.extra["agent_give_up"] and ep.extra["api_error"].startswith("500")


def test_abstained_never_correct():
    import pandas as pd
    from src.eval import aggregate  # noqa: F401
    ab = pd.Series([True, False]); v = pd.Series(["correct", "correct"])
    assert (((v == "correct") & ~ab).astype(float)).tolist() == [0.0, 1.0]


def test_snippet_centers_on_matching_sentence():
    from src.methods.agent import make_snippet
    t = "Вводное предложение ни о чём. " * 20 + "Размер выплаты участникам олимпиады составляет 114 МРП. Конец."
    assert "114 МРП" in make_snippet(t, "размер выплаты олимпиады", 100)
    assert make_snippet(t, "zzz", 30) == t[:30]
