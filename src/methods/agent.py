"""AGENT (PLAN_V2 §3.4): модель вместо готового {context} получает инструменты и сама ищет фрагменты.

Три варианта по индексу поиска: agent_bm25, agent_emb, agent_hyb (RRF без реранкера, чтобы эпизод не ждал cross-encoder).
Правило сравнения (PLAN §2.3): промпт ответа тот же, что у остальных методов (src/answer.py), меняется только {context}:
вместо документов подставляется заглушка STUB, а инструменты передаются определениями function calling.

Инструменты:
  search_corpus(query, k)               -> JSON [{chunk_id, doc_id, snippet}]  (окно вокруг предложения, лучше всего совпадающего с запросом)
  get_chunk(chunk_id)                   -> полный текст чанка с идентификаторами документа и фрагмента
  compute(expression)                   -> результат арифметики (безопасный разбор через ast, без eval)
  write_to_notion(title, body, sources) -> mock://notion/<файл>.md  (мок без сети, пишет в runs/notion_mock)

Метрики эпизода (PLAN §7):
  n_tool_calls         сколько вызовов инструментов сделала модель (в зачёт идут только выполненные, не больше лимита)
  tool_call_accuracy   доля структурно корректных вызовов по схеме BFCL: существующая функция, обязательные параметры,
                       типы, допустимые значения (chunk_id есть в корпусе, k в 1..K_MAX, выражение разбирается)
  agent_give_up        отказ «ответа нет» или исчерпан лимит вызовов без финального ответа
  evidence_recall      (как у остальных методов) gold-чанк ПРОЧИТАН через get_chunk
  search_recall        gold-чанк хотя бы мелькнул в выдаче search_corpus
Токены и стоимость — сумма по всем шагам цикла (история отправляется заново на каждом шаге).
"""
from __future__ import annotations

import ast
import hashlib
import json
import operator
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import openai

from ..answer import Answer, answer_from_response, build_messages
from ..chunking import Chunk
from ..config import ROOT, base_cfg
from ..llm_client import LLMResponse, ToolResponse, complete_tools
from . import Context, format_chunks

INDEXES = {"agent_bm25": "bm25", "agent_emb": "emb", "agent_hyb": "hyb"}

STUB = ("(документы не загружены заранее. Находи нужные фрагменты инструментами: search_corpus ищет фрагменты, "
        "get_chunk читает фрагмент целиком, compute считает, write_to_notion сохраняет результат. "
        "Отвечай только по найденным фрагментам.)")
NOTION_SUFFIX = " После ответа сохрани результат со ссылками на документы инструментом write_to_notion."

TOOLS = [
    {"type": "function", "function": {
        "name": "search_corpus",
        "description": "Поиск фрагментов корпуса по запросу. Возвращает chunk_id, doc_id и начало фрагмента; для полного текста вызови get_chunk.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "поисковый запрос на русском"},
            "k": {"type": "integer", "description": "сколько фрагментов вернуть, 1..8"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_chunk",
        "description": "Полный текст фрагмента по его chunk_id (например doc_003_c07).",
        "parameters": {"type": "object", "properties": {"chunk_id": {"type": "string"}}, "required": ["chunk_id"]}}},
    {"type": "function", "function": {
        "name": "compute",
        "description": "Арифметика над числами из документов: + - * / // % ** и скобки. Например (20-8)/4.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "write_to_notion",
        "description": "Сохранить результат в Notion: заголовок, текст и список источников (chunk_id или doc_id).",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"}, "body": {"type": "string"},
            "sources": {"type": "array", "items": {"type": "string"}}},
            "required": ["title", "body", "sources"]}}},
]
_JSON_TYPE = {"string": str, "integer": int, "array": list}
SCHEMAS = {t["function"]["name"]: (t["function"]["parameters"]["required"], t["function"]["parameters"]["properties"])
           for t in TOOLS}


def agent_cfg() -> dict:
    return {"max_tool_calls": 8, "search_k_default": 5, "search_k_max": 8, "snippet_chars": 240,
            "notion_dir": "runs/notion_mock", "notion_instruction": False, **base_cfg().get("agent", {})}


# ---------- compute: безопасная арифметика ----------
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UN = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _ev(node):
    if isinstance(node, ast.Expression):
        return _ev(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        a, b = _ev(node.left), _ev(node.right)
        if isinstance(node.op, ast.Pow) and (abs(b) > 100 or abs(a) > 1e6):
            raise ValueError("слишком большая степень")
        return _BIN[type(node.op)](a, b)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
        return _UN[type(node.op)](_ev(node.operand))
    raise ValueError("недопустимое выражение")


def safe_eval(expression: str) -> str:
    if len(expression) > 200:
        raise ValueError("выражение длиннее 200 символов")
    try:
        v = _ev(ast.parse(expression.strip(), mode="eval"))
    except SyntaxError as e:
        raise ValueError("синтаксическая ошибка") from e
    except ZeroDivisionError as e:
        raise ValueError("деление на ноль") from e
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(round(v, 6))


# ---------- Notion: мок ----------
def write_notion_mock(title: str, body: str, sources: list[str], out_dir: Path) -> str:
    """Пишет страницу в .md и возвращает mock-URL. Имя файла — хэш содержимого: повтор не плодит дубликаты."""
    out_dir.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha1(f"{title}\n{body}\n{sources}".encode("utf-8")).hexdigest()[:10] + ".md"
    text = f"# {title}\n\n{body}\n\n## Источники\n" + "\n".join(f"- {s}" for s in sources) + "\n"
    (out_dir / name).write_text(text, encoding="utf-8")
    return f"mock://notion/{name}"


# ---------- поиск ----------
def default_search(index: str, size: str, query: str, k: int) -> list[Chunk]:
    from . import bm25, embedding_rag, hybrid      # тяжёлые импорты только при реальном поиске
    if index == "bm25":
        return list(bm25.retrieve(size, query, k))
    if index == "emb":
        return list(embedding_rag.retrieve(size, query, k))
    if index == "hyb":
        return list(hybrid.retrieve(size, query, k, rerank=False))
    raise ValueError(f"неизвестный индекс агента: {index}")


class Toolbox:
    def __init__(self, size: str, index: str, chunks: dict[str, Chunk], search_fn: Callable | None = None,
                 notion_dir: Path | None = None):
        self.size, self.index, self.chunks = size, index, chunks
        self.doc_ids = {c.doc_id for c in chunks.values()}
        self.search_fn = search_fn or default_search
        self.notion_dir = notion_dir or ROOT / agent_cfg()["notion_dir"]
        self.search_ids: list[str] = []     # chunk_id, показанные в выдаче поиска
        self.opened_ids: list[str] = []     # chunk_id, прочитанные через get_chunk
        self.shown: list[str] = []          # текст, который увидела модель (для проверки цитаты)
        self.notion_url: str | None = None
        self.retrieval_ms = 0.0

    def validate(self, name: str, args) -> tuple[bool, str]:
        """Структурная корректность вызова по схеме BFCL."""
        if name not in SCHEMAS:
            return False, f"неизвестная функция {name!r}"
        if not isinstance(args, dict):
            return False, "аргументы не объект JSON"
        required, props = SCHEMAS[name]
        missing = [p for p in required if p not in args]
        if missing:
            return False, f"нет обязательных параметров: {missing}"
        extra = [p for p in args if p not in props]
        if extra:
            return False, f"лишние параметры: {extra}"
        for p, v in args.items():
            t = _JSON_TYPE[props[p]["type"]]
            if isinstance(v, bool) or not isinstance(v, t):
                return False, f"параметр {p}: ожидается {props[p]['type']}"
        if name == "search_corpus":
            if not args["query"].strip():
                return False, "пустой запрос"
            if "k" in args and not 1 <= args["k"] <= agent_cfg()["search_k_max"]:
                return False, "k вне допустимого диапазона"
        elif name == "get_chunk":
            if args["chunk_id"] not in self.chunks:
                return False, "такого chunk_id нет в корпусе"
        elif name == "compute":
            try:
                safe_eval(args["expression"])
            except ValueError as e:
                return False, f"выражение не вычисляется: {e}"
        elif name == "write_to_notion":
            if not args["title"].strip() or not args["body"].strip():
                return False, "пустой заголовок или текст"
            if not all(isinstance(s, str) and (s in self.chunks or s in self.doc_ids) for s in args["sources"]):
                return False, "источник не chunk_id и не doc_id корпуса"
        return True, "ok"

    def run(self, name: str, args: dict) -> str:
        """Выполняет УЖЕ проверенный вызов."""
        cfg = agent_cfg()
        t0 = time.perf_counter()
        if name == "search_corpus":
            got = self.search_fn(self.index, self.size, args["query"], args.get("k", cfg["search_k_default"]))
            rows = [{"chunk_id": c.chunk_id, "doc_id": c.doc_id,
                     "snippet": make_snippet(c.text, args["query"], cfg["snippet_chars"])} for c in got]
            self.search_ids += [r["chunk_id"] for r in rows]
            out = json.dumps(rows, ensure_ascii=False)
            self.shown.append(out)
        elif name == "get_chunk":
            self.opened_ids.append(args["chunk_id"])
            out = format_chunks([self.chunks[args["chunk_id"]]])
            self.shown.append(out)
        elif name == "compute":
            out = safe_eval(args["expression"])
        else:
            self.notion_url = write_notion_mock(args["title"], args["body"], args["sources"], self.notion_dir)
            out = self.notion_url
        self.retrieval_ms += (time.perf_counter() - t0) * 1000
        return out


def make_snippet(text: str, query: str, n: int) -> str:
    """Окно в n символов, начинающееся с предложения чанка, лучше всего совпадающего с запросом по словам
    (иначе факт, лежащий глубже начала чанка, агенту не виден). Без совпадений - начало чанка."""
    flat = re.sub(r"\s+", " ", text)
    q = {w for w in re.findall(r"\w+", query.lower()) if len(w) > 3}
    best, best_pos, pos = 0, 0, 0
    for sent in re.split(r"(?<=[.!?])\s+", flat):
        sc = len(q & set(re.findall(r"\w+", sent.lower())))
        if sc > best:
            best, best_pos = sc, pos
        pos += len(sent) + 1
    return flat[best_pos: best_pos + n]


@dataclass
class Episode:
    context: Context        # text: всё, что модель увидела из корпуса; chunk_ids: прочитанные get_chunk
    answer: Answer
    extra: dict = field(default_factory=dict)


def run_episode(size: str, question: str, *, salt: str = "", index: str = "bm25",
                chunks: dict[str, Chunk] | None = None, search_fn: Callable | None = None,
                llm: Callable[..., ToolResponse] | None = None, notion_dir: Path | None = None,
                max_calls: int | None = None) -> Episode:
    cfg = agent_cfg()
    max_calls = cfg["max_tool_calls"] if max_calls is None else max_calls
    if chunks is None:
        from .bm25 import chunks_of
        chunks = {c.chunk_id: c for c in chunks_of(size)}
    tb = Toolbox(size, index, chunks, search_fn, notion_dir)
    llm = llm or complete_tools

    messages = build_messages(STUB + (NOTION_SUFFIX if cfg["notion_instruction"] else ""), question)
    first_prompt = messages[0]["content"]
    log: list[dict] = []
    tot = dict(p=0, c=0, cached=0, ms=0.0, cost=0.0)
    all_cached, last, final, steps, api_error = True, None, None, 0, None

    for _ in range(max_calls + 3):           # страховка: лимит вызовов + принудительный финал + запас
        force = len(log) >= max_calls
        try:
            tr = llm(messages, TOOLS, tool_choice="none" if force else "auto", salt=f"{salt}:s{steps}")
        except openai.APIStatusError as e:    # напр. Ollama 500 «token repeat limit reached»: эпизод = неудача, прогон идёт дальше
            if e.status_code < 500:
                raise
            api_error = f"{e.status_code}: {e.message}"
            steps += 1
            break
        steps += 1
        r = tr.resp
        last = r
        tot["p"] += r.prompt_tokens; tot["c"] += r.completion_tokens; tot["cached"] += r.cached_tokens
        tot["ms"] += r.latency_ms; tot["cost"] += r.cost_usd
        all_cached = all_cached and r.from_local_cache
        if not tr.tool_calls:
            final = r
            break
        messages.append({"role": "assistant", "content": r.text or None, "tool_calls": [
            {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
            for tc in tr.tool_calls]})
        for tc in tr.tool_calls:
            if len(log) >= max_calls:         # сверх лимита не выполняем и не засчитываем, но API требует ответ на каждый вызов
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": "лимит вызовов инструментов исчерпан: дай финальный ответ"})
                continue
            try:
                args = json.loads(tc["arguments"] or "{}")
            except json.JSONDecodeError:
                args = None
            ok, why = tb.validate(tc["name"], args) if args is not None else (False, "аргументы не JSON")
            out = tb.run(tc["name"], args) if ok else f"ошибка: {why}"
            log.append({"name": tc["name"], "args": args if args is not None else tc["arguments"], "valid": ok, "reason": why})
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": out})

    hit_cap = final is None
    agg = LLMResponse(
        text=final.text if final else "", prompt_tokens=tot["p"], completion_tokens=tot["c"], cached_tokens=tot["cached"],
        latency_ms=tot["ms"], cost_usd=tot["cost"], model=last.model if last else "", backend=last.backend if last else "",
        from_local_cache=all_cached, model_api=last.model_api if last else None)
    ans = answer_from_response(agg)
    n_valid = sum(c["valid"] for c in log)
    extra = dict(
        prompt=first_prompt, agent_index=index, n_steps=steps, n_tool_calls=len(log),
        tool_call_accuracy=(n_valid / len(log)) if log else None,
        tool_calls=log, hit_call_cap=hit_cap and not api_error, api_error=api_error, agent_give_up=bool(hit_cap or ans.abstained),
        search_chunk_ids=list(dict.fromkeys(tb.search_ids)), notion_url=tb.notion_url,
        agent_trace=messages[1:])
    ctx = Context(text="\n\n".join(tb.shown), chunk_ids=list(dict.fromkeys(tb.opened_ids)), retrieval_ms=tb.retrieval_ms)
    return Episode(ctx, ans, extra)
