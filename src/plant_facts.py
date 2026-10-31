"""Этап 2: внедрение контролируемых фактов, gold.json и questions.json.

Принципы (из плана, раздел 3.4):
  * каждый факт про УНИКАЛЬНУЮ вымышленную сущность («программа «Тавро-47»»), которой нет в фоне,
    поэтому модель не может ответить из общих знаний;
  * формулировка в стиле окружающих пунктов; пункт вставляется на случайное место раздела,
    нумерация пунктов в разделе пересчитывается — шва не видно;
  * ответ однозначно проверяется (число с единицей или название подразделения).

Вложенные вопросы: факт, добавленный «на уровне 8k», лежит в первых документах и поэтому
присутствует во всех трёх корпусах; факты уровня 30k лежат в документах между 8K и 30K и т.д.
Размещаем только в документах, которые гарантированно целиком попадают в корпус своего уровня.

Типы:
  T1 — один факт, вопрос на значение;
  T2 — два факта о РАЗНЫХ программах в РАЗНЫХ документах, вопрос на сравнение;
  T3 — два НЕСОВМЕСТИМЫХ утверждения об одной сущности в двух документах, вопрос на поиск противоречия.

Запуск:  python -m src.plant_facts     (после make raw)
Выход:   corpus/built/planted/doc_*.txt, data/gold.json, data/questions.json
"""
from __future__ import annotations

import json
import random
import re
from itertools import product
from pathlib import Path
from typing import Callable

from .config import ROOT, base_cfg, load_yaml
from .tokens import count_tokens

CASES = ("nom", "gen", "dat", "acc", "prep")
KINDS = {
    "программа": ("программа", "программы", "программе", "программу", "программе"),
    "стипендия": ("стипендия", "стипендии", "стипендии", "стипендию", "стипендии"),
    "стажировка": ("стажировка", "стажировки", "стажировке", "стажировку", "стажировке"),
    "премия": ("премия", "премии", "премии", "премию", "премии"),
    "олимпиада": ("олимпиада", "олимпиады", "олимпиаде", "олимпиаду", "олимпиаде"),
    "школа": ("школа", "школы", "школе", "школу", "школе"),
}
NAME_PREFIXES = [
    "Тавро", "Келор", "Мирант", "Ольвия", "Зарнис", "Кварта", "Лумен", "Нерида", "Астрон", "Вельта",
    "Корунд", "Ривена", "Талис", "Эндор", "Фарос", "Гелион", "Ориаль", "Ксенар", "Бельфор", "Дарвен",
    "Эритон", "Салмар", "Тонвис", "Ульрик", "Хардан",
]
OFFICE_KINDS = ["Бюро", "Центр", "Отдел"]
OFFICE_AREAS = [
    "грантовой поддержки", "международных обменов", "молодёжных проектов", "научного наставничества",
    "проектного обучения", "инновационных стартапов", "академического партнёрства",
    "стипендиального обеспечения", "стажировок и практик", "олимпиадного движения",
    "образовательных инициатив", "лабораторных исследований", "студенческой науки",
    "цифровых компетенций", "дополнительного образования", "языковой подготовки",
    "профессиональной ориентации", "экспертной оценки", "мониторинга программ", "корпоративных проектов",
    "академической репутации", "технологического предпринимательства", "кадрового резерва", "летних школ",
]


def good_numbers(lo: int, hi: int) -> list[int]:
    """Числа, после которых единица измерения стоит в одной и той же форме (родительный мн. = именительный):
    5-20, 25-30, ... Исключаем 1-4, 21-24 и т.п., чтобы не склонять в ответах («22 рабочих дня» vs «22 рабочих дней»)."""
    return [n for n in range(lo, hi + 1) if n % 10 in (0, 5, 6, 7, 8, 9) or 11 <= n % 100 <= 14]


# Атрибуты. stmts[0] и stmts[1] — два разных варианта формулировки (для T3 используются разные).
ATTRS: dict[str, dict] = {
    "deadline": dict(
        lo=5, hi=30, unit="рабочих дней",
        stmts=["Заявки на участие в {e_prep} принимаются в течение {v}.",
               "Срок приёма заявок на участие в {e_prep} составляет {v}."],
        q_single="В течение какого срока принимаются заявки на участие в {e_prep}?",
        q_more="Какая из двух программ, «{A}» или «{B}», устанавливает более длительный срок приёма заявок?",
        q_less="Какая из двух программ, «{A}» или «{B}», устанавливает более короткий срок приёма заявок?",
        q_contra="Есть ли в корпусе противоречащие друг другу сведения о сроке приёма заявок на участие в {e_prep}?"),
    "amount": dict(
        lo=10, hi=200, unit="МРП",
        stmts=["Размер ежемесячной выплаты участникам {e_gen} составляет {v}.",
               "Участникам {e_gen} ежемесячно выплачивается {v}."],
        q_single="Каков размер ежемесячной выплаты участникам {e_gen}?",
        q_more="Для какой из двух программ, «{A}» или «{B}», размер ежемесячной выплаты больше?",
        q_less="Для какой из двух программ, «{A}» или «{B}», размер ежемесячной выплаты меньше?",
        q_contra="Есть ли в корпусе противоречащие друг другу сведения о размере ежемесячной выплаты участникам {e_gen}?"),
    "quota": dict(
        lo=5, hi=60, unit="мест",
        stmts=["В рамках {e_gen} предусмотрено {v}.",
               "Квота на участие в {e_prep} установлена в размере {v}."],
        q_single="Сколько мест предусмотрено в рамках {e_gen}?",
        q_more="В какой из двух программ, «{A}» или «{B}», предусмотрено больше мест?",
        q_less="В какой из двух программ, «{A}» или «{B}», предусмотрено меньше мест?",
        q_contra="Есть ли в корпусе противоречащие друг другу сведения о количестве мест в рамках {e_gen}?"),
    "score": dict(
        lo=50, hi=90, unit="баллов",
        stmts=["Для зачисления в {e_acc} необходимо набрать не менее {v}.",
               "Минимальный проходной балл для поступления в {e_acc} составляет {v}."],
        q_single="Какой минимальный балл необходим для зачисления в {e_acc}?",
        q_more="В какую из двух программ, «{A}» или «{B}», для зачисления нужно набрать больше баллов?",
        q_less="В какую из двух программ, «{A}» или «{B}», для зачисления нужно набрать меньше баллов?",
        q_contra="Есть ли в корпусе противоречащие друг другу сведения о минимальном проходном балле для зачисления в {e_acc}?"),
    "duration": dict(
        lo=5, hi=20, unit="недель",
        stmts=["Продолжительность {e_gen} составляет {v}.",
               "Обучение в рамках {e_gen} длится {v}."],
        q_single="Какова продолжительность {e_gen}?",
        q_more="Какая из двух программ, «{A}» или «{B}», имеет большую продолжительность?",
        q_less="Какая из двух программ, «{A}» или «{B}», имеет меньшую продолжительность?",
        q_contra="Есть ли в корпусе противоречащие друг другу сведения о продолжительности {e_gen}?"),
    "office": dict(
        lo=None, hi=None, unit=None,           # значение — название подразделения; в сравнениях не участвует
        stmts=["Заявки по {e_dat} принимает подразделение «{v}».",
               "Приём документов по {e_dat} осуществляет подразделение «{v}»."],
        q_single="Какое подразделение принимает заявки по {e_dat}?",
        q_contra="Есть ли в корпусе противоречащие друг другу сведения о подразделении, принимающем заявки по {e_dat}?"),
}
COMPARABLE = [k for k, v in ATTRS.items() if "q_more" in v]


def ent_forms(kind: str, name: str) -> dict[str, str]:
    return {f"e_{c}": f"{form} «{name}»" for c, form in zip(CASES, KINDS[kind])}


def value_str(attr: str, value) -> str:
    return value if attr == "office" else f"{value} {ATTRS[attr]['unit']}"


def disp(attr: str, value) -> str:
    """Значение для текста ответа: подразделение — в кавычках-ёлочках."""
    return f"«{value}»" if attr == "office" else value_str(attr, value)


def statement(attr: str, variant: int, kind: str, name: str, value) -> str:
    return ATTRS[attr]["stmts"][variant].format(v=value_str(attr, value), **ent_forms(kind, name))


# ---------- размещение ----------

CLAUSE = re.compile(r"^(\d+)\.(\d+)\. (.*)$")


def insert_clause(text: str, section: int, rng: random.Random, clause_text: str) -> str:
    """Вставляет пункт в случайное место раздела и пересчитывает нумерацию пунктов раздела."""
    lines = text.split("\n")
    idxs = [i for i, l in enumerate(lines) if (m := CLAUSE.match(l)) and int(m.group(1)) == section]
    k = rng.randint(0, len(idxs))
    at = idxs[k] if k < len(idxs) else idxs[-1] + 1
    lines.insert(at, f"{section}.0. {clause_text}")
    n = 0
    for i, l in enumerate(lines):
        m = CLAUSE.match(l)
        if m and int(m.group(1)) == section:
            n += 1
            lines[i] = f"{section}.{n}. {m.group(3)}"
    return "\n".join(lines)


def doc_offsets(docs: dict[str, str], count: Callable[[str], int]) -> dict[str, tuple[int, int]]:
    """Токенные границы (начало, конец) каждого документа в общем потоке."""
    out, pos = {}, 0
    for doc_id, text in docs.items():
        n = count(text) + 1
        out[doc_id] = (pos, pos + n)
        pos += n
    return out


def eligible_docs(offsets: dict[str, tuple[int, int]], sizes: dict[str, int], level: str) -> list[str]:
    """Документы, которые гарантированно целиком лежат в корпусе уровня `level`, но не в предыдущем."""
    keys = list(sizes)
    i = keys.index(level)
    start_min = sizes[keys[i - 1]] * 1.025 if i > 0 else 0
    end_max = sizes[level] * (0.875 if i == 0 else 0.9)   # запас: вставки сдвигают границы
    return [d for d, (s, e) in offsets.items() if s >= start_min and e <= end_max]


# ---------- планирование фактов ----------

class NameFactory:
    def __init__(self, rng: random.Random):
        pool = [f"{p}-{n}" for p, n in product(NAME_PREFIXES, range(11, 100))]
        rng.shuffle(pool)
        self._pool = pool

    def new(self) -> str:
        return self._pool.pop()


def _value(rng: random.Random, attr: str, exclude=()) -> object:
    if attr == "office":
        while True:
            v = f"{rng.choice(OFFICE_KINDS)} {rng.choice(OFFICE_AREAS)}"
            if v not in exclude:
                return v
    spec = ATTRS[attr]
    cands = [n for n in good_numbers(spec["lo"], spec["hi"]) if n not in exclude]
    return rng.choice(cands)


def plan(levels_cfg: dict, eligible: dict[str, list[str]], rng: random.Random) -> tuple[list[dict], list[dict]]:
    names = NameFactory(rng)
    facts: list[dict] = []
    questions: list[dict] = []
    c = {"T1": 0, "T2": 0, "T3": 0}

    def fact(**kw) -> dict:
        kw["statement"] = statement(kw["attr"], kw["variant"], kw["kind"], kw["name"], kw["value"])
        facts.append(kw)
        return kw

    for level, counts in levels_cfg.items():
        docs = eligible[level]
        if len(docs) < 2:
            raise ValueError(f"Для уровня {level} недостаточно документов ({len(docs)}): проверь размеры.")
        for _ in range(counts["T1"]):
            c["T1"] += 1
            attr, kind = rng.choice(list(ATTRS)), rng.choice(list(KINDS))
            name, value = names.new(), None
            value = _value(rng, attr)
            fid = f"f_T1_{c['T1']:03d}"
            f = fact(fact_id=fid, type="T1", attr=attr, kind=kind, name=name, value=value, variant=rng.randint(0, 1),
                     doc_id=rng.choice(docs), first_corpus=level, paired_with=None)
            questions.append(dict(
                qid=f"q_T1_{c['T1']:03d}", type="T1", first_corpus=level, fact_ids=[fid],
                question=ATTRS[attr]["q_single"].format(**ent_forms(kind, name)),
                gold_answer=value_str(attr, value)))
        for _ in range(counts["T2"]):
            c["T2"] += 1
            attr = rng.choice(COMPARABLE)
            direction = rng.choice(["more", "less"])
            na, nb = names.new(), names.new()
            va = _value(rng, attr)
            vb = _value(rng, attr, exclude=(va,))
            da, db = rng.sample(docs, 2)
            ida, idb = f"f_T2_{c['T2']:03d}a", f"f_T2_{c['T2']:03d}b"
            fact(fact_id=ida, type="T2", attr=attr, kind="программа", name=na, value=va, variant=rng.randint(0, 1),
                 doc_id=da, first_corpus=level, paired_with=idb)
            fact(fact_id=idb, type="T2", attr=attr, kind="программа", name=nb, value=vb, variant=rng.randint(0, 1),
                 doc_id=db, first_corpus=level, paired_with=ida)
            bigger_is_a = va > vb
            win_is_a = bigger_is_a if direction == "more" else not bigger_is_a
            wname, wv, lname, lv = (na, va, nb, vb) if win_is_a else (nb, vb, na, va)
            questions.append(dict(
                qid=f"q_T2_{c['T2']:03d}", type="T2", first_corpus=level, fact_ids=[ida, idb],
                question=ATTRS[attr][f"q_{direction}"].format(A=na, B=nb),
                gold_answer=f"«{wname}»: {value_str(attr, wv)} против {value_str(attr, lv)} у «{lname}»"))
        for _ in range(counts["T3"]):
            c["T3"] += 1
            attr, kind = rng.choice(list(ATTRS)), rng.choice(list(KINDS))
            name = names.new()
            va = _value(rng, attr)
            vb = _value(rng, attr, exclude=(va,))
            da, db = rng.sample(docs, 2)
            ida, idb = f"f_T3_{c['T3']:03d}a", f"f_T3_{c['T3']:03d}b"
            fact(fact_id=ida, type="T3", attr=attr, kind=kind, name=name, value=va, variant=0,
                 doc_id=da, first_corpus=level, paired_with=idb)
            fact(fact_id=idb, type="T3", attr=attr, kind=kind, name=name, value=vb, variant=1,
                 doc_id=db, first_corpus=level, paired_with=ida)
            questions.append(dict(
                qid=f"q_T3_{c['T3']:03d}", type="T3", first_corpus=level, fact_ids=[ida, idb], contradiction=True,
                question=ATTRS[attr]["q_contra"].format(**ent_forms(kind, name)),
                gold_answer=(f"Да, противоречие: в {da} указано {disp(attr, va)}, "
                             f"а в {db} — {disp(attr, vb)}")))
    return facts, questions


def plant(docs: dict[str, str], facts: list[dict], rng: random.Random) -> dict[str, str]:
    out = dict(docs)
    for f in facts:
        section = rng.randint(2, 6)
        out[f["doc_id"]] = insert_clause(out[f["doc_id"]], section, rng, f["statement"])
    return out


def main(count: Callable[[str], int] = count_tokens) -> None:
    cfg = load_yaml("configs/corpus.yaml")
    seed = base_cfg()["seed"]
    raw_dir = ROOT / cfg["raw_dir"]
    docs = {p.stem: p.read_text(encoding="utf-8") for p in sorted(raw_dir.glob("doc_*.txt"))}
    if not docs:
        raise SystemExit("В corpus/raw нет документов. Сначала: make raw")

    offsets = doc_offsets(docs, count)
    elig = {lvl: eligible_docs(offsets, cfg["sizes_tokens"], lvl) for lvl in cfg["facts"]["levels"]}
    rng = random.Random(f"{seed}-facts")
    facts, questions = plan(cfg["facts"]["levels"], elig, rng)
    planted = plant(docs, facts, rng)

    out_dir = ROOT / cfg["planted_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("doc_*.txt"):
        old.unlink()
    for doc_id, text in planted.items():
        (out_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")

    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    (data / "gold.json").write_text(json.dumps({"facts": facts}, ensure_ascii=False, indent=2), encoding="utf-8")
    (data / "questions.json").write_text(json.dumps({"questions": questions}, ensure_ascii=False, indent=2),
                                         encoding="utf-8")

    print(f"Фактов: {len(facts)}, вопросов: {len(questions)}")
    for lvl, docs_l in elig.items():
        n_f = sum(1 for f in facts if f["first_corpus"] == lvl)
        print(f"  уровень {lvl}: документов для вставки {len(docs_l)} ({docs_l[0]}..{docs_l[-1]}), фактов {n_f}")


if __name__ == "__main__":
    main()
