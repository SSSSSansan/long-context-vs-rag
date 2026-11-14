"""Шаг 1.3: набор A-hard — те же факты набора A, но вопрос БЕЗ имени сущности, плюс дистракторы.

Как сущность опознаётся без имени. Рядом с каждым фактом набора A (в том же чанке, следующей строкой) стоит
строка-СВЯЗКА с тремя признаками: год учреждения + аудитория + курирующее подразделение. Вопрос A-hard называет
эти три признака вместо имени: «...олимпиады, учреждённой в 2017 году для магистрантов и курируемой
подразделением «Центр научной деятельности»?». Подходит ровно одна сущность. Подразделение берётся из частых в
фоновом корпусе (оно встречается десятки раз), поэтому само по себе ничего не опознаёт: опознаёт сочетание.

Связки нужны каждой сущности, названной в вопросе: T1 — одна, T2 — две (две программы), T3 — по связке у каждого
из двух противоречащих утверждений (они лежат в разных документах; без связки второе не найти без имени).
То есть по одной связке на каждый факт набора A.

Дистракторы: для каждой сущности вопроса `distractors_per_entity` «двойников» с той же формулировкой факта, другим
именем и другим значением; их связка совпадает с целевой в двух признаках из трёх и отличается в одном
(так лексически вопрос похож на дистрактор почти так же, как на настоящий факт).

Всё детерминировано (seed), без LLM. Записи связок/дистракторов лежат в gold.json с полем role:
  gold (по умолчанию) | link | distractor_fact | distractor_link
"""
from __future__ import annotations

import random
import re
from collections import Counter
from itertools import product

from .plant_facts import ATTRS, CASES, KINDS, NameFactory, _value, statement, value_str

AUDIENCES = ["магистрантов", "бакалавров", "аспирантов", "докторантов",
             "выпускников", "первокурсников", "преподавателей", "сотрудников"]
YEARS = list(range(2014, 2025))
FEATURES = ("year", "audience", "office")
UNIT_RE = re.compile(r"«((?:Центр|Бюро|Отдел|Служба|Управление) [^»]+)»")
# согласование причастий с падежом названия сущности: «олимпиады, учреждённой...», «олимпиаду, учреждённую...»
PARTICIPLE = {"nom": "ая", "gen": "ой", "dat": "ой", "acc": "ую", "prep": "ой"}
PAIR_MARK = ", «{A}» или «{B}»,"          # так во всех q_more/q_less (проверяется тестом)


def frequent_offices(texts: list[str], n: int) -> list[str]:
    """Самые частые названия подразделений в фоновых документах (детерминированно: частота, затем алфавит)."""
    counts = Counter(m for t in texts for m in UNIT_RE.findall(t))
    return [o for o, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


def link_statement(kind: str, name: str, feats: dict, style: str = "full", variant: int = 0) -> str:
    """Строка-связка. variant (0/1) нужен T3: два утверждения об одной сущности лежат в разных документах,
    и их связки должны различаться текстом (иначе пункт встретится в корпусе дважды)."""
    y, a, o = feats["year"], feats["audience"], feats["office"]
    k = KINDS[kind][0]
    if style == "compact":
        return (f"«{name}» ({k}): {y} г., для {a}, подразделение «{o}»." if variant == 0
                else f"{k.capitalize()} «{name}»: учреждена в {y} г., для {a}, курирует «{o}».")
    if variant == 0:
        return f"{k.capitalize()} «{name}» учреждена в {y} году для {a} и курируется подразделением «{o}»."
    return f"Подразделение «{o}» курирует {KINDS[kind][3]} «{name}», учреждённую в {y} году для {a}."


def describe(feats: dict, case: str | None = None, kind: str | None = None) -> str:
    """Описание сущности по трём признакам. С kind и case — «олимпиады, учреждённой ...»; без них — «учреждённая ...»."""
    sfx = PARTICIPLE[case or "nom"]
    core = (f"учреждённ{sfx} в {feats['year']} году для {feats['audience']} "
            f"и курируем{sfx} подразделением «{feats['office']}»")
    return f"{KINDS[kind][CASES.index(case)]}, {core}" if kind and case else core


def entities_matching(link_records: list[dict], kind: str, feats: dict) -> set[str]:
    """Имена сущностей, чьи связки подходят под ВСЕ признаки вопроса (вид + год + аудитория + подразделение)."""
    return {r["name"] for r in link_records
            if r["kind"] == kind and all(r[k] == feats[k] for k in FEATURES)}


def _hard_question(q: dict, ents: list[dict], by_id: dict, link_ids: list[str], dist_ids: list[str]) -> dict:
    f0 = by_id[q["fact_ids"][0]]
    attr, kind = f0["attr"], f0["kind"]
    spec = ATTRS[attr]
    if q["type"] == "T2":
        direction = next(d for d in ("more", "less")
                         if q["question"] == spec[f"q_{d}"].format(A=ents[0]["name"], B=ents[1]["name"]))
        tpl = spec[f"q_{direction}"]
        assert PAIR_MARK in tpl, f"шаблон {attr}/{direction} не содержит {PAIR_MARK!r}"
        text = tpl.replace(PAIR_MARK, " — {A} или {B} —").format(A=describe(ents[0]), B=describe(ents[1]))
        # имена в эталоне оставляем: судье нужен конкретный ответ; описание добавляем, чтобы ответ описанием тоже засчитывался
        head, tail = q["gold_answer"].split(": ", 1)
        winner = ents[0] if head.startswith(f"«{ents[0]['name']}»") else ents[1]
        gold = f"{head} ({describe(winner)}): {tail}"
    else:
        forms = {f"e_{c}": describe(ents[0], c, kind) for c in CASES}
        text = spec["q_single" if q["type"] == "T1" else "q_contra"].format(**forms)
        gold = q["gold_answer"]
    return dict(
        qid=f"{q['qid']}h", parent_qid=q["qid"], type=q["type"], first_corpus=q["first_corpus"], set="A", anchor=False,
        fact_ids=list(q["fact_ids"]), link_fact_ids=link_ids, distractor_fact_ids=dist_ids,
        features=[dict(kind=kind, **{k: e[k] for k in FEATURES}) for e in ents],
        question=text, gold_answer=gold, **({"contradiction": True} if q["type"] == "T3" else {}))


def plan_hard(facts: list[dict], questions: list[dict], doc_ids: list[str], offices: list[str],
              rng: random.Random, style: str = "full", distractors_per_entity: int = 1):
    """-> (новые записи для gold.json, вопросы A-hard). Сами `facts`/`questions` не меняет."""
    by_id = {f["fact_id"]: f for f in facts}
    core = [f for f in facts if f.get("set") == "A"]
    used_names = {f["name"] for f in facts}
    names = NameFactory(rng)
    combos = list(product(YEARS, AUDIENCES, offices))
    rng.shuffle(combos)
    taken: set[tuple] = set()

    feats_of: dict[str, dict] = {}             # имя сущности -> признаки (T3: одна сущность на два утверждения)
    for f in core:
        if f["name"] in feats_of:
            continue
        for y, a, o in combos:
            if (y, a, o) in taken:
                continue
            if f["attr"] == "office" and o == f["value"]:       # подразделение-признак не должно совпадать с ответом
                continue
            taken.add((y, a, o))
            feats_of[f["name"]] = dict(year=y, audience=a, office=o)
            break
        else:
            raise ValueError("Не хватило уникальных сочетаний год/аудитория/подразделение.")

    records: list[dict] = []
    for f in core:                                                # связка на каждый факт набора A
        records.append(dict(
            fact_id=f"l_{f['fact_id']}", role="link", of=f["fact_id"], type=f["type"], attr="link", kind=f["kind"],
            name=f["name"], doc_id=f["doc_id"], first_corpus=f["first_corpus"], set="A", **feats_of[f["name"]],
            statement=link_statement(f["kind"], f["name"], feats_of[f["name"]], style, f["variant"])))

    hard_qs: list[dict] = []
    for q in (x for x in questions if x.get("set") == "A" and x.get("anchor", True)):
        qf = [by_id[i] for i in q["fact_ids"]]
        ent_names = list(dict.fromkeys(f["name"] for f in qf))    # T1/T3 -> 1, T2 -> 2
        link_ids = [f"l_{f['fact_id']}" for f in qf]
        dist_ids: list[str] = []
        for ei, ename in enumerate(ent_names):
            ef = next(f for f in qf if f["name"] == ename)
            tf = feats_of[ename]
            for j in range(distractors_per_entity):
                name = names.new()
                while name in used_names:
                    name = names.new()
                used_names.add(name)
                value = _value(rng, ef["attr"], exclude=tuple(f["value"] for f in qf))
                alter = FEATURES[(ei + j) % 3]                    # отличается ровно один признак из трёх
                pool = {"year": YEARS, "audience": AUDIENCES, "office": offices}[alter]
                cands = [v for v in pool if v != tf[alter] and
                         tuple((v if k == alter else tf[k]) for k in FEATURES) not in taken
                         and not (alter == "office" and ef["attr"] == "office" and v == value)]
                if not cands:
                    raise ValueError("Не нашлось свободного признака для дистрактора.")
                df = dict(tf, **{alter: rng.choice(cands)})
                taken.add(tuple(df[k] for k in FEATURES))
                did = f"d_{q['qid']}_{ei}{j}"
                dist_ids += [did, f"dl_{q['qid']}_{ei}{j}"]
                doc = rng.choice(doc_ids)
                records.append(dict(
                    fact_id=did, role="distractor_fact", for_qid=q["qid"], type=q["type"], attr=ef["attr"],
                    kind=ef["kind"], name=name, value=value, variant=ef["variant"], doc_id=doc,
                    first_corpus=ef["first_corpus"], set="A",
                    statement=statement(ef["attr"], ef["variant"], ef["kind"], name, value)))
                records.append(dict(
                    fact_id=f"dl_{q['qid']}_{ei}{j}", role="distractor_link", of=did, for_qid=q["qid"], type=q["type"],
                    attr="link", kind=ef["kind"], name=name, doc_id=doc, first_corpus=ef["first_corpus"], set="A",
                    **df, altered=alter, statement=link_statement(ef["kind"], name, df, style, ef["variant"])))
        ents = [dict(feats_of[n]) | {"name": n} for n in ent_names]
        hard_qs.append(_hard_question(q, ents, by_id, link_ids, dist_ids))
    return records, hard_qs
