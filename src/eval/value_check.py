"""value_in_answer: детерминированная проверка, что в ответе есть значение из эталона (без LLM).

Зачем: судья оценивает «тот же факт», и ответ с одним верным именем сущности без числа может получить correct
(вопросы «что из двух» T2/T3 угадываются с шансом 50%). Итоговая accuracy = судья == correct И value_in_answer.

Правила (gold_answer -> что обязано быть в ответе):
  1. Числа из эталона (после выброса «названий в ёлочках», doc_NNN и описаний в скобках): ВСЕ должны встретиться в
     ответе. «17 недель» = «17» = «17 нед.»; запятая/точка в дробях равны; пробелы нормализуются.
     Так для T2 («75 баллов против 66»), T3 с числами и T1 с числом («114 МРП»).
  2. Нет чисел, есть названия в «ёлочках» (T3 про подразделения): ВСЕ названия должны встретиться в ответе.
  3. Иначе (T1 про подразделение): весь эталон как строка должен встретиться в ответе.
Сравнение без регистра, ё = е, кавычки и лишние пробелы не важны.
  Правило 1 для вопросов «что из двух» (T2: в эталоне « против », структура «победитель: X против Y у ...») —
  основное правило winner-value-only (введено 18.11, ПОСТФАКТУМ по разбору калибровки): достаточно числа ПОБЕДИТЕЛЯ
  (первое число эталона). Строгое «оба числа» осталось как диагностика: value_in_answer_strict. Имя победителя без
  числа по-прежнему НЕ засчитывается (защита от угадывания 50%). T1 и T3 не менялись.
Проверяемый текст = answer + quote — ровно то, что видит судья (run.pred_text). Только `answer` дал бы формальный
брак: на V1-данных модели кладут значения в цитату, а в T3 ответ часто просто «true» (проверено на runs/ и v1/).
"""
from __future__ import annotations

import re

_DOC = re.compile(r"\bdoc_\d+\b", re.I)
_QUOTED = re.compile(r"«[^»]*»")
_PAREN = re.compile(r"\([^)]*\)")
_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def norm(s: str) -> str:
    s = s.lower().replace("ё", "е")
    s = re.sub(r"[«»\"'“”„‘’]", "", s)
    return re.sub(r"\s+", " ", s).strip(" .,;:!?")


def _numbers(s: str) -> list[str]:
    return [n.replace(",", ".") for n in _NUM.findall(s)]


def gold_numbers(gold: str) -> list[str]:
    core = _DOC.sub(" ", _QUOTED.sub(" ", _PAREN.sub(" ", gold)))
    return _numbers(core)


def gold_names(gold: str) -> list[str]:
    return [norm(m) for m in _QUOTED.findall(_PAREN.sub(" ", gold))]


_VERSUS = re.compile(r"\sпротив\s")        # «противоречие» (T3) не совпадает: после «против» нет пробела


def is_t2_gold(gold: str) -> bool:
    return bool(_VERSUS.search(gold)) and len(gold_numbers(gold)) >= 2


def value_in_answer(gold_answer: str, answer: str, quote: str = "", strict: bool = False) -> bool:
    """Основная проверка. strict=True: для T2 требуются оба числа (диагностика); иначе только число победителя."""
    answer = f"{answer or ''} {quote or ''}".strip()
    nums = gold_numbers(gold_answer)
    if nums:
        have = set(_numbers(_DOC.sub(" ", _QUOTED.sub(" ", answer))))      # цифры внутри названий («Талис-56») не считаем
        if not strict and is_t2_gold(gold_answer):
            return nums[0] in have                                         # победитель — первое число эталона
        return all(n in have for n in nums)
    names = gold_names(gold_answer)
    a = norm(answer)
    if names:
        return all(n in a for n in names)
    return norm(gold_answer) in a


def value_in_answer_strict(gold_answer: str, answer: str, quote: str = "") -> bool:
    return value_in_answer(gold_answer, answer, quote, strict=True)
