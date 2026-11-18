import pandas as pd
import pytest

from src.eval.value_check import gold_names, gold_numbers, value_in_answer as v, value_in_answer_strict as vs

T2 = "«Нерида-91» (учреждённая в 2020 году для докторантов и курируемая подразделением «Бюро карьерного развития»): 17 недель против 20 недель у «Талис-56»"
T2_PLAIN = "«Вельта-76»: 75 баллов против 66 баллов у «Ольвия-49»"
T3 = "Да, противоречие: в doc_001 указано 18 недель, а в doc_002 — 7 недель"
T3_NAMES = "Да, противоречие: в doc_002 указано «Бюро лабораторных исследований», а в doc_001 — «Отдел академической репутации»"


def test_gold_numbers_ignore_names_docs_and_descriptors():
    assert gold_numbers(T2) == ["17", "20"]            # 2020, 91, 56 — из описания и названий, не значения
    assert gold_numbers(T2_PLAIN) == ["75", "66"]
    assert gold_numbers(T3) == ["18", "7"]             # doc_001 / doc_002 не числа
    assert gold_numbers("114 МРП") == ["114"]
    assert gold_names(T3_NAMES) == ["бюро лабораторных исследований", "отдел академической репутации"]


def test_t2_name_only_is_rejected():
    assert not v(T2, "Программа «Нерида-91» имеет меньшую продолжительность.")
    assert not v(T2_PLAIN, "Вельта-76")
    assert not vs(T2_PLAIN, "Вельта-76")


def test_t2_with_both_values_accepted_and_paraphrase():
    assert v(T2, "«Нерида-91» — 17 недель, у «Талис-56» 20 недель")
    assert v(T2, "Нерида-91: 17 и 20")
    assert v(T2, "17 нед. против 20 нед.")
    assert v(T2_PLAIN, "у Вельта-76 75 баллов, у Ольвия-49 — 66")


def test_t2_winner_value_only_is_accepted_as_main_rule():
    """Основное правило T2: значение победителя (первое число эталона) достаточно, второго числа может не быть."""
    assert v(T2, "Меньшую продолжительность имеет программа «Нерида-91» — 17 недель.")
    assert v(T2, "17 недель")
    assert v(T2_PLAIN, "У «Вельта-76» 75 баллов")


def test_t2_strict_still_requires_both_numbers():
    """Строгое правило «оба числа» сохранено как диагностика (value_ok_strict)."""
    assert not vs(T2, "17 недель")
    assert vs(T2, "17 недель против 20 недель")


def test_t2_wrong_winner_or_name_only_is_rejected():
    assert not v(T2, "«Талис-56» — 20 недель")                                  # значение проигравшего, не победителя
    assert not v(T2, "Более короткий срок у «Нерида-91».")                      # имя без числа: защита от угадывания 50%
    assert not v(T2, "Меньше баллов нужно для зачисления")                      # ни имени, ни числа
    assert not v(T2, "")


def test_t3_and_t1_unchanged_by_winner_rule():
    assert not v(T3, "Да, 18 недель")                                           # T3: нужны оба значения
    assert not vs(T3, "Да, 18 недель")
    assert v("114 МРП", "114 МРП") and vs("114 МРП", "114 МРП")


def test_t3_word_protivorechie_is_not_confused_with_t2_marker():
    assert not v(T3_NAMES, "Да, «Бюро лабораторных исследований»")              # нужны оба имени, как раньше


def test_digits_inside_names_do_not_count_as_values():
    assert not v(T2, "«Талис-56» (20 недель) учреждена в 2020 году, «Нерида-91»")     # нет 17


def test_t3_numbers_and_names():
    assert v(T3, "Да, противоречие: 18 недель в doc_001 и 7 недель в doc_002")
    assert not v(T3, "Да, противоречие есть")
    assert v(T3_NAMES, "Да: «Бюро лабораторных исследований» и «Отдел академической репутации»")
    assert v(T3_NAMES, "Бюро лабораторных исследований / отдел академической репутации")
    assert not v(T3_NAMES, "Да, противоречие есть, в doc_001 и doc_002 разные подразделения")


def test_decimal_comma_and_dot_equal():
    assert v("Срок составляет 2,5 месяца", "2.5 месяца")
    assert v("Срок составляет 2.5 месяца", "около 2,5 мес.")


def test_t1_number():
    assert v("114 МРП", "Размер выплаты — 114 МРП")
    assert not v("114 МРП", "Размер выплаты — 141 МРП")
    assert not v("114 МРП", "")


def test_t1_name_case_yo_and_foreign_name():
    assert v("Центр проектного обучения", "Заявки принимает «Центр проектного обучения».")
    assert v("Центр проектного обучения", "центр ПРОЕКТНОГО обучения")
    assert v("Отдел учёта", "отдел учета")                          # ё / е
    assert not v("Центр проектного обучения", "Бюро регистрации и учёта")
    assert not v("Центр проектного обучения", None)


def test_quote_counts_as_part_of_answer_like_the_judge_sees_it():
    assert not v(T2, "Программа «Нерида-91» короче.")
    assert v(T2, "Программа «Нерида-91» короче.", "Срок — 17 недель; … Срок — 20 недель.")
    assert v(T3_NAMES, "true", "(doc_002) подразделение «Бюро лабораторных исследований» … (doc_001) «Отдел академической репутации»")
    assert v(T2, "Нерида-91", "Срок — 17 недель.")                      # число победителя в цитате: основное правило принимает
    assert not vs(T2, "Нерида-91", "Срок — 17 недель.")                 # строгое («оба числа») — нет


def test_abstention_is_not_value():
    assert not v("114 МРП", "В предоставленных документах ответа нет.")


def test_aggregate_final_accuracy_requires_value():
    from src.eval import aggregate as ag
    rows = [dict(gold_answer=T2, answer="Нерида-91", verdict="correct"),
            dict(gold_answer=T2, answer="Нерида-91: 17 и 20", verdict="correct"),
            dict(gold_answer=T2, answer="Нерида-91: 17 и 20", verdict="partial")]
    df = pd.DataFrame(rows)
    df["correct_judge"] = (df.verdict == "correct").astype(float)
    df["value_ok"] = [v(g, a) for g, a in zip(df.gold_answer, df.answer)]
    df["correct"] = df.correct_judge * df.value_ok.astype(float)
    assert list(df.correct) == [0.0, 1.0, 0.0] and list(df.correct_judge) == [1.0, 1.0, 0.0]
    assert ag.value_in_answer is v                                  # aggregate использует ту же функцию
