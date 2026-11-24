from src.eval.exact_metrics import quote_grounded, quote_segments

CTX = (
    "ДОКУМЕНТ doc_011 (фрагмент doc_011_c03)\n"
    "Для зачисления в программу «Дарвен-49» необходимо набрать не менее 57 баллов.\n\n"
    "ДОКУМЕНТ doc_006 (фрагмент doc_006_c07)\n"
    "Минимальный проходной балл для поступления в программу «Хардан-79» составляет 59 баллов."
)


def test_single_quote_still_works():
    assert quote_grounded("Для зачисления в программу «Дарвен-49» необходимо набрать не менее 57 баллов.", CTX)


def test_joined_quote_two_documents_with_parenthesis_tags():
    q = ("Для зачисления в программу «Дарвен-49» необходимо набрать не менее 57 баллов. (doc_011) "
         "Минимальный проходной балл для поступления в программу «Хардан-79» составляет 59 баллов. (doc_006)")
    assert quote_grounded(q, CTX)


def test_joined_quote_two_documents_with_prefix_tags():
    q = ('doc_011: "Для зачисления в программу «Дарвен-49» необходимо набрать не менее 57 баллов." '
         'doc_006: "Минимальный проходной балл для поступления в программу «Хардан-79» составляет 59 баллов."')
    assert quote_grounded(q, CTX)
    assert len(quote_segments(q)) == 2


def test_one_invented_segment_fails():
    q = ("Для зачисления в программу «Дарвен-49» необходимо набрать не менее 57 баллов. (doc_011) "
         "Проходной балл в программу «Хардан-79» составляет 99 баллов. (doc_006)")
    assert not quote_grounded(q, CTX)


def test_empty_or_tags_only_is_false():
    assert not quote_grounded("", CTX)
    assert not quote_grounded("(doc_011) doc_006:", CTX)
    assert not quote_grounded(None, CTX)


def test_yo_and_spaces_normalised():
    assert quote_grounded("Для   зачисления в программу  Дарвен-49 необходимо набрать не менее 57 баллов.", CTX)


# --- нормализация формы цитаты (23 ноября): кавычки „“, ссылки на пункт, ё/е, тире; слова и числа не меняются ---
CTX2 = ("ДОКУМЕНТ doc_001 (фрагмент doc_001_c04)\n"
        "5.12. Для зачисления в премию «Дарвен-49» необходимо набрать не менее 59 баллов.\n"
        "5.15. Приём заявок — до 1 марта.\n\n"
        "ДОКУМЕНТ doc_002 (фрагмент doc_002_c10)\n"
        "6.13. Минимальный проходной балл для поступления в премию «Дарвен-49» составляет 57 баллов.")


def test_low_quotes_and_outer_quotes():
    assert quote_grounded("«Для зачисления в премию „Дарвен-49“ необходимо набрать не менее 59 баллов.»", CTX2)


def test_locator_after_doc_tag_is_not_text():
    q = ("doc_001, п. 5.12: «Для зачисления в премию „Дарвен-49“ необходимо набрать не менее 59 баллов.»; "
         "doc_002, п. 6.13: «Минимальный проходной балл для поступления в премию „Дарвен-49“ составляет 57 баллов.»")
    assert quote_grounded(q, CTX2)
    assert len(quote_segments(q)) == 2


def test_two_quotes_without_doc_tags_and_slash_join():
    a = "Для зачисления в премию «Дарвен-49» необходимо набрать не менее 59 баллов."
    b = "Минимальный проходной балл для поступления в премию «Дарвен-49» составляет 57 баллов."
    assert quote_grounded(f"«{a}»; «{b}»", CTX2)
    assert quote_grounded(f"{a} / {b}", CTX2)
    assert not quote_grounded(f"«{a}»; «Минимальный проходной балл составляет 99 баллов.»", CTX2)   # один кусок выдуман


def test_non_adjacent_clauses_with_numbers():
    assert quote_grounded("5.12. Для зачисления в премию «Дарвен-49» необходимо набрать не менее 59 баллов. "
                          "6.13. Минимальный проходной балл для поступления в премию «Дарвен-49» составляет 57 баллов.", CTX2)


def test_chunk_id_tag_and_dash_normalisation():
    assert quote_grounded("doc_001_c04: «Приём заявок — до 1 марта.»", CTX2)
    assert quote_grounded("Приём заявок – до 1 марта.", CTX2)           # короткое тире вместо длинного


def test_normalisation_does_not_accept_paraphrase_or_changed_number():
    assert not quote_grounded("«Для зачисления в премию „Дарвен-49“ нужно набрать не менее 59 баллов.»", CTX2)   # «нужно» вместо «необходимо»
    assert not quote_grounded("«Для зачисления в премию „Дарвен-49“ необходимо набрать не менее 60 баллов.»", CTX2)  # другое число
    assert not quote_grounded("Приём заявок до 1 марта.", CTX2)         # пропущено тире: нормализуется только вид тире, не его наличие
