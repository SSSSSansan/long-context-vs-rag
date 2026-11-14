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
