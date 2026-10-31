import pytest

from src import build_corpus as b


def dummy(t: str) -> int:
    return len(t) // 4


LINES = [f"{i}. " + " ".join(["слово"] * 12) for i in range(400)]


def test_assemble_hits_target():
    text, _ = b.assemble(LINES, 500, dummy, tol=0.05)
    assert 475 <= dummy(text) <= 500


def test_corpora_are_nested():
    # строка ~19 токенов; цели выбраны так, чтобы шаг в одну строку укладывался в допуск 5%
    small, _ = b.assemble(LINES, 600, dummy, tol=0.05)
    big, _ = b.assemble(LINES, 1800, dummy, tol=0.05)
    assert big.startswith(small)


def test_not_enough_material_raises():
    with pytest.raises(ValueError):
        b.assemble(LINES[:5], 10_000, dummy, tol=0.03)


def test_stream_lines_separates_docs():
    lines = b.stream_lines([("doc_001", "ДОКУМЕНТ doc_001\nA\n"), ("doc_002", "ДОКУМЕНТ doc_002\nB\n")])
    assert lines == ["ДОКУМЕНТ doc_001", "A", "", "ДОКУМЕНТ doc_002", "B"]
