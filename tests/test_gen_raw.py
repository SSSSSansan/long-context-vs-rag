from src import gen_raw_corpus as g


def test_num_gen_forms():
    assert g.num_gen(1, "рабочего дня", "рабочих дней") == "1 рабочего дня"
    assert g.num_gen(14, "рабочего дня", "рабочих дней") == "14 рабочих дней"
    assert g.num_gen(21, "рабочего дня", "рабочих дней") == "21 рабочего дня"
    assert g.num_gen(11, "года", "лет") == "11 лет"
    assert g.num_gen(71, "года", "лет") == "71 года"


def test_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    g.generate(a)
    g.generate(b)
    names = sorted(p.name for p in a.glob("doc_*.txt"))
    assert len(names) == 48
    for n in names:
        assert (a / n).read_bytes() == (b / n).read_bytes()


def test_no_unformatted_placeholders(tmp_path):
    g.generate(tmp_path)
    for p in tmp_path.glob("doc_*.txt"):
        t = p.read_text(encoding="utf-8")
        assert "{" not in t and "}" not in t
        assert t.startswith("ДОКУМЕНТ doc_")
