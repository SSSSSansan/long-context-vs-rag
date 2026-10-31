from src import chunking as c


def dummy(t: str) -> int:
    return len(t) // 4


def make_corpus() -> str:
    parts = []
    for d in (1, 2):
        parts.append(f"ДОКУМЕНТ doc_{d:03d}\nЗаголовок\n")
        parts.extend(f"{d}.{i}. " + " ".join(["слово"] * 20) for i in range(30))
        parts.append("")
    return "\n".join(parts)


def test_split_docs():
    docs = c.split_docs(make_corpus())
    assert [d for d, _ in docs] == ["doc_001", "doc_002"]


def test_chunk_ids_format_and_unique():
    chunks = c.chunk_corpus(make_corpus(), 300, 40, dummy)
    ids = [x.chunk_id for x in chunks]
    assert len(ids) == len(set(ids))
    assert ids[0] == "doc_001_c00"
    assert all(x.chunk_id.startswith(x.doc_id + "_c") for x in chunks)


def test_deterministic():
    a = c.chunk_corpus(make_corpus(), 300, 40, dummy)
    b = c.chunk_corpus(make_corpus(), 300, 40, dummy)
    assert a == b


def test_every_line_covered():
    corpus = make_corpus()
    joined = "\n".join(x.text for x in c.chunk_corpus(corpus, 300, 40, dummy))
    for line in corpus.split("\n"):
        if line.strip():
            assert line in joined


def test_size_limit_and_overlap():
    chunks = [x for x in c.chunk_corpus(make_corpus(), 300, 40, dummy) if x.doc_id == "doc_001"]
    assert all(x.n_tokens <= 300 * 1.05 for x in chunks)
    # соседние чанки делят хотя бы одну строку (перекрытие)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert set(prev.text.split("\n")) & set(nxt.text.split("\n"))
