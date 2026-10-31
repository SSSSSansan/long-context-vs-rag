"""Шаг 1.4: сборка вложенных корпусов 8K ⊂ 30K ⊂ 100K ровно по токенам.

Идея: все документы выстраиваются в один длинный «поток» строк (doc_001, doc_002, ...).
Корпус размера N — это ПРЕФИКС потока, обрезанный по границе строки так, чтобы
уложиться в N токенов. Поэтому корпуса вложены автоматически: 8K — начало 30K, а 30K —
начало 100K. Один и тот же вопрос можно задавать на всех трёх размерах, и меняется
только количество отвлекающего материала.

Последний документ в корпусе может оказаться оборванным по границе пункта — это нормально.

Если есть corpus/built/planted (результат make facts), корпуса собираются из документов с внедрёнными
фактами, а data/gold.json дополняется: в каких чанках лежит факт, на какой доле корпуса и в каких размерах.
Присутствие каждого факта проверяется жёстко: в своём размере и больших ровно один раз, в меньших — ни разу.

Запуск:  python -m src.build_corpus
Выход:   corpus/built/{8k,30k,100k}/corpus.txt и meta.json (корпуса в git не коммитятся)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .chunking import chunk_corpus
from .config import ROOT, load_yaml
from .tokens import count_tokens


def load_raw_docs(raw_dir: Path) -> list[tuple[str, str]]:
    """[(doc_id, text)] в порядке doc_001, doc_002, ..."""
    return [(p.stem, p.read_text(encoding="utf-8")) for p in sorted(raw_dir.glob("doc_*.txt"))]


def stream_lines(docs: list[tuple[str, str]]) -> list[str]:
    """Один поток строк; между документами пустая строка."""
    lines: list[str] = []
    for i, (_, text) in enumerate(docs):
        if i:
            lines.append("")
        lines.extend(text.rstrip("\n").split("\n"))
    return lines


def assemble(lines: list[str], target: int, count: Callable[[str], int],
             tol: float) -> tuple[str, int]:
    """Самый длинный префикс потока, который укладывается в `target` токенов.

    Возвращает (текст, число_строк). Бросает ValueError, если материала не хватает
    или результат оказался меньше target*(1-tol).
    """
    def text_of(n: int) -> str:
        return "\n".join(lines[:n]) + "\n"

    if count(text_of(len(lines))) < target * (1 - tol):
        raise ValueError(f"Фонового материала не хватает для {target} токенов: "
                         f"есть {count(text_of(len(lines)))}. Увеличь n_docs/длину документов.")
    lo, hi = 0, len(lines)            # ищем max n: count(text_of(n)) <= target
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(text_of(mid)) <= target:
            lo = mid
        else:
            hi = mid - 1
    text = text_of(lo)
    tokens = count(text)
    if tokens < target * (1 - tol):
        raise ValueError(f"Корпус {target}: получилось {tokens} токенов, это ниже допуска {tol:.0%}.")
    return text, lo


def finalize_gold(corpora: dict[str, str], count: Callable[[str], int] = count_tokens) -> None:
    """Дописывает в data/gold.json chunk_ids, positions_pct, corpus_sizes и проверяет присутствие фактов."""
    gold_path = ROOT / "data" / "gold.json"
    if not gold_path.exists():
        return
    keys = list(corpora)                       # ["8k", "30k", "100k"] — по возрастанию
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    chunks = chunk_corpus(corpora[keys[-1]], count=count)
    for f in gold["facts"]:
        first = keys.index(f["first_corpus"])
        f["corpus_sizes"], f["positions_pct"] = [], {}
        for i, key in enumerate(keys):
            n = corpora[key].count(f["statement"])
            if i >= first and n != 1:
                raise RuntimeError(f"Факт {f['fact_id']} должен встречаться в корпусе {key} ровно 1 раз, найдено {n}. "
                                   f"Вероятно, документ попал на границу обрезки: пересобери (make facts corpus).")
            if i < first and n != 0:
                raise RuntimeError(f"Факт {f['fact_id']} не должен быть в корпусе {key}, но найден.")
            if n == 1:
                f["corpus_sizes"].append(key)
                f["positions_pct"][key] = round(100 * corpora[key].index(f["statement"]) / len(corpora[key]), 1)
        # все чанки, содержащие пункт (из-за перекрытия их может быть два)
        f["chunk_ids"] = [c.chunk_id for c in chunks if f["statement"] in c.text]
        if not f["chunk_ids"]:
            raise RuntimeError(f"Факт {f['fact_id']} не найден ни в одном чанке.")
    gold_path.write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")
    per = {k: sum(1 for f in gold["facts"] if k in f["corpus_sizes"]) for k in keys}
    print("факты в корпусах:", per)


def main(count: Callable[[str], int] = count_tokens) -> None:
    cfg = load_yaml("configs/corpus.yaml")
    tol = cfg["size_tolerance"]
    planted_dir = ROOT / cfg["planted_dir"]
    src_dir = planted_dir if any(planted_dir.glob("doc_*.txt")) else ROOT / cfg["raw_dir"]
    docs = load_raw_docs(src_dir)
    if not docs:
        raise SystemExit("В corpus/raw нет документов. Сначала: make raw")
    print(f"источник документов: {src_dir.relative_to(ROOT)}")
    lines = stream_lines(docs)

    corpora: dict[str, str] = {}
    print(f"{'корпус':8s}{'цель':>9s}{'токенов':>9s}{'документов':>12s}{'последний документ':>22s}")
    for key, target in cfg["sizes_tokens"].items():
        text, _ = assemble(lines, target, count, tol)
        corpora[key] = text
        doc_ids = [l.split()[1] for l in text.split("\n") if l.startswith("ДОКУМЕНТ doc_")]
        full = {d for d, t in docs if t.rstrip("\n") in text}
        truncated = doc_ids[-1] not in full
        out = ROOT / cfg["built_dir"] / key
        out.mkdir(parents=True, exist_ok=True)
        (out / "corpus.txt").write_text(text, encoding="utf-8")
        tokens = count(text)
        (out / "meta.json").write_text(json.dumps({
            "size_key": key, "target_tokens": target, "tokens": tokens,
            "n_docs": len(doc_ids), "doc_ids": doc_ids, "last_doc_truncated": truncated,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{key:8s}{target:9d}{tokens:9d}{len(doc_ids):12d}{('оборван' if truncated else 'целиком'):>22s}")

    if src_dir == planted_dir:
        finalize_gold(corpora, count)


if __name__ == "__main__":
    main()
