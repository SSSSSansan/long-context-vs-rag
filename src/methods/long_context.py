"""LC: весь корпус в контекст. Корпус идёт в начале промпта (общий префикс для prompt caching)."""
from __future__ import annotations

from functools import lru_cache

from ..config import ROOT, load_yaml
from . import Context


@lru_cache(maxsize=None)
def load_corpus(size: str) -> str:
    return (ROOT / load_yaml("configs/corpus.yaml")["built_dir"] / size / "corpus.txt").read_text(encoding="utf-8")


def build_context(size: str, question: str, k: int | None = None, corpus_text: str | None = None) -> Context:
    return Context(text=corpus_text if corpus_text is not None else load_corpus(size), full_corpus=True)
