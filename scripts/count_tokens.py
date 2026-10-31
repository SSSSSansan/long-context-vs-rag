"""Сколько токенов в сгенерированном корпусе (токенайзер из configs/base.yaml).

    python -m scripts.count_tokens
"""
from pathlib import Path

import tiktoken

from src.config import ROOT, backend_cfg

enc = tiktoken.get_encoding(backend_cfg()["tokenizer"])
docs = sorted((ROOT / "corpus/raw").glob("doc_*.txt"))
toks = [len(enc.encode(d.read_text(encoding="utf-8"))) for d in docs]
print(f"токенайзер: {backend_cfg()['tokenizer']}")
print(f"документов: {len(docs)}")
print(f"токенов всего: {sum(toks)}")
print(f"на документ min/avg/max: {min(toks)} / {sum(toks)//len(toks)} / {max(toks)}")
print("для 100K с запасом на факты и обрезку нужно >= 110000")
