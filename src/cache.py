"""Дисковый кэш ответов по хэшу (backend + модель + промпт + параметры).

Падение на 900-м вызове из 1080 не должно стоить денег: при перезапуске
уже посчитанные вызовы читаются с диска.
Повторы (repeat=1,2,3) различаются полем `salt` — иначе 3 повтора дали бы 1 вызов.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import ROOT, base_cfg


def _dir() -> Path:
    d = ROOT / base_cfg()["budget"]["cache_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def get(key: str) -> dict | None:
    p = _dir() / f"{key}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def put(key: str, value: dict) -> None:
    (_dir() / f"{key}.json").write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
