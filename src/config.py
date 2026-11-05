"""Загрузка конфигов и .env. Единственное место, где читаются yaml и переменные окружения."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@lru_cache(maxsize=None)
def load_yaml(rel_path: str) -> dict:
    with open(ROOT / rel_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def base_cfg() -> dict:
    return load_yaml("configs/base.yaml")


def backend_name() -> str:
    name = os.getenv("LLM_BACKEND", "local").lower()
    if name not in ("openai", "local"):
        raise ValueError(f"LLM_BACKEND must be openai|local, got {name!r}")
    return name


def backend_cfg() -> dict:
    cfg = base_cfg()["backends"][backend_name()]
    model = os.getenv("CHAT_MODEL")      # переопределение модели ответов (судья остаётся прежним)
    if model and backend_name() == "openai":
        cfg = {**cfg, "chat_model": model, "price_per_1m_usd": base_cfg()["model_prices"][model]}
    return cfg
