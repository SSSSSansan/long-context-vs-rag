import pytest

from src.llm_client import (ContextTooLong, ContextTruncated,
                            check_fits, check_not_truncated)


def test_fits_ok_and_none():
    check_fits(10_000, 32768)
    check_fits(None, 32768)          # токенайзер недоступен — не блокируем


def test_fits_raises():
    with pytest.raises(ContextTooLong):
        check_fits(31_500, 32768)    # > 95% окна


def test_truncation_detected():
    # отправили ~48K, провайдер принял 16386 (реальный случай с Ollama)
    with pytest.raises(ContextTruncated):
        check_not_truncated(16_386, 48_000)


def test_no_false_positive_when_tokenizer_costlier():
    check_not_truncated(30_000, 20_000)   # Qwen тратит больше токенов, чем o200k — это норма


def test_small_prompts_not_checked():
    check_not_truncated(10, 1_500)
