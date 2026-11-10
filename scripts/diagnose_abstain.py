"""Диагностика аномалии: gpt-4o-mini, LC, 8K, T1 — accuracy 0.20, abstain до 100% при evidence recall 1.0.

Читает runs/*/raw.jsonl без новых вызовов API. Для каждого вызова gpt-4o-mini/LC/8K/T1 выводит
question, gold, сырой ответ модели, parsed_ok, abstained, вердикт судьи, и проверяет гипотезы:

  (a) ответ не распарсился как JSON -> засчитан как отказ по умолчанию
  (b) abstained определяется подстрокой и сработал ложно (фраза есть в raw_text,
      но НЕ в распарсенном поле "answer" — например, в quote или в тексте вне JSON)
  (c) факт реально есть в промпте, который получила модель (проверка по gold_answer
      как подстроке отправленного prompt)
  (d) модель действительно отказывается, это не объясняется (a)/(b) — поведение, а не баг
  (e) ошибка судьи: вердикт не "abstain", хотя ответ — ровно фраза отказа

Запуск: python -m scripts.diagnose_abstain
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ABSTAIN_PHRASE = "В предоставленных документах ответа нет"

TARGET_MODEL = "gpt-4o-mini-2024-07-18"
TARGET_METHOD = "lc"
TARGET_SIZE = "8k"
TARGET_TYPE = "T1"


def load_rows() -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(ROOT / "runs" / "*" / "raw.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if (
                    d.get("model") == TARGET_MODEL
                    and d.get("method") == TARGET_METHOD
                    and d.get("size") == TARGET_SIZE
                    and d.get("type") == TARGET_TYPE
                ):
                    d["_src"] = path
                    rows.append(d)
    return rows


def classify(d: dict) -> str:
    parsed_ok = d.get("parsed_ok")
    abstained = d.get("abstained")
    answer_field = d.get("answer") or ""
    verdict = d.get("verdict")

    if not parsed_ok:
        return "a_parse_failure"
    if abstained and ABSTAIN_PHRASE not in answer_field:
        return "b_false_positive_substring"
    if abstained and ABSTAIN_PHRASE in answer_field:
        if verdict != "abstain":
            return "e_judge_mismatch"
        return "d_genuine_abstain"
    return "f_other"  # не отказался вовсе


def fact_in_prompt(d: dict) -> bool | None:
    prompt = d.get("prompt")
    gold = d.get("gold_answer")
    if not prompt or not gold:
        return None
    # число/короткая фраза могут входить как часть другого слова — проверяем как подстроку,
    # этого достаточно, чтобы отличить "факта в контексте нет" от "факт есть, модель не достала"
    return gold in prompt


def main() -> None:
    rows = load_rows()
    print(f"Найдено вызовов gpt-4o-mini / LC / 8K / T1: {len(rows)}")
    if not rows:
        print("Нет данных — проверьте runs/*/raw.jsonl")
        return

    counts: dict[str, int] = {}
    per_call = []
    for d in rows:
        cat = classify(d)
        counts[cat] = counts.get(cat, 0) + 1
        present = fact_in_prompt(d)
        per_call.append((d, cat, present))

    print("\n=== Сводка по гипотезам ===")
    labels = {
        "a_parse_failure": "(а) JSON не распарсился -> ложный отказ",
        "b_false_positive_substring": "(б) подстрока сработала ложно (фраза не в поле answer)",
        "d_genuine_abstain": "(г) модель реально написала фразу отказа в answer",
        "e_judge_mismatch": "(д) судья дал вердикт != abstain при явном тексте отказа",
        "f_other": "модель НЕ отказалась (answer != фраза отказа)",
    }
    for key, label in labels.items():
        print(f"  {label}: {counts.get(key, 0)}")

    present_true = sum(1 for _, _, p in per_call if p is True)
    present_false = sum(1 for _, _, p in per_call if p is False)
    present_unknown = sum(1 for _, _, p in per_call if p is None)
    print("\n=== (в) Факт реально в промпте модели? ===")
    print(f"  gold_answer найден как подстрока prompt: {present_true}")
    print(f"  gold_answer НЕ найден как подстрока prompt: {present_false}")
    print(f"  нельзя проверить (нет prompt/gold в raw.jsonl): {present_unknown}")

    print("\n=== 5 примеров дословно ===")
    for d, cat, present in per_call[:5]:
        print("-" * 60)
        print("source:", d["_src"])
        print("qid:", d.get("qid"), "| repeat:", d.get("repeat"))
        print("question (из questions.json, см. qid) | gold_answer:", repr(d.get("gold_answer")))
        print("raw_text (сырой ответ модели):", repr(d.get("raw_text")))
        print("parsed_ok:", d.get("parsed_ok"), "| abstained:", d.get("abstained"))
        print("judge verdict:", d.get("verdict"), "| judge_rationale:", d.get("judge_rationale"))
        print("evidence_recall:", d.get("evidence_recall"))
        print("gold_answer в отправленном prompt:", present)
        print("категория:", cat)


if __name__ == "__main__":
    main()
