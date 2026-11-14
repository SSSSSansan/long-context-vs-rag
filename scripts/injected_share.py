"""Доля внедрённого текста (факты + связки + дистракторы) в корпусах 8K/30K/100K. Бесплатно, без сети.

    python -m scripts.injected_share

Нужен data/gold.json (make facts). Считает токены строк-утверждений (+4 токена на номер пункта и перевод строки)
и делит на номинальный размер корпуса. После make corpus можно сверить с фактическим размером в meta.json.
"""
from __future__ import annotations

import json
from collections import defaultdict

from src.config import ROOT, load_yaml
from src.tokens import count_tokens

OVERHEAD = 4          # «2.5. » + перевод строки
GROUPS = [("gold", "факты"), ("link", "связки"), ("distractor_fact", "дистракторы: факты"),
          ("distractor_link", "дистракторы: связки")]


def main() -> None:
    cfg = load_yaml("configs/corpus.yaml")
    sizes = cfg["sizes_tokens"]
    keys = list(sizes)
    facts = json.loads((ROOT / "data/gold.json").read_text(encoding="utf-8"))["facts"]
    tok: dict[str, dict[str, int]] = {k: defaultdict(int) for k in keys}
    cnt: dict[str, dict[str, int]] = {k: defaultdict(int) for k in keys}
    for f in facts:
        role = f.get("role", "gold")
        t = count_tokens(f["statement"]) + OVERHEAD
        first = keys.index(f["first_corpus"])
        for k in keys[first:]:                     # факт уровня L есть во всех корпусах размера >= L
            tok[k][role] += t
            cnt[k][role] += 1
    print(f"{'':26s}" + "".join(f"{k.upper():>16s}" for k in keys))
    for role, label in GROUPS:
        print(f"{label:26s}" + "".join(f"{cnt[k][role]:>5d} стр {tok[k][role]:>5d} ток" for k in keys))
    print(f"{'ИТОГО внедрено':26s}" + "".join(f"{sum(cnt[k].values()):>5d} стр {sum(tok[k].values()):>5d} ток" for k in keys))
    print(f"{'доля от размера корпуса':26s}" + "".join(f"{100 * sum(tok[k].values()) / sizes[k]:>15.1f}%" for k in keys))
    for role, label in GROUPS[1:]:
        if any(cnt[k][role] for k in keys):
            print(f"  из них {label:22s}" + "".join(f"{100 * tok[k][role] / sizes[k]:>15.1f}%" for k in keys))


if __name__ == "__main__":
    main()
