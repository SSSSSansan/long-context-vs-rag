"""Мощность точного парного теста McNemar при n пар (по умолчанию 18). Бесплатно, без API, детерминированно (точный расчёт, не симуляция).

Модель: доля несогласных пар psi (вопрос верен только в одном из двух условий); из них доля (psi+delta)/(2*psi) в пользу A, где
delta = разница accuracy между условиями. Мощность = P(p_value < alpha) при таких параметрах.
Alpha: 0.05 без поправки и 0.05/m (граница Холма для самого малого p) при m тестах в семье (по умолчанию 15, как в семье «якорь»).
   python -m scripts.power_mcnemar [--n 18] [--m 15]
"""
from __future__ import annotations

import argparse
import math

from src.eval.aggregate import mcnemar_exact


def power(n: int, delta: float, psi: float, alpha: float) -> float:
    """P(p<alpha) при n парах, разнице accuracy delta и доле несогласных пар psi (psi >= |delta|)."""
    pa = (psi + delta) / 2          # P(верно только в A)
    pb = (psi - delta) / 2          # P(верно только в B)
    total = 0.0
    for oa in range(n + 1):
        for ob in range(n - oa + 1):
            p = math.comb(n, oa) * math.comb(n - oa, ob) * pa ** oa * pb ** ob * (1 - pa - pb) ** (n - oa - ob)
            if mcnemar_exact(oa, ob) < alpha:
                total += p
    return total


def min_discordant(n: int, alpha: float) -> int:
    """Наименьшее число несогласных пар, при котором p<alpha, если ВСЕ они в одну сторону."""
    for d in range(1, n + 1):
        if mcnemar_exact(d, 0) < alpha:
            return d
    return n + 1   # недостижимо


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--m", type=int, default=15, help="число тестов в семье (для границы Холма alpha/m)")
    a = ap.parse_args()
    print(f"n = {a.n} пар; семья из m = {a.m} тестов")
    for name, alpha in (("без поправки", 0.05), (f"Холм (alpha/m = {0.05 / a.m:.4f})", 0.05 / a.m)):
        d = min_discordant(a.n, alpha)
        print(f"\n[{name}] p<alpha возможно, только если несогласных пар >= {d} и все в одну сторону: "
              f"минимально различимая разница accuracy = {d}/{a.n} = {d / a.n:.2f}" if d <= a.n else
              f"\n[{name}] при n={a.n} p<alpha недостижимо ни при каких данных")
        if d <= a.n:
            print("  мощность (точный расчёт) при доле несогласных psi:")
            print("  delta   " + "".join(f"psi={p:<6.2f}" for p in (0.2, 0.4, 0.6, 0.8)))
            for delta in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
                cells = [f"{power(a.n, delta, p, alpha):<10.2f}" if p >= delta else f"{'-':<10}" for p in (0.2, 0.4, 0.6, 0.8)]
                print(f"  {delta:<7.1f} " + "".join(cells))


if __name__ == "__main__":
    main()
