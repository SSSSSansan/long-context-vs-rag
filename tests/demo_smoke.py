"""Дымовая проверка демо: каждый экран рендерится без исключений и без сообщения «экран не удалось построить» (streamlit.testing).
Запуск в окружении демо:  python tests/demo_smoke.py [путь/к/app.py]"""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent.parent / "demo/app.py")


def problems(at) -> list[str]:
    out = [f"исключение: {e.message}" for e in at.exception]
    out += [w.value for w in at.warning if "не удалось построить" in str(w.value)]
    return out


at = AppTest.from_file(APP, default_timeout=90).run()
if not at.sidebar.radio:
    print("FAIL приложение не построило боковую панель:", [e.value for e in at.error] or [e.message for e in at.exception])
    sys.exit(1)
bad = 0
for p in at.sidebar.radio[0].options:
    at.sidebar.radio[0].set_value(p).run()
    pr = problems(at)
    bad += bool(pr)
    print(("OK  " if not pr else "FAIL"), p, pr)
# калькулятор: крайние значения ползунков
at.sidebar.radio[0].set_value("Деньги").run()
for label, vals in (("Вопросов в месяц", (0, 1_000_000)), ("Доля кэша LC, %", (0, 90))):
    for v in vals:
        s = next(x for x in at.slider if x.label == label)
        s.set_value(v).run()
        pr = problems(at)
        bad += bool(pr)
        print(("OK  " if not pr else "FAIL"), f"калькулятор: {label} = {v}", pr)
sys.exit(1 if bad else 0)
