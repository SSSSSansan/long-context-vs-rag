"""Проверка путей из README.md и REPORT.md: каждый упомянутый файл репозитория должен существовать. Без сети и API.
   python -m scripts.check_links"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH = re.compile(r"(?<![\w/.-])((?:results|data|configs|scripts|src|docs|demo|tests|corpus|runs|notebooks|v1)/[\w.\-/]+\.[A-Za-z0-9]+|(?:results|data|configs|scripts|src|docs|demo|tests|corpus|notebooks|v1)/[\w.\-]+/)")
SKIP_PREFIX = ("runs/",)   # runs/ в .gitignore: сырые логи не хранятся в репозитории


def main() -> int:
    bad, ok = [], 0
    for name in ("README.md", "REPORT.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            for m in PATH.finditer(line):
                p = m.group(1).rstrip(".,;:)")
                if p.startswith(SKIP_PREFIX) or any(c in p for c in "*<>{}"):
                    continue
                if (ROOT / p).exists():
                    ok += 1
                else:
                    bad.append(f"{name}:{i}: {p}")
    print(f"проверено существующих путей: {ok}; битых: {len(bad)}")
    print("\n".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
