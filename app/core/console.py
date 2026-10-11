from __future__ import annotations

import sys


def _safe_console_text(value: str) -> str:
    encoding = (sys.stdout.encoding or "").lower()
    if "utf" in encoding:
        return value

    replacements = {
        "─": "-",
        "•": "-",
        "—": "-",
        "–": "-",
        "│": "|",
    }
    return "".join(replacements.get(char, char) for char in value)


def section(title: str) -> None:
    print(f"\n{title}")
    print(_safe_console_text("─" * len(title)))


def status(label: str, value: str) -> None:
    print(f"{label}: {value}")


def safe_text(value: str) -> str:
    return _safe_console_text(value)


def configure_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
