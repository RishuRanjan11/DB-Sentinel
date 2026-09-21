from __future__ import annotations


def section(title: str) -> None:
    print(f"\n{title}")
    print("─" * len(title))


def status(label: str, value: str) -> None:
    print(f"{label}: {value}")
