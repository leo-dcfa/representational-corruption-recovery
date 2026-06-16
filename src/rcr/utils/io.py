"""Small JSON/JSONL helpers used across datagen, eval, and stats."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=indent)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r") as fh:
        return json.load(fh)
