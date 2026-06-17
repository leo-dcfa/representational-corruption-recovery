"""Small JSON/JSONL helpers used across datagen, eval, and stats."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def _json_default(o: Any) -> Any:
    """Coerce common non-JSON types (numpy scalars/arrays, Path) to plain Python.

    Validators and stats routinely produce numpy.bool_/float64/int64 and arrays;
    json.dump can't serialize those. This default makes every write_json call
    robust without each caller having to sanitize first.
    """
    # numpy scalars expose .item(); arrays expose .tolist()
    if hasattr(o, "item") and not hasattr(o, "tolist"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
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
        json.dump(obj, fh, ensure_ascii=False, indent=indent, default=_json_default)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r") as fh:
        return json.load(fh)
