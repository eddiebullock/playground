"""JSONL checkpoint keyed by (model, item, condition, permutation, seed)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

KEY_FIELDS = ("model", "item", "condition", "permutation", "seed")


def record_key(record: dict) -> tuple:
    missing = [field for field in KEY_FIELDS if field not in record]
    if missing:
        raise RuntimeError(f"JSONL record missing key fields: {missing}")
    return tuple(record[field] for field in KEY_FIELDS)


class JsonlStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.done: set[tuple] = set()
        if self.path.exists():
            self.done = set(self._load_keys())

    def _load_keys(self) -> Iterable[tuple]:
        with self.path.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"{self.path.name}:{lineno} is not valid JSON") from exc
                yield record_key(record)

    def contains(self, record: dict) -> bool:
        return record_key(record) in self.done

    def append(self, record: dict) -> bool:
        """Write the record. Return False if that key was already stored."""
        key = record_key(record)
        if key in self.done:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
        self.done.add(key)
        return True
