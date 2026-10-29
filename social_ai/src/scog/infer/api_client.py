"""Thin client for a closed model: retries, a minimum interval, and a disk cache.

The cache key is (model, prompt hash, options in the given order, seed).
The transport is injected so tests do not call a network.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable

RETRY_STATUS = {429, 500, 502, 503, 504}


def cache_key(model: str, prompt: str, options: list[str], seed: int) -> str:
    payload = json.dumps(
        {"model": model, "prompt": prompt, "options": list(options), "seed": seed},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ClosedModelClient:
    def __init__(
        self,
        transport: Callable[[dict], dict],
        cache_dir: str | Path,
        *,
        max_retries: int = 5,
        min_interval_s: float = 0.0,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.transport = transport
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.min_interval_s = min_interval_s
        self.sleeper = sleeper
        self.clock = clock
        self._last_call = 0.0

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def complete(self, model: str, prompt: str, options: list[str], seed: int) -> dict:
        key = cache_key(model, prompt, options, seed)
        path = self._path(key)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        request = {"model": model, "prompt": prompt, "options": list(options), "seed": seed}
        response = self._send(request)
        path.write_text(json.dumps(response, sort_keys=True), encoding="utf-8")
        return response

    def _send(self, request: dict) -> dict:
        last_error = "no attempt"
        for attempt in range(self.max_retries):
            self._wait()
            try:
                response = self.transport(request)
            except TimeoutError as exc:
                last_error = str(exc)
                self.sleeper(min(8.0, 0.5 * (2**attempt)))
                continue
            status = int(response.get("status", 200))
            if status in RETRY_STATUS:
                last_error = f"status {status}"
                self.sleeper(min(8.0, 0.5 * (2**attempt)))
                continue
            if status >= 400:
                raise RuntimeError(f"closed model rejected the request with status {status}")
            return response
        raise RuntimeError(f"closed model failed after {self.max_retries} attempts: {last_error}")

    def _wait(self) -> None:
        if self.min_interval_s <= 0:
            self._last_call = self.clock()
            return
        elapsed = self.clock() - self._last_call
        if elapsed < self.min_interval_s:
            self.sleeper(self.min_interval_s - elapsed)
        self._last_call = self.clock()
