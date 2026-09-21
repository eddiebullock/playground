"""Non-blocking Thinker access.

The Driver loop never waits on the network. A background worker posts the
latest submitted frame and publishes the result when it arrives. Timeouts
and connection failures leave the last-good advisory in place.

Swap HttpThinkerBackend for LocalMockBackend (or a future in-process Qwen
wrapper) by passing a different backend into AsyncThinker — the rest of the
loop does not change.
"""

from __future__ import annotations

import io
import threading
import time
from typing import Optional, Protocol

import cv2
import numpy as np
import requests

from .advisory import AdvisoryView, from_server_json, safe_default


class ThinkerBackend(Protocol):
    def infer(self, jpeg_bytes: bytes, frame_index: int, client_ts: float) -> AdvisoryView:
        ...


class HttpThinkerBackend:
    def __init__(self, url: str, timeout_s: float = 8.0) -> None:
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s
        self.session = requests.Session()

    def infer(self, jpeg_bytes: bytes, frame_index: int, client_ts: float) -> AdvisoryView:
        t0 = time.time()
        try:
            resp = self.session.post(
                f"{self.url}/think",
                files={"image": ("frame.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
                data={"frame_index": str(frame_index), "client_ts": str(client_ts)},
                timeout=self.timeout_s,
            )
            rtt_ms = (time.time() - t0) * 1000.0
            resp.raise_for_status()
            view = from_server_json(resp.json(), rtt_ms=rtt_ms)
            return view
        except requests.Timeout:
            view = safe_default()
            view.status = "timeout"
            view.rtt_ms = (time.time() - t0) * 1000.0
            view.raw_text = f"TIMEOUT after {self.timeout_s}s talking to {self.url}"
            return view
        except requests.RequestException as exc:
            view = safe_default()
            view.status = "unreachable"
            view.rtt_ms = (time.time() - t0) * 1000.0
            view.raw_text = f"UNREACHABLE {self.url}: {exc}"
            return view

    def health(self) -> bool:
        try:
            r = self.session.get(f"{self.url}/health", timeout=2.0)
            return r.ok
        except requests.RequestException:
            return False


class LocalMockBackend:
    """In-process stand-in. Proves swapping HTTP for local is one constructor change."""

    def __init__(self) -> None:
        import sys
        from pathlib import Path

        thinker_dir = str(Path(__file__).resolve().parents[1] / "thinker")
        if thinker_dir not in sys.path:
            sys.path.insert(0, thinker_dir)
        from model import MockInferencer  # type: ignore

        self._inf = MockInferencer()

    def infer(self, jpeg_bytes: bytes, frame_index: int, client_ts: float) -> AdvisoryView:
        t0 = time.time()
        adv = self._inf.infer(jpeg_bytes)
        rtt_ms = (time.time() - t0) * 1000.0
        return from_server_json({**adv.to_dict(), "frame_index": frame_index}, rtt_ms=rtt_ms)


class AsyncThinker:
    """Latest-frame-wins worker. submit() never blocks; latest() is lock-copy."""

    def __init__(self, backend: ThinkerBackend) -> None:
        self.backend = backend
        self._lock = threading.Lock()
        self._pending: Optional[tuple[bytes, int, float]] = None
        self._latest = safe_default()
        self._busy = False
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._loop, name="thinker-worker", daemon=True)
        self._worker.start()

    def submit(self, frame_bgr: np.ndarray, frame_index: int) -> None:
        ok, jpeg = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        payload = (jpeg.tobytes(), frame_index, time.time())
        with self._lock:
            self._pending = payload

    def latest(self, now: Optional[float] = None) -> AdvisoryView:
        now = now if now is not None else time.time()
        with self._lock:
            view = AdvisoryView(**self._latest.to_dict())
        if view.model_ts > 0:
            view.age_ms = max(0.0, (now - view.model_ts) * 1000.0)
            view.is_stale = view.status != "ok" or view.age_ms > 0.0
            # Fresh means this cycle's Thinker result. Anything older is stale
            # even if status=ok — the Driver loop always reads "last known".
            # The harness stamps is_stale using the Thinker frame_index vs current.
        return view

    def close(self) -> None:
        self._stop.set()
        self._worker.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            job = None
            with self._lock:
                if self._pending is not None:
                    job = self._pending
                    self._pending = None
                    self._busy = True
            if job is None:
                time.sleep(0.01)
                continue
            jpeg_bytes, frame_index, client_ts = job
            result = self.backend.infer(jpeg_bytes, frame_index, client_ts)
            with self._lock:
                if result.status == "ok":
                    self._latest = result
                else:
                    # Keep last-good advisory; annotate why this attempt failed.
                    kept = AdvisoryView(**self._latest.to_dict())
                    kept.status = result.status
                    kept.rtt_ms = result.rtt_ms
                    kept.raw_text = (
                        f"{kept.raw_text} | last_attempt={result.status}: {result.raw_text}"
                    )
                    # If we have never succeeded, fall through to the failure default.
                    if kept.model_id in ("none",):
                        self._latest = result
                    else:
                        self._latest = kept
                self._busy = False
