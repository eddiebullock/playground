#!/usr/bin/env python3
"""Thinker HTTP server. Copy the entire thinker/ folder to the HPC GPU node.

Laptop (no GPU, contract test):
    python server.py --mock --host 127.0.0.1 --port 8000

HPC GPU node (real Qwen-VL):
    python server.py --host 127.0.0.1 --port 8000

Do not run the real model on the CSD3 login node. Bind localhost only; the
laptop reaches this via an SSH tunnel (see ../HPC_TRANSFER.md).
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

from model import Inferencer, MockInferencer, QwenInferencer

app = FastAPI(title="Thinker", version="0.1.0")
_inferencer: Optional[Inferencer] = None
_started_at = time.time()


def get_inferencer() -> Inferencer:
    if _inferencer is None:
        raise RuntimeError("inferencer not loaded")
    return _inferencer


@app.get("/health")
def health() -> dict:
    inf = get_inferencer()
    return {
        "ok": True,
        "model_id": inf.model_id,
        "uptime_s": time.time() - _started_at,
    }


@app.post("/think")
async def think(
    image: UploadFile = File(...),
    frame_index: int = Form(-1),
    client_ts: float = Form(0.0),
) -> JSONResponse:
    """Send a camera frame, receive structured HAZARD/ACTION plus timestamps."""
    if not image.content_type or not image.content_type.startswith("image/"):
        # Some clients send application/octet-stream; still try to decode.
        pass
    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="empty image")
    received_ts = time.time()
    adv = get_inferencer().infer(payload)
    body = adv.to_dict()
    body.update(
        {
            "frame_index": frame_index,
            "client_ts": client_ts,
            "received_ts": received_ts,
            "server_ts": time.time(),
        }
    )
    return JSONResponse(body)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Thinker advisory server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--mock",
        action="store_true",
        help="Deterministic fake model. Use on the laptop. Do not use this as the HPC Thinker.",
    )
    p.add_argument(
        "--model-id",
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="Hugging Face id or local path of the Qwen-VL checkpoint.",
    )
    return p.parse_args()


def main() -> None:
    global _inferencer
    args = parse_args()
    if args.mock:
        _inferencer = MockInferencer()
        print("[thinker] MOCK mode — not Qwen-VL", flush=True)
    else:
        _inferencer = QwenInferencer(model_id=args.model_id)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
