"""Small structured logger. Job preambles also echo args in bash."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("esvlm")
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def echo_run_args(args: dict[str, Any], logger: logging.Logger | None = None) -> None:
    """Print every run argument. A previous job silently ran the smoke path
    because MODE was ignored; fail-open visibility is the fix."""

    log = logger or setup_logging()
    payload = {
        "utc": datetime.now(timezone.utc).isoformat(),
        **{k: _jsonable(v) for k, v in args.items()},
    }
    log.info("run_args %s", json.dumps(payload, sort_keys=True))


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)
