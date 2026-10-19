"""Global seeding. Every results file should also record the seed."""

from __future__ import annotations

import os
import random
from typing import Any


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def seed_generation(seed: int, *parts: Any) -> int:
    """Stable per-item seed derived from a base seed and trial keys."""

    material = "|".join(str(p) for p in (seed, *parts))
    derived = (int.from_bytes(material.encode("utf-8"), "little") + seed) % (2**31 - 1)
    seed_everything(derived)
    return derived
