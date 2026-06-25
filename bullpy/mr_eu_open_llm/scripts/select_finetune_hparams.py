#!/usr/bin/env python3
"""Pick best LoRA hyperparameters from hparam sweep finetune_metrics.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import BEST_MODEL_KEY, LOCAL_RESULTS_DIR


def load_metrics(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def select_best_sweep(sweep_root: Path, model_key: str) -> Dict[str, Any]:
    pattern = sweep_root / model_key
    candidates: List[Dict[str, Any]] = []
    for metrics_path in sorted(pattern.rglob("finetune_metrics.json")):
        m = load_metrics(metrics_path)
        if m is None or m.get("model_key") != model_key:
            continue
        val_acc = m.get("best_val_accuracy")
        if val_acc is None:
            continue
        candidates.append(
            {
                "val_accuracy": float(val_acc),
                "learning_rate": m.get("learning_rate"),
                "lora_r": (m.get("lora") or {}).get("r"),
                "lora_alpha": (m.get("lora") or {}).get("lora_alpha"),
                "run_dir": str(metrics_path.parent),
                "adapter_final": m.get("adapter_final"),
            }
        )
    if not candidates:
        raise RuntimeError(f"No finetune_metrics.json under {pattern}")
    candidates.sort(key=lambda x: (-x["val_accuracy"], str(x["learning_rate"]), -int(x["lora_r"] or 0)))
    best = candidates[0]
    return {"model_key": model_key, "best": best, "candidates": candidates}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=BEST_MODEL_KEY or "gemma4")
    ap.add_argument(
        "--sweep_root",
        type=Path,
        default=LOCAL_RESULTS_DIR / "finetune" / "hparam_sweep",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=LOCAL_RESULTS_DIR / "stats" / "best_finetune_hparams.json",
    )
    args = ap.parse_args()
    summary = select_best_sweep(args.sweep_root, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
