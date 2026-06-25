"""
Build or document human RDM for RSA (Study 2).

If data/human_rdm.npy is missing, writes a placeholder manifest and exits non-zero
unless --allow_placeholder is set (dev only).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from config import LOCAL_DATA_DIR, PROJECT_ROOT


def main() -> None:
    ap = argparse.ArgumentParser(description="Human RDM builder (pending behavioural study).")
    ap.add_argument(
        "--output",
        type=Path,
        default=LOCAL_DATA_DIR / "human_rdm.npy",
    )
    ap.add_argument("--allow_placeholder", action="store_true")
    ap.add_argument("--n_stimuli", type=int, default=118)
    args = ap.parse_args()

    if args.output.exists() and not args.allow_placeholder:
        print(f"Human RDM already exists: {args.output}")
        return

    meta_path = PROJECT_ROOT / "data" / "human_rdm_source.json"
    meta = {
        "status": "pending",
        "note": "Collect 4AFC human data or import published RDM before confirmatory RSA.",
        "n_stimuli": args.n_stimuli,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    if args.allow_placeholder:
        rng = np.random.default_rng(42)
        n = args.n_stimuli
        fake = rng.standard_normal((n, 64))
        rdm = 1.0 - np.corrcoef(fake)
        np.fill_diagonal(rdm, 0.0)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, rdm.astype(np.float32))
        print(f"Wrote placeholder RDM to {args.output} (dev only).")
    else:
        raise SystemExit(
            f"No human RDM at {args.output}. Run a human study or pass --allow_placeholder for dev."
        )


if __name__ == "__main__":
    main()
