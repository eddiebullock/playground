"""
After Step 5 activation extract: build activation RDMs and run RSA vs human
trait-sensitivity RDM (study4 only).

Usage (local, after pull):
  python study4_rmet/scripts/activation_rsa.py \
    --activations_root study4_rmet/results/activations \
    --human_rdm study4_rmet/results/alignment/human_trait_sensitivity_rdm.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

STUDY4_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
import sys

if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from alignment_analyses import compute_rdm_from_vectors, perm_rsa  # noqa: E402


def find_layer_files(model_dir: Path) -> List[Path]:
    return sorted(model_dir.glob("layer*_rmet_seed*.npy"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--activations_root",
        type=Path,
        default=STUDY4_ROOT / "results" / "activations",
    )
    ap.add_argument(
        "--human_rdm",
        type=Path,
        default=STUDY4_ROOT / "results" / "alignment" / "human_trait_sensitivity_rdm.npy",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=STUDY4_ROOT / "results" / "alignment",
    )
    ap.add_argument("--tag", default="full", help="activations subdir tag (full|smoke3)")
    ap.add_argument("--n_perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    human = np.load(args.human_rdm)
    args.outdir.mkdir(parents=True, exist_ok=True)
    summary: Dict = {"human_rdm": str(args.human_rdm), "tag": args.tag, "models": {}}

    for model in ("qwen3vl", "gemma4", "molmo2"):
        model_dir = args.activations_root / model / args.tag
        if not model_dir.exists():
            summary["models"][model] = {"status": "missing", "path": str(model_dir)}
            continue
        layers = find_layer_files(model_dir)
        if not layers:
            summary["models"][model] = {"status": "no_layer_files", "path": str(model_dir)}
            continue
        per_layer = []
        for path in layers:
            acts = np.load(path)
            # acts may be (n, 1, d) or (n, d)
            if acts.ndim == 3:
                acts = acts.reshape(acts.shape[0], -1)
            rdm = compute_rdm_from_vectors(acts)
            layer_name = path.stem.split("_")[0]  # layer12
            rdm_path = args.outdir / f"model_{model}_{layer_name}_activation_rdm.npy"
            np.save(rdm_path, rdm)
            rsa = perm_rsa(human, rdm, n_perm=args.n_perm, seed=args.seed)
            per_layer.append(
                {
                    "layer_file": str(path),
                    "rdm_path": str(rdm_path),
                    "shape": list(acts.shape),
                    "rsa": rsa,
                }
            )
        # Pick mid-depth layer if available (second of three fractions), else best |rho|
        headline = max(per_layer, key=lambda d: abs(d["rsa"]["rho"]))
        summary["models"][model] = {
            "status": "ok",
            "n_layers": len(per_layer),
            "layers": per_layer,
            "headline_max_abs_rho": headline,
        }

    out_json = args.outdir / "a2_activation_rsa_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
