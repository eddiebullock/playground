"""
Diagnostic for A5's all-null family: does the error rate of the B columns come from the
recovery-only shortcut of 100 label shuffles in companion cells, or from the observer?

Four cells of one label-free observer per replicate, as in a recovery all-null family. Each
replicate is tested twice on the same scores: with E* from 100 shuffles and from 1000. Only
the E* columns can differ, so the comparison is paired. A second arm uses a prior scale
calibrated on more draws, to see what a small true B contributes. Synthetic data only.
A diagnostic of the shortcut, not a reroll of the check.

Usage:
    python -m scripts.a5_family_diagnostic --seed 6042 --replicates 2000 --workers 4 \
        --priors 0.473 --output research/reproduction/a5_family_diagnostic.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np  # noqa: E402

from scripts import a5_analyse as a5  # noqa: E402
from scripts import a5_design  # noqa: E402
from scripts import a5_recovery as rec  # noqa: E402

_WORLD = None


def wilson(hits: int, n: int, z: float = 1.959964) -> List[float]:
    """Wilson 95% interval for a binomial rate."""
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(centre - half), float(centre + half)]


def _init(items: Sequence[Any]) -> None:
    global _WORLD
    _WORLD = rec.World(items)


def _one(task: Tuple[float, int, int]) -> Dict[str, Any]:
    prior, seed, number = task
    assert _WORLD is not None
    rng = np.random.default_rng([seed, int(round(prior * 1e6)), number])
    spec = rec.base_spec(prior=prior)
    views = [_WORLD.judgement(_WORLD.scores(spec, rng)[0]) for _ in range(rec.R["companion_cells"] + 1)]
    full = [a5.estimate(j, shuffles=1000) for j in views]
    short = [a5.Estimates(e.parts, e.rows, e.B_row, e.rows["E"] - a5.shuffled_e_rows(j, 100, a5.D["seed"]), e.T, e.margin)
             for e, j in zip(full, views)]
    alpha = a5.D["alpha"]
    out: Dict[str, Any] = {"b_over_t": float(np.mean([np.nanmean(e.B_row) / e.T for e in full]))}
    for name, cells in (("shuffles_1000", full), ("shuffles_100", short)):
        tests = a5.family_tests(cells)
        out[name] = {"any_B": any(t["p_B"] < alpha for t in tests), "any_E": any(t["p_E"] < alpha for t in tests)}
    return out


def arm(rows: Sequence[Dict[str, Any]], name: str) -> Dict[str, Any]:
    n = len(rows)
    out: Dict[str, Any] = {"n": n}
    for key in ("any_B", "any_E"):
        hits = sum(r[name][key] for r in rows)
        out[key] = {"hits": int(hits), "rate": hits / n, "wilson_95": wilson(int(hits), n)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, default=Path("data/eu_emotions_full_manifest.json"))
    parser.add_argument("--lookup", type=Path, default=Path("data/eu_emotions_human_entropy.json"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--priors", type=float, nargs="+", required=True, help="Prior scales of the label-free observer.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    items = rec.real_items(args.manifest, args.lookup)
    out: Dict[str, Any] = {"seed": args.seed, "design_sha256": a5_design.block_hash(), "uses_model_output": False, "arms": {}}
    with multiprocessing.Pool(args.workers, _init, (items,)) as pool:
        for prior in args.priors:
            rows = pool.map(_one, [(prior, args.seed, k) for k in range(args.replicates)], chunksize=20)
            out["arms"][str(prior)] = {"mean_b_over_t": float(np.mean([r["b_over_t"] for r in rows])),
                                       "shuffles_1000": arm(rows, "shuffles_1000"), "shuffles_100": arm(rows, "shuffles_100")}
    args.output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
