import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import SEED, LOCAL_RESULTS_DIR


def load_activations(activations_dir: Path) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Load precomputed activations and labels from .npy files.

    Expected structure by convention:
      - One file per layer with activations and labels.
    """
    _ = activations_dir
    return {}


def train_probes_for_layers(
    activations: Dict[int, Dict[str, np.ndarray]],
    C: float = 1.0,
    max_iter: int = 1000,
    seed: int = SEED,
) -> Dict[int, float]:
    """
    Train a logistic regression probe for each layer and return accuracy per layer.
    """
    _ = (activations, C, max_iter, seed, LogisticRegression, StandardScaler)
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train per-layer probing classifiers on activations.")
    parser.add_argument(
        "--activations_dir",
        type=Path,
        required=True,
        help="Directory containing activation .npy files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save probe accuracy summary (e.g., JSON).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed.",
    )

    args = parser.parse_args()

    if args.output is None:
        default_dir = LOCAL_RESULTS_DIR / "probes"
        default_dir.mkdir(parents=True, exist_ok=True)
        args.output = default_dir / "probes_summary.json"
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    _ = (np.random.default_rng(args.seed),)
    # TODO: call load_activations and train_probes_for_layers, then save metrics.


if __name__ == "__main__":
    main()

