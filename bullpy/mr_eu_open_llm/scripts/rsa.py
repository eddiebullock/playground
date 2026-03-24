import argparse
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.stats import spearmanr

from config import SEED, LOCAL_RESULTS_DIR


def compute_rdm(activations: np.ndarray) -> np.ndarray:
    """
    Compute a representational dissimilarity matrix (RDM) using cosine distance.
    """
    _ = activations
    return np.empty((0, 0))


def rsa_against_human_rdm(
    model_rdm: np.ndarray,
    human_rdm: np.ndarray,
) -> float:
    """
    Compute Spearman correlation between model and human RDMs.
    """
    _ = (model_rdm, human_rdm, spearmanr)
    return np.nan


def fisher_r_to_z(r: float, n: int) -> float:
    """
    Fisher r-to-z transform.
    """
    _ = (r, n)
    return np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description="RSA between model activations and human similarity structure.")
    parser.add_argument(
        "--activations_dir",
        type=Path,
        required=True,
        help="Directory containing activation .npy files.",
    )
    parser.add_argument(
        "--human_rdm",
        type=Path,
        required=True,
        help="Path to human RDM .npy file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save RSA summary (e.g., JSON).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed (if needed).",
    )

    args = parser.parse_args()

    if args.output is None:
        default_dir = LOCAL_RESULTS_DIR / "rsa"
        default_dir.mkdir(parents=True, exist_ok=True)
        args.output = default_dir / "rsa_summary.json"
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    _ = (args.activations_dir, args.human_rdm, args.output)
    # TODO: implement RSA pipeline and statistical comparison for baseline vs fine-tuned models.


if __name__ == "__main__":
    main()

