import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from config import SEED, MODELS, LOCAL_RESULTS_DIR


def select_confused_pairs() -> List[Tuple[str, str]]:
    """
    Return a list of mental state label pairs that are frequently confused.
    """
    return []


def run_activation_patching(
    model_key: str,
    layer_idx: int,
    pairs: List[Tuple[str, str]],
    output_path: Path,
    seed: int = SEED,
) -> Dict[str, Any]:
    """
    Run activation patching for a set of mental state pairs and write results to disk.
    """
    _ = (model_key, layer_idx, pairs, output_path, seed, np)
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Activation patching with TransformerLens for mental state pairs.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODELS.keys()),
        help="Model key defined in config.MODELS.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer index to patch (e.g., chosen based on probe peaks).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save patching results JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed.",
    )

    args = parser.parse_args()

    if args.output is None:
        default_dir = LOCAL_RESULTS_DIR / "patching"
        default_dir.mkdir(parents=True, exist_ok=True)
        args.output = default_dir / f"patching_{args.model}_layer{args.layer}_seed{args.seed}.json"
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    pairs = select_confused_pairs()
    _ = run_activation_patching(
        model_key=args.model,
        layer_idx=args.layer,
        pairs=pairs,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

