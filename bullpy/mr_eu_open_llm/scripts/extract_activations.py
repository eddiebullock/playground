import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np

from config import SEED, MODELS, DATASETS, LOCAL_RESULTS_DIR


def register_hooks(model: Any) -> Dict[str, Any]:
    """
    Register forward hooks on transformer layers and return a handle structure
    that can be used to collect activations.
    """
    _ = model
    return {}


def extract_activations(
    model_key: str,
    dataset_key: str,
    split: str,
    output_dir: Path,
    seed: int = SEED,
) -> None:
    """
    Run the model in inference mode over the specified dataset split and save
    mean-pooled layer activations as .npy files.
    """
    _ = (model_key, dataset_key, split, output_dir, seed, np)
    output_dir.mkdir(parents=True, exist_ok=True)
    # TODO: implement activation extraction with early/mid/late layer selection.


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract transformer activations for mechanistic interpretability.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODELS.keys()),
        help="Model key defined in config.MODELS.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASETS.keys()),
        help="Dataset key defined in config.DATASETS.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save activation .npy files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed.",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = LOCAL_RESULTS_DIR / "activations" / args.model / args.dataset

    extract_activations(
        model_key=args.model,
        dataset_key=args.dataset,
        split=args.split,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

