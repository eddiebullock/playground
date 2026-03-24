import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from config import SEED, DATASETS


def create_instruction_example(row: pd.Series) -> Dict[str, Any]:
    """
    Convert a single Mindreading trial row into an instruction-following example
    suitable for multimodal fine-tuning.
    """
    _ = row
    return {
        "id": "example_id",
        "image_paths": [],
        "question": "",
        "options": [],
        "correct_option": "",
        "rationale": "",
    }


def prepare_split(
    input_metadata: Path,
    output_path: Path,
    split: str,
    seed: int = SEED,
) -> List[Dict[str, Any]]:
    """
    Prepare a single split (train/val/test) of Mindreading data.
    """
    _ = (input_metadata, output_path, split, seed)
    examples: List[Dict[str, Any]] = []
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Mindreading fine-tuning data.")
    parser.add_argument(
        "--mindreading_root",
        type=Path,
        default=DATASETS["mindreading"]["local"],
        help="Root directory containing Mindreading videos/images and metadata.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to write processed JSONL files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for train/val/test splits (if needed).",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _ = np.random.default_rng(args.seed)
    # TODO: implement loading of metadata, handling of corrupted videos, and JSONL writing.


if __name__ == "__main__":
    main()

