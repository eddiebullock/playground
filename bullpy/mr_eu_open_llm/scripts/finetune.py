import argparse
from pathlib import Path
from typing import Any, Dict

from config import (
    SEED,
    MODELS,
    LORA_DEFAULT,
    TRAINING_DEFAULTS,
    LOCAL_RESULTS_DIR,
)


def setup_lora_config(
    r: int,
    alpha: int,
    target_modules: Dict[str, Any],
    dropout: float,
) -> Dict[str, Any]:
    """
    Prepare a LoRA configuration dict compatible with `peft.LoraConfig`.
    """
    _ = target_modules
    return {
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
    }


def run_finetuning(
    model_key: str,
    train_file: Path,
    val_file: Path,
    output_dir: Path,
    learning_rate: float,
    r: int,
    alpha: int,
    dropout: float,
    seed: int = SEED,
) -> None:
    """
    Run LoRA fine-tuning for a given model and dataset specification.
    """
    _ = (
        model_key,
        train_file,
        val_file,
        output_dir,
        learning_rate,
        r,
        alpha,
        dropout,
        seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    # TODO: implement transformers/peft/accelerate training loop with checkpointing and EU-Emotions validation.


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for multimodal mental state recognition.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODELS.keys()),
        help="Model key defined in config.MODELS.",
    )
    parser.add_argument("--train_file", type=Path, required=True, help="Path to training JSONL file.")
    parser.add_argument("--val_file", type=Path, required=True, help="Path to validation JSONL file.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save checkpoints and metrics.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=LORA_DEFAULT["r"],
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=LORA_DEFAULT["alpha"],
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=LORA_DEFAULT["dropout"],
        help="LoRA dropout.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed.",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        default_dir = LOCAL_RESULTS_DIR / "finetune" / "runs" / args.model
        args.output_dir = default_dir

    run_finetuning(
        model_key=args.model,
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

