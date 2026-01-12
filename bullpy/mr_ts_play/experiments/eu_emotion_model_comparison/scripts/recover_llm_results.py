#!/usr/bin/env python3
"""
Recover LLM results from checkpoint or cache files.

If LLM evaluation was interrupted, this script can recover partial results
and generate metrics from what was completed.
"""

import json
import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.eu_emotion_model_comparison.evaluation.metrics import compute_metrics, save_per_emotion_results, save_confusion_matrix
from experiments.eu_emotion_model_comparison.evaluation.metrics import compute_confusion_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def recover_llm_results(model_name: str, results_dir: Path):
    """Recover and process LLM results from checkpoint or cache."""
    model_output_dir = results_dir / model_name.lower().replace('-', '_')
    
    if not model_output_dir.exists():
        logger.error(f"Model directory not found: {model_output_dir}")
        return False
    
    # Check for checkpoint file
    checkpoint_file = model_output_dir / "predictions_checkpoint.json"
    predictions_file = model_output_dir / "predictions.json"
    
    predictions = None
    
    # Try checkpoint first (partial results)
    if checkpoint_file.exists():
        logger.info(f"Found checkpoint file: {checkpoint_file}")
        with open(checkpoint_file, 'r') as f:
            predictions = json.load(f)
        logger.info(f"Recovered {len(predictions)} predictions from checkpoint")
    
    # Try full predictions file
    elif predictions_file.exists():
        logger.info(f"Found predictions file: {predictions_file}")
        with open(predictions_file, 'r') as f:
            predictions = json.load(f)
        logger.info(f"Loaded {len(predictions)} predictions")
    
    # Try to recover from cache
    else:
        logger.info("No predictions file found. Checking cache...")
        cache_dir = Path("experiments/eu_emotion_model_comparison/data/llm_cache")
        if cache_dir.exists():
            cache_files = list(cache_dir.glob(f"{model_name.replace('-', '_')}_*.json"))
            if cache_files:
                logger.info(f"Found {len(cache_files)} cache files")
                # Could reconstruct predictions from cache, but would need trial definitions
                logger.warning("Cache files found but need trial definitions to reconstruct predictions")
                return False
    
    if predictions is None or len(predictions) == 0:
        logger.error("No predictions found to recover")
        return False
    
    # Compute metrics
    logger.info("Computing metrics from recovered predictions...")
    metrics = compute_metrics(predictions)
    
    # Save results
    logger.info("Saving recovered results...")
    
    # Save predictions
    with open(predictions_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    # Save metrics
    metrics_file = model_output_dir / "metrics.json"
    metrics_json = json.loads(json.dumps(metrics, default=str))
    with open(metrics_file, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    # Save per-emotion results
    per_emotion_file = model_output_dir / "per_emotion_results.csv"
    save_per_emotion_results(
        metrics['per_emotion_metrics'],
        per_emotion_file,
        model_name,
    )
    
    # Save confusion matrix
    confusion_matrix = compute_confusion_matrix(predictions, normalize=True)
    confusion_file = model_output_dir / "confusion_matrix.png"
    save_confusion_matrix(confusion_matrix, confusion_file, model_name)
    confusion_matrix.to_csv(model_output_dir / "confusion_matrix.csv")
    
    logger.info(f"✅ Recovered results for {model_name}")
    logger.info(f"   Accuracy: {metrics['overall_accuracy']:.4f}")
    logger.info(f"   Trials: {len(predictions)}")
    
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Recover LLM results from checkpoint or cache")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name (e.g., gpt-4o-mini)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/eu_emotion_model_comparison',
        help='Results directory'
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)
    
    success = recover_llm_results(args.model, results_dir)
    
    if success:
        logger.info("Recovery successful!")
    else:
        logger.error("Recovery failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
