"""
Evaluation metrics for LLM-augmented emotion recognition.

Computes accuracy, per-emotion accuracy, confusion matrix, and other metrics.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def compute_accuracy(predictions: List[Dict]) -> float:
    """
    Compute overall accuracy.
    
    Args:
        predictions: List of prediction dictionaries with 'correct_label' and 'predicted_label'
    
    Returns:
        Overall accuracy (0.0 to 1.0)
    """
    if not predictions:
        return 0.0
    
    correct = sum(1 for p in predictions if p['correct_label'] == p['predicted_label'])
    return correct / len(predictions)


def compute_per_emotion_accuracy(predictions: List[Dict]) -> Dict[str, float]:
    """
    Compute accuracy for each emotion label.
    
    Args:
        predictions: List of prediction dictionaries
    
    Returns:
        Dictionary mapping emotion labels to accuracy scores
    """
    emotion_counts = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for pred in predictions:
        emotion = pred['correct_label']
        emotion_counts[emotion]['total'] += 1
        if pred['correct_label'] == pred['predicted_label']:
            emotion_counts[emotion]['correct'] += 1
    
    return {
        emotion: counts['correct'] / counts['total'] if counts['total'] > 0 else 0.0
        for emotion, counts in emotion_counts.items()
    }


def compute_confusion_matrix(predictions: List[Dict], normalize: bool = True) -> pd.DataFrame:
    """
    Compute confusion matrix for predictions.
    
    Args:
        predictions: List of prediction dictionaries
        normalize: If True, normalize rows to show proportions
    
    Returns:
        DataFrame with confusion matrix
    """
    all_labels = set()
    for pred in predictions:
        all_labels.add(pred['correct_label'])
        all_labels.add(pred['predicted_label'])
    
    all_labels = sorted(list(all_labels))
    label_to_idx = {label: i for i, label in enumerate(all_labels)}
    
    n = len(all_labels)
    cm = np.zeros((n, n), dtype=float)
    
    for pred in predictions:
        true_idx = label_to_idx[pred['correct_label']]
        pred_idx = label_to_idx[pred['predicted_label']]
        cm[true_idx, pred_idx] += 1
    
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm / row_sums
    
    cm_df = pd.DataFrame(
        cm,
        index=all_labels,
        columns=all_labels,
    )
    
    return cm_df


def compute_metrics(predictions: List[Dict]) -> Dict:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        predictions: List of prediction dictionaries
    
    Returns:
        Dictionary with all computed metrics
    """
    metrics = {
        'overall_accuracy': compute_accuracy(predictions),
        'per_emotion_accuracy': compute_per_emotion_accuracy(predictions),
        'num_trials': len(predictions),
    }
    
    # Confusion matrix
    confusion_matrix = compute_confusion_matrix(predictions, normalize=True)
    metrics['confusion_matrix'] = confusion_matrix.to_dict()
    
    return metrics


def save_results(
    predictions: List[Dict],
    metrics: Dict,
    output_dir: Path,
    condition_name: str = "results",
):
    """
    Save evaluation results to files.
    
    Args:
        predictions: List of prediction dictionaries
        metrics: Metrics dictionary
        output_dir: Output directory
        condition_name: Name for this condition (e.g., "clip_only", "llm_augmented")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    predictions_file = output_dir / "predictions.json"
    with open(predictions_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    # Save metrics
    metrics_file = output_dir / "metrics.json"
    # Convert numpy types to native Python types for JSON
    metrics_json = json.loads(json.dumps(metrics, default=str))
    with open(metrics_file, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    # Save confusion matrix as CSV
    if 'confusion_matrix' in metrics:
        cm_df = pd.DataFrame(metrics['confusion_matrix'])
        cm_file = output_dir / "confusion_matrix.csv"
        cm_df.to_csv(cm_file)
    
    # Save per-emotion accuracy as CSV
    if 'per_emotion_accuracy' in metrics:
        per_emotion_df = pd.DataFrame(
            list(metrics['per_emotion_accuracy'].items()),
            columns=['emotion', 'accuracy']
        )
        per_emotion_file = output_dir / "per_emotion_accuracy.csv"
        per_emotion_df.to_csv(per_emotion_file, index=False)
    
    # Save summary text
    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Results for: {condition_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}\n")
        f.write(f"Number of Trials: {metrics['num_trials']}\n\n")
        f.write("Per-Emotion Accuracy:\n")
        f.write("-" * 50 + "\n")
        for emotion, acc in sorted(metrics['per_emotion_accuracy'].items()):
            f.write(f"  {emotion:30s}: {acc:.4f}\n")
    
    logger.info(f"Results saved to {output_dir}")


