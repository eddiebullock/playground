"""
Evaluation metrics for model comparison.

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
    """Compute overall accuracy."""
    if not predictions:
        return 0.0
    
    correct = sum(1 for p in predictions if p['correct_label'] == p['predicted_label'])
    return correct / len(predictions)


def compute_per_emotion_accuracy(predictions: List[Dict]) -> Dict[str, Dict]:
    """
    Compute detailed per-emotion metrics.
    
    Returns:
        Dictionary with emotion -> {accuracy, precision, recall, f1, support, confusion}
    """
    emotion_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'predicted': defaultdict(int)})
    
    for pred in predictions:
        true_label = pred['correct_label']
        pred_label = pred['predicted_label']
        
        emotion_stats[true_label]['total'] += 1
        emotion_stats[true_label]['predicted'][pred_label] += 1
        
        if true_label == pred_label:
            emotion_stats[true_label]['correct'] += 1
    
    per_emotion = {}
    for emotion, stats in emotion_stats.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        
        # Precision: of all predictions for this emotion, how many were correct?
        total_predicted_as_emotion = sum(
            s['predicted'].get(emotion, 0) for s in emotion_stats.values()
        )
        precision = stats['correct'] / total_predicted_as_emotion if total_predicted_as_emotion > 0 else 0.0
        
        # Recall: of all true instances, how many were correctly predicted?
        recall = accuracy  # Same as accuracy for per-emotion
        
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Most common confusion
        confusion_dict = dict(stats['predicted'])
        confusion_dict.pop(emotion, None)  # Remove correct predictions
        most_confused = max(confusion_dict.items(), key=lambda x: x[1]) if confusion_dict else (None, 0)
        
        per_emotion[emotion] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': stats['total'],
            'most_confused': most_confused[0] if most_confused[0] else None,
            'confusion_count': most_confused[1] if most_confused[0] else 0,
        }
    
    return per_emotion


def compute_confusion_matrix(predictions: List[Dict], normalize: bool = True) -> pd.DataFrame:
    """Compute confusion matrix."""
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


def compute_top_k_accuracy(predictions: List[Dict], k: int = 2) -> float:
    """
    Compute top-k accuracy if scores are available.
    
    Args:
        predictions: List of predictions with 'scores' dict
        k: Top k predictions to consider
    
    Returns:
        Top-k accuracy
    """
    if not predictions:
        return 0.0
    
    correct = 0
    for pred in predictions:
        if 'scores' not in pred:
            continue
        
        scores = pred['scores']
        correct_label = pred['correct_label']
        
        # Get top k labels
        sorted_labels = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_k_labels = [label for label, _ in sorted_labels[:k]]
        
        if correct_label in top_k_labels:
            correct += 1
    
    return correct / len(predictions) if predictions else 0.0


def compute_metrics(predictions: List[Dict]) -> Dict:
    """Compute comprehensive evaluation metrics."""
    metrics = {
        'overall_accuracy': compute_accuracy(predictions),
        'per_emotion_metrics': compute_per_emotion_accuracy(predictions),
        'num_trials': len(predictions),
    }
    
    # Confusion matrix
    confusion_matrix = compute_confusion_matrix(predictions, normalize=True)
    metrics['confusion_matrix'] = confusion_matrix.to_dict()
    
    # Top-2 accuracy if scores available
    if any('scores' in p for p in predictions):
        metrics['top_2_accuracy'] = compute_top_k_accuracy(predictions, k=2)
    
    return metrics


def save_per_emotion_results(
    per_emotion_metrics: Dict[str, Dict],
    output_file: Path,
    model_name: str,
):
    """Save per-emotion results to CSV."""
    rows = []
    for emotion, metrics in sorted(per_emotion_metrics.items()):
        rows.append({
            'emotion': emotion,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'support': metrics['support'],
            'most_confused': metrics['most_confused'],
            'confusion_count': metrics['confusion_count'],
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    logger.info(f"Saved per-emotion results for {model_name} to {output_file}")


def save_confusion_matrix(
    confusion_matrix: pd.DataFrame,
    output_file: Path,
    model_name: str,
):
    """Save confusion matrix as PNG."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            cbar_kws={'label': 'Proportion'},
        )
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved confusion matrix for {model_name} to {output_file}")
    except ImportError:
        logger.warning("matplotlib/seaborn not available, skipping confusion matrix plot")
        # Save as CSV instead
        confusion_matrix.to_csv(output_file.with_suffix('.csv'))
    except Exception as e:
        logger.error(f"Error saving confusion matrix: {e}")
