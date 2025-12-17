"""
Evaluation metrics for emotion recognition.
Includes top-k accuracy for few-shot learning scenarios.
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from typing import List, Tuple, Dict


def top_k_accuracy(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    k: int = 5,
) -> float:
    """
    Compute top-k accuracy.
    
    Args:
        y_true: (N,) true labels
        y_pred_proba: (N, num_classes) predicted probabilities
        k: Number of top predictions to consider
    Returns:
        accuracy: Top-k accuracy
    """
    # Get top-k predictions
    top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
    
    # Check if true label is in top-k
    correct = 0
    for i, true_label in enumerate(y_true):
        if true_label in top_k_preds[i]:
            correct += 1
    
    return correct / len(y_true)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: List[str],
    top_k_values: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: (N,) true labels
        y_pred: (N,) predicted labels
        y_pred_proba: (N, num_classes) predicted probabilities
        class_names: List of class names
        top_k_values: Tuple of k values for top-k accuracy
    Returns:
        metrics: Dictionary of metric names and values
    """
    metrics = {}
    
    # Top-1 accuracy (standard accuracy)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Top-k accuracy
    for k in top_k_values:
        if k <= y_pred_proba.shape[1]:  # Only if k <= num_classes
            metrics[f'top_{k}_accuracy'] = top_k_accuracy(y_true, y_pred_proba, k)
    
    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    
    # Macro and weighted averages
    metrics['f1_macro'] = report['macro avg']['f1-score']
    metrics['f1_weighted'] = report['weighted avg']['f1-score']
    metrics['precision_macro'] = report['macro avg']['precision']
    metrics['recall_macro'] = report['macro avg']['recall']
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """Print metrics in a readable format."""
    print(f"\n{title}")
    print("-" * 60)
    
    # Accuracy metrics
    if 'accuracy' in metrics:
        print(f"Top-1 Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    
    for k in [5, 10, 20, 50]:
        key = f'top_{k}_accuracy'
        if key in metrics:
            print(f"Top-{k} Accuracy:  {metrics[key]:.4f} ({metrics[key]*100:.2f}%)")
    
    # F1 scores
    if 'f1_macro' in metrics:
        print(f"\nF1 Macro:        {metrics['f1_macro']:.4f}")
    if 'f1_weighted' in metrics:
        print(f"F1 Weighted:     {metrics['f1_weighted']:.4f}")
    
    print("-" * 60)


def compare_to_random(num_classes: int, top_k_values: Tuple[int, ...] = (1, 5, 10)):
    """
    Compute random baseline performance for comparison.
    
    Args:
        num_classes: Number of classes
        top_k_values: Tuple of k values
    Returns:
        random_metrics: Dictionary of random baseline metrics
    """
    random_metrics = {}
    
    # Top-1 random: 1/num_classes
    random_metrics['accuracy'] = 1.0 / num_classes
    
    # Top-k random: k/num_classes (if k <= num_classes)
    for k in top_k_values:
        if k <= num_classes:
            random_metrics[f'top_{k}_accuracy'] = min(k / num_classes, 1.0)
    
    return random_metrics

