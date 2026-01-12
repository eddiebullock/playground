#!/usr/bin/env python3
"""
Create ensemble predictions by combining multiple models.

Supports:
- Average ensemble (mean of scores)
- Weighted ensemble (weighted mean of scores)
- Voting ensemble (majority vote)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from collections import Counter

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.eu_emotion_model_comparison.evaluation.evaluator import ModelEvaluator
from experiments.eu_emotion_model_comparison.evaluation.metrics import (
    compute_metrics,
    save_per_emotion_results,
    save_confusion_matrix,
    compute_confusion_matrix,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_predictions(results_dir: Path, model_name: str) -> List[Dict]:
    """Load predictions for a model."""
    model_dir = results_dir / model_name.lower().replace('-', '_')
    predictions_file = model_dir / "predictions.json"
    
    if not predictions_file.exists():
        logger.warning(f"Predictions not found for {model_name} at {predictions_file}")
        return []
    
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    logger.info(f"Loaded {len(predictions)} predictions for {model_name}")
    return predictions


def create_average_ensemble(
    all_predictions: Dict[str, List[Dict]],
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """Create ensemble by averaging scores across models."""
    if weights is None:
        weights = {model: 1.0 for model in all_predictions.keys()}
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    # Group predictions by trial_id
    trial_predictions = {}
    for model_name, predictions in all_predictions.items():
        for pred in predictions:
            trial_id = pred['trial_id']
            if trial_id not in trial_predictions:
                trial_predictions[trial_id] = {
                    'trial_id': trial_id,
                    'video_path': pred['video_path'],
                    'correct_label': pred['correct_label'],
                    'candidate_labels': pred['candidate_labels'],
                    'scores': {},
                    'model_scores': {},
                }
            
            # Store scores from this model
            weight = weights[model_name]
            for label, score in pred['scores'].items():
                if label not in trial_predictions[trial_id]['scores']:
                    trial_predictions[trial_id]['scores'][label] = 0.0
                trial_predictions[trial_id]['scores'][label] += score * weight
            
            # Store individual model scores for analysis
            trial_predictions[trial_id]['model_scores'][model_name] = pred['scores']
    
    # Create final predictions
    ensemble_predictions = []
    for trial_id, pred in trial_predictions.items():
        # Find label with highest score
        predicted_label = max(pred['scores'].items(), key=lambda x: x[1])[0]
        
        ensemble_predictions.append({
            'trial_id': pred['trial_id'],
            'video_path': pred['video_path'],
            'correct_label': pred['correct_label'],
            'predicted_label': predicted_label,
            'candidate_labels': pred['candidate_labels'],
            'scores': pred['scores'],
            'model_scores': pred['model_scores'],  # For analysis
        })
    
    return ensemble_predictions


def create_voting_ensemble(
    all_predictions: Dict[str, List[Dict]],
) -> List[Dict]:
    """Create ensemble by majority voting."""
    # Group predictions by trial_id
    trial_predictions = {}
    for model_name, predictions in all_predictions.items():
        for pred in predictions:
            trial_id = pred['trial_id']
            if trial_id not in trial_predictions:
                trial_predictions[trial_id] = {
                    'trial_id': trial_id,
                    'video_path': pred['video_path'],
                    'correct_label': pred['correct_label'],
                    'candidate_labels': pred['candidate_labels'],
                    'votes': Counter(),
                    'model_predictions': {},
                }
            
            # Count vote for predicted label
            predicted_label = pred['predicted_label']
            trial_predictions[trial_id]['votes'][predicted_label] += 1
            trial_predictions[trial_id]['model_predictions'][model_name] = predicted_label
    
    # Create final predictions
    ensemble_predictions = []
    for trial_id, pred in trial_predictions.items():
        # Get most common prediction
        if pred['votes']:
            predicted_label = pred['votes'].most_common(1)[0][0]
        else:
            # Fallback: use first candidate
            predicted_label = pred['candidate_labels'][0]
        
        # Create scores from vote counts
        total_votes = sum(pred['votes'].values())
        scores = {label: count / total_votes for label, count in pred['votes'].items()}
        # Add zero scores for labels with no votes
        for label in pred['candidate_labels']:
            if label not in scores:
                scores[label] = 0.0
        
        ensemble_predictions.append({
            'trial_id': pred['trial_id'],
            'video_path': pred['video_path'],
            'correct_label': pred['correct_label'],
            'predicted_label': predicted_label,
            'candidate_labels': pred['candidate_labels'],
            'scores': scores,
            'model_predictions': pred['model_predictions'],  # For analysis
        })
    
    return ensemble_predictions


def main():
    parser = argparse.ArgumentParser(
        description="Create ensemble predictions from multiple models"
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results/eu_emotion_model_comparison',
        help='Results directory containing model predictions'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        required=True,
        help='Model names to include in ensemble'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['average', 'weighted', 'voting'],
        default='average',
        help='Ensemble method'
    )
    parser.add_argument(
        '--weights',
        type=str,
        nargs='+',
        help='Weights for weighted ensemble (same order as --models)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/eu_emotion_model_comparison',
        help='Output directory for ensemble results'
    )
    parser.add_argument(
        '--ensemble_name',
        type=str,
        default=None,
        help='Name for ensemble (default: method_models)'
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    # Load predictions for all models
    logger.info(f"Loading predictions for models: {', '.join(args.models)}")
    all_predictions = {}
    for model_name in args.models:
        predictions = load_predictions(results_dir, model_name)
        if predictions:
            all_predictions[model_name] = predictions
        else:
            logger.warning(f"Skipping {model_name} (no predictions found)")
    
    if not all_predictions:
        logger.error("No predictions loaded! Check model names and results directory.")
        return
    
    logger.info(f"Loaded predictions from {len(all_predictions)} models")
    
    # Create ensemble
    if args.method == 'voting':
        ensemble_predictions = create_voting_ensemble(all_predictions)
        ensemble_name = args.ensemble_name or f"ensemble_voting_{'_'.join(sorted(all_predictions.keys()))}"
    elif args.method == 'weighted':
        if not args.weights or len(args.weights) != len(args.models):
            logger.error("--weights must be provided and match number of models for weighted ensemble")
            return
        
        weights = {model: float(w) for model, w in zip(args.models, args.weights)}
        ensemble_predictions = create_average_ensemble(all_predictions, weights)
        ensemble_name = args.ensemble_name or f"ensemble_weighted_{'_'.join(sorted(all_predictions.keys()))}"
    else:  # average
        ensemble_predictions = create_average_ensemble(all_predictions)
        ensemble_name = args.ensemble_name or f"ensemble_average_{'_'.join(sorted(all_predictions.keys()))}"
    
    logger.info(f"Created {len(ensemble_predictions)} ensemble predictions using {args.method} method")
    
    # Compute metrics
    metrics = compute_metrics(ensemble_predictions)
    logger.info(f"Ensemble accuracy: {metrics['overall_accuracy']:.4f}")
    
    # Save results
    ensemble_output_dir = output_dir / ensemble_name
    ensemble_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    predictions_file = ensemble_output_dir / "predictions.json"
    with open(predictions_file, 'w') as f:
        json.dump(ensemble_predictions, f, indent=2)
    
    # Save metrics
    metrics_file = ensemble_output_dir / "metrics.json"
    metrics_json = json.loads(json.dumps(metrics, default=str))
    with open(metrics_file, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    # Save per-emotion results
    per_emotion_file = ensemble_output_dir / "per_emotion_results.csv"
    save_per_emotion_results(
        metrics['per_emotion_metrics'],
        per_emotion_file,
        ensemble_name,
    )
    
    # Save confusion matrix
    confusion_matrix = compute_confusion_matrix(ensemble_predictions, normalize=True)
    confusion_file = ensemble_output_dir / "confusion_matrix.png"
    save_confusion_matrix(confusion_matrix, confusion_file, ensemble_name)
    confusion_matrix.to_csv(ensemble_output_dir / "confusion_matrix.csv")
    
    logger.info(f"✅ Ensemble results saved to {ensemble_output_dir}")
    logger.info(f"   Accuracy: {metrics['overall_accuracy']:.4f}")
    logger.info(f"   Method: {args.method}")
    logger.info(f"   Models: {', '.join(all_predictions.keys())}")


if __name__ == '__main__':
    main()
