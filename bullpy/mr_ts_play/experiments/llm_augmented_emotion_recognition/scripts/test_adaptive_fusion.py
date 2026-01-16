#!/usr/bin/env python3
"""
Test Adaptive Fusion: Emotion-Specific Fusion Weights

This script implements the adaptive fusion study:
1. Learn fusion weights per emotion on validation set
2. Test fusion on held-out test set
3. Report both validation and test accuracy

Following proper ML practice: learn on validation, test on test set.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def normalize_label(label: str) -> str:
    """Normalize emotion label to lowercase."""
    return label.lower().strip()


def load_predictions(predictions_file: Path) -> List[Dict]:
    """Load predictions from JSON file."""
    with open(predictions_file, 'r') as f:
        data = json.load(f)
    
    # Handle both formats: {"predictions": [...]} and [...]
    if isinstance(data, dict) and 'predictions' in data:
        return data['predictions']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unexpected format in {predictions_file}")


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Normalize scores to [0, 1] range using softmax."""
    values = np.array(list(scores.values()))
    exp_values = np.exp(values - np.max(values))  # Numerical stability
    normalized = exp_values / exp_values.sum()
    return dict(zip(scores.keys(), normalized))


def normalize_path(path: str) -> str:
    """Normalize path for matching (remove leading/trailing slashes, normalize separators)."""
    if not path:
        return ''
    # Normalize separators and remove leading/trailing whitespace
    path = path.replace('\\', '/').strip()
    # Remove leading slash if present
    if path.startswith('/'):
        path = path[1:]
    return path


def evaluate_fusion(
    predictions: List[Dict],
    clip_predictions: Dict[str, Dict],
    llm_predictions: Dict[str, Dict],
    emotion_weights: Dict[str, Dict[str, float]],
    normalize: bool = True
) -> Tuple[float, Dict[str, Dict]]:
    """
    Evaluate fusion with emotion-specific weights.
    
    Returns:
        overall_accuracy: Overall accuracy
        per_emotion_metrics: Dict with accuracy per emotion
    """
    correct = 0
    total = 0
    per_emotion = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for pred in predictions:
        trial_id = pred['trial_id']
        correct_label = normalize_label(pred['correct_label'])
        candidate_labels = [normalize_label(l) for l in pred['candidate_labels']]
        
        # Get predictions from both models
        clip_pred = clip_predictions.get(trial_id)
        llm_pred = llm_predictions.get(trial_id)
        
        if not clip_pred or not llm_pred:
            logger.warning(f"Missing predictions for {trial_id}")
            continue
        
        # Get fusion weights for this emotion
        weights = emotion_weights.get(correct_label, {'clip': 0.5, 'llm': 0.5})
        
        # Get scores for candidate labels
        clip_scores = clip_pred.get('scores', {})
        llm_scores = llm_pred.get('scores', {})
        
        # Normalize if requested
        if normalize:
            clip_scores = normalize_scores(clip_scores)
            llm_scores = normalize_scores(llm_scores)
        
        # Fuse scores
        fused_scores = {}
        for label in candidate_labels:
            clip_score = clip_scores.get(label, 0.0)
            llm_score = llm_scores.get(label, 0.0)
            fused_scores[label] = weights['clip'] * clip_score + weights['llm'] * llm_score
        
        # Predict label with highest fused score
        predicted_label = normalize_label(max(fused_scores.items(), key=lambda x: x[1])[0])
        
        # Check if correct
        if predicted_label == correct_label:
            correct += 1
            per_emotion[correct_label]['correct'] += 1
        total += 1
        per_emotion[correct_label]['total'] += 1
    
    # Calculate per-emotion accuracy
    per_emotion_metrics = {}
    for emotion, counts in per_emotion.items():
        if counts['total'] > 0:
            per_emotion_metrics[emotion] = {
                'accuracy': counts['correct'] / counts['total'],
                'correct': counts['correct'],
                'total': counts['total']
            }
    
    overall_accuracy = correct / total if total > 0 else 0.0
    return overall_accuracy, per_emotion_metrics


def learn_fusion_weights(
    val_predictions: List[Dict],
    clip_predictions: Dict[str, Dict],
    llm_predictions: Dict[str, Dict],
    weight_range: np.ndarray = np.arange(0.1, 1.0, 0.1),
    normalize: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Learn optimal fusion weights per emotion on validation set.
    
    Args:
        val_predictions: Validation set predictions
        clip_predictions: CLIP predictions (trial_id -> prediction dict)
        llm_predictions: LLM predictions (trial_id -> prediction dict)
        weight_range: Range of clip weights to try (llm_weight = 1 - clip_weight)
        normalize: Whether to normalize scores before fusion
    
    Returns:
        emotion_weights: Dict mapping emotion -> {'clip': weight, 'llm': weight}
    """
    logger.info("Learning fusion weights per emotion on validation set...")
    
    # Group predictions by emotion
    emotion_samples = defaultdict(list)
    for pred in val_predictions:
        emotion = normalize_label(pred['correct_label'])
        emotion_samples[emotion].append(pred)
    
    emotion_weights = {}
    
    for emotion, samples in emotion_samples.items():
        logger.info(f"  Learning weights for '{emotion}' ({len(samples)} samples)...")
        
        best_weights = {'clip': 0.5, 'llm': 0.5}  # Default to 50/50
        best_acc = 0.0
        
        if len(samples) == 0:
            logger.warning(f"    No samples found for '{emotion}', using default 50/50 weights")
            emotion_weights[emotion] = best_weights
            continue
        
        # Try different weight combinations
        for clip_weight in weight_range:
            llm_weight = 1.0 - clip_weight
            weights = {'clip': clip_weight, 'llm': llm_weight}
            
            # Evaluate fusion with these weights
            correct = 0
            total = 0
            
            for pred in samples:
                trial_id = pred['trial_id']
                correct_label = normalize_label(pred['correct_label'])
                candidate_labels = [normalize_label(l) for l in pred['candidate_labels']]
                
                clip_pred = clip_predictions.get(trial_id)
                llm_pred = llm_predictions.get(trial_id)
                
                if not clip_pred or not llm_pred:
                    continue
                
                # Get scores
                clip_scores = clip_pred.get('scores', {})
                llm_scores = llm_pred.get('scores', {})
                
                # Normalize if requested
                if normalize:
                    clip_scores = normalize_scores(clip_scores)
                    llm_scores = normalize_scores(llm_scores)
                
                # Fuse scores
                fused_scores = {}
                for label in candidate_labels:
                    clip_score = clip_scores.get(label, 0.0)
                    llm_score = llm_scores.get(label, 0.0)
                    fused_scores[label] = clip_weight * clip_score + llm_weight * llm_score
                
                # Predict
                predicted_label = normalize_label(max(fused_scores.items(), key=lambda x: x[1])[0])
                
                if predicted_label == correct_label:
                    correct += 1
                total += 1
            
            acc = correct / total if total > 0 else 0.0
            
            if acc > best_acc:
                best_acc = acc
                best_weights = weights
        
        emotion_weights[emotion] = best_weights
        logger.info(f"    Best weights: CLIP={best_weights['clip']:.2f}, LLM={best_weights['llm']:.2f}, Acc={best_acc:.1%}")
    
    return emotion_weights


def main():
    parser = argparse.ArgumentParser(
        description="Test adaptive fusion with emotion-specific weights"
    )
    parser.add_argument(
        '--clip_predictions',
        type=str,
        default='results/eu_emotion_model_comparison/clip_finetuned/predictions.json',
        help='Path to CLIP predictions JSON file'
    )
    parser.add_argument(
        '--llm_predictions',
        type=str,
        default='results/llm_only_eu_emotion_google/results.json',
        help='Path to LLM predictions JSON file'
    )
    parser.add_argument(
        '--val_trials',
        type=str,
        default='data/trial_definitions/eu_emotion_val.json',
        help='Path to validation trial definitions'
    )
    parser.add_argument(
        '--test_trials',
        type=str,
        default='data/trial_definitions/eu_emotion_test.json',
        help='Path to test trial definitions'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/adaptive_fusion_emotion_specific',
        help='Output directory for results'
    )
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='Normalize scores before fusion (softmax)'
    )
    parser.add_argument(
        '--weight_step',
        type=float,
        default=0.1,
        help='Step size for weight search (default: 0.1)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("Adaptive Fusion Study: Emotion-Specific Weights")
    logger.info("="*80)
    logger.info(f"CLIP predictions: {args.clip_predictions}")
    logger.info(f"LLM predictions: {args.llm_predictions}")
    logger.info(f"Validation trials: {args.val_trials}")
    logger.info(f"Test trials: {args.test_trials}")
    logger.info(f"Normalize scores: {args.normalize}")
    logger.info("")
    
    # Load trial definitions
    logger.info("Loading trial definitions...")
    with open(args.val_trials, 'r') as f:
        val_data = json.load(f)
    with open(args.test_trials, 'r') as f:
        test_data = json.load(f)
    
    # Handle both formats: {"trials": [...]} and [...]
    if isinstance(val_data, dict) and 'trials' in val_data:
        val_trials = val_data['trials']
    elif isinstance(val_data, list):
        val_trials = val_data
    else:
        raise ValueError(f"Unexpected format in {args.val_trials}")
    
    if isinstance(test_data, dict) and 'trials' in test_data:
        test_trials = test_data['trials']
    elif isinstance(test_data, list):
        test_trials = test_data
    else:
        raise ValueError(f"Unexpected format in {args.test_trials}")
    
    logger.info(f"Validation trials: {len(val_trials)}")
    logger.info(f"Test trials: {len(test_trials)}")
    logger.info("")
    
    # Load predictions
    logger.info("Loading predictions...")
    clip_predictions_raw = load_predictions(Path(args.clip_predictions))
    llm_predictions_raw = load_predictions(Path(args.llm_predictions))
    
    # Convert to dict keyed by trial_id
    clip_predictions = {pred['trial_id']: pred for pred in clip_predictions_raw}
    llm_predictions = {pred['trial_id']: pred for pred in llm_predictions_raw}
    
    # Also create mapping by stimulus_path (for cases where trial_ids don't match)
    clip_predictions_by_path = {}
    llm_predictions_by_path = {}
    for pred in clip_predictions_raw:
        path = pred.get('video_path') or pred.get('stimulus_path', '')
        if path:
            normalized = normalize_path(path)
            clip_predictions_by_path[normalized] = pred
    for pred in llm_predictions_raw:
        path = pred.get('video_path') or pred.get('stimulus_path', '')
        if path:
            normalized = normalize_path(path)
            llm_predictions_by_path[normalized] = pred
    
    logger.info(f"CLIP predictions: {len(clip_predictions)}")
    logger.info(f"LLM predictions: {len(llm_predictions)}")
    logger.info("")
    
    # Filter predictions to validation/test sets
    # Try matching by trial_id first, then by stimulus_path
    val_clip_preds = {}
    val_llm_preds = {}
    test_clip_preds = {}
    test_llm_preds = {}
    
    for trial in val_trials:
        trial_id = trial.get('trial_id')
        stimulus_path = trial.get('stimulus_path', '')
        normalized_path = normalize_path(stimulus_path) if stimulus_path else ''
        
        # Try trial_id first
        if trial_id and trial_id in clip_predictions:
            val_clip_preds[trial_id] = clip_predictions[trial_id]
        elif normalized_path and normalized_path in clip_predictions_by_path:
            pred = clip_predictions_by_path[normalized_path]
            val_clip_preds[trial_id or normalized_path] = pred
        
        if trial_id and trial_id in llm_predictions:
            val_llm_preds[trial_id] = llm_predictions[trial_id]
        elif normalized_path and normalized_path in llm_predictions_by_path:
            pred = llm_predictions_by_path[normalized_path]
            val_llm_preds[trial_id or normalized_path] = pred
    
    for trial in test_trials:
        trial_id = trial.get('trial_id')
        stimulus_path = trial.get('stimulus_path', '')
        normalized_path = normalize_path(stimulus_path) if stimulus_path else ''
        
        # Try trial_id first
        if trial_id and trial_id in clip_predictions:
            test_clip_preds[trial_id] = clip_predictions[trial_id]
        elif normalized_path and normalized_path in clip_predictions_by_path:
            pred = clip_predictions_by_path[normalized_path]
            test_clip_preds[trial_id or normalized_path] = pred
        
        if trial_id and trial_id in llm_predictions:
            test_llm_preds[trial_id] = llm_predictions[trial_id]
        elif normalized_path and normalized_path in llm_predictions_by_path:
            pred = llm_predictions_by_path[normalized_path]
            test_llm_preds[trial_id or normalized_path] = pred
    
    logger.info(f"Validation: CLIP={len(val_clip_preds)}, LLM={len(val_llm_preds)}")
    logger.info(f"Test: CLIP={len(test_clip_preds)}, LLM={len(test_llm_preds)}")
    logger.info("")
    
    # Create validation predictions list (for learning weights)
    val_predictions = []
    for trial in val_trials:
        trial_id = trial.get('trial_id')
        stimulus_path = trial.get('stimulus_path', '')
        normalized_path = normalize_path(stimulus_path) if stimulus_path else ''
        lookup_key = trial_id if trial_id in val_clip_preds else (normalized_path if normalized_path in val_clip_preds else None)
        
        if lookup_key and lookup_key in val_clip_preds and lookup_key in val_llm_preds:
            correct_label = normalize_label(trial['correct_label'])
            val_predictions.append({
                'trial_id': lookup_key,  # Use the key that worked
                'correct_label': correct_label,
                'candidate_labels': val_clip_preds[lookup_key]['candidate_labels']
            })
    
    # Create test predictions list
    test_predictions = []
    for trial in test_trials:
        trial_id = trial.get('trial_id')
        stimulus_path = trial.get('stimulus_path', '')
        normalized_path = normalize_path(stimulus_path) if stimulus_path else ''
        lookup_key = trial_id if trial_id in test_clip_preds else (normalized_path if normalized_path in test_clip_preds else None)
        
        if lookup_key and lookup_key in test_clip_preds and lookup_key in test_llm_preds:
            correct_label = normalize_label(trial['correct_label'])
            test_predictions.append({
                'trial_id': lookup_key,  # Use the key that worked
                'correct_label': correct_label,
                'candidate_labels': test_clip_preds[lookup_key]['candidate_labels']
            })
    
    logger.info(f"Validation samples with both predictions: {len(val_predictions)}")
    logger.info(f"Test samples with both predictions: {len(test_predictions)}")
    logger.info("")
    
    # Learn fusion weights on validation set
    logger.info("="*80)
    logger.info("Step 1: Learning fusion weights on validation set")
    logger.info("="*80)
    weight_range = np.arange(0.0, 1.0 + args.weight_step, args.weight_step)
    emotion_weights = learn_fusion_weights(
        val_predictions,
        val_clip_preds,
        val_llm_preds,
        weight_range=weight_range,
        normalize=args.normalize
    )
    
    # Save learned weights
    weights_file = output_dir / "learned_weights.json"
    with open(weights_file, 'w') as f:
        json.dump(emotion_weights, f, indent=2)
    logger.info(f"\nSaved learned weights to: {weights_file}")
    
    # Evaluate on validation set (with learned weights)
    logger.info("")
    logger.info("="*80)
    logger.info("Step 2: Evaluating on validation set (with learned weights)")
    logger.info("="*80)
    val_acc, val_per_emotion = evaluate_fusion(
        val_predictions,
        val_clip_preds,
        val_llm_preds,
        emotion_weights,
        normalize=args.normalize
    )
    logger.info(f"Validation Accuracy: {val_acc:.2%}")
    
    # Evaluate on test set (with learned weights)
    logger.info("")
    logger.info("="*80)
    logger.info("Step 3: Evaluating on test set (held-out, never used for tuning)")
    logger.info("="*80)
    test_acc, test_per_emotion = evaluate_fusion(
        test_predictions,
        test_clip_preds,
        test_llm_preds,
        emotion_weights,
        normalize=args.normalize
    )
    logger.info(f"Test Accuracy: {test_acc:.2%}")
    
    # Compare to baseline (simple 50/50 fusion)
    logger.info("")
    logger.info("="*80)
    logger.info("Step 4: Comparison to baseline (50/50 fusion)")
    logger.info("="*80)
    baseline_weights = {emotion: {'clip': 0.5, 'llm': 0.5} for emotion in emotion_weights.keys()}
    baseline_val_acc, _ = evaluate_fusion(
        val_predictions,
        val_clip_preds,
        val_llm_preds,
        baseline_weights,
        normalize=args.normalize
    )
    baseline_test_acc, _ = evaluate_fusion(
        test_predictions,
        test_clip_preds,
        test_llm_preds,
        baseline_weights,
        normalize=args.normalize
    )
    logger.info(f"Baseline (50/50) Validation Accuracy: {baseline_val_acc:.2%}")
    logger.info(f"Baseline (50/50) Test Accuracy: {baseline_test_acc:.2%}")
    logger.info("")
    logger.info(f"Improvement on validation: {val_acc - baseline_val_acc:+.2%}")
    logger.info(f"Improvement on test: {test_acc - baseline_test_acc:+.2%}")
    
    # Save results
    results = {
        'validation_accuracy': val_acc,
        'test_accuracy': test_acc,
        'baseline_validation_accuracy': baseline_val_acc,
        'baseline_test_accuracy': baseline_test_acc,
        'improvement_validation': val_acc - baseline_val_acc,
        'improvement_test': test_acc - baseline_test_acc,
        'learned_weights': emotion_weights,
        'per_emotion_validation': val_per_emotion,
        'per_emotion_test': test_per_emotion,
        'normalize_scores': args.normalize,
        'num_validation_samples': len(val_predictions),
        'num_test_samples': len(test_predictions),
    }
    
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved results to: {results_file}")
    
    # Print summary
    logger.info("")
    logger.info("="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Validation Accuracy (learned weights): {val_acc:.2%}")
    logger.info(f"Test Accuracy (learned weights):       {test_acc:.2%}")
    logger.info(f"Baseline Validation Accuracy (50/50):   {baseline_val_acc:.2%}")
    logger.info(f"Baseline Test Accuracy (50/50):         {baseline_test_acc:.2%}")
    logger.info(f"")
    logger.info(f"Improvement on validation: {val_acc - baseline_val_acc:+.2%}")
    logger.info(f"Improvement on test:       {test_acc - baseline_test_acc:+.2%}")
    logger.info(f"")
    logger.info(f"Generalization gap (val - test): {val_acc - test_acc:.2%}")
    logger.info("")
    logger.info("="*80)
    logger.info("✅ Study complete! Results saved to:")
    logger.info(f"   - {weights_file}")
    logger.info(f"   - {results_file}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
