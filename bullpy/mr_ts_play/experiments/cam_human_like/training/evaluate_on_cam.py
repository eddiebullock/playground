#!/usr/bin/env python3
"""
Evaluate a fine-tuned CLIP model on the CAM test set.

This script loads a fine-tuned model and evaluates it on CAM,
comparing performance to the zero-shot baseline (37%).
"""

import argparse
import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.cam_human_like.run_experiment import main as run_experiment_main
from experiments.cam_human_like.models.clip_wrapper import CLIPWrapper
from experiments.cam_human_like.dataset import CAMDataset
from experiments.cam_human_like.trials.forced_choice import ForcedChoiceTrial
import torch


def compute_metrics(predictions, trials):
    """
    Compute evaluation metrics from predictions and trials.
    
    Args:
        predictions: List of prediction dicts with keys:
            - 'is_correct': bool
            - 'predicted_label': str
            - 'correct_label': str
        trials: List of trial objects with 'modality' attribute
    
    Returns:
        Dictionary with metrics: accuracy, face_accuracy, voice_accuracy
    """
    if not predictions:
        return {'accuracy': 0.0, 'face_accuracy': 0.0, 'voice_accuracy': 0.0}
    
    # Overall accuracy
    correct = sum(1 for p in predictions if p['is_correct'])
    accuracy = correct / len(predictions)
    
    # Split by modality
    face_predictions = []
    voice_predictions = []
    
    for pred, trial in zip(predictions, trials):
        if trial.modality == 'face':
            face_predictions.append(pred)
        elif trial.modality == 'voice':
            voice_predictions.append(pred)
    
    face_accuracy = sum(1 for p in face_predictions if p['is_correct']) / len(face_predictions) if face_predictions else 0.0
    voice_accuracy = sum(1 for p in voice_predictions if p['is_correct']) / len(voice_predictions) if voice_predictions else 0.0
    
    return {
        'accuracy': accuracy,
        'face_accuracy': face_accuracy,
        'voice_accuracy': voice_accuracy,
    }


def evaluate_finetuned_model(
    model_path: str,
    trial_definitions_file: str,
    data_root: str,
    splits_dir: str = None,
    split_name: str = "test",
    device: str = "cpu",
    num_frames: int = 8,
    use_multiframe: bool = True,
):
    """
    Evaluate a fine-tuned CLIP model on CAM test set.
    
    Args:
        model_path: Path to fine-tuned model (directory containing config.json, pytorch_model.bin)
        trial_definitions_file: Path to CAM trial definitions JSON
        data_root: Root directory of CAM stimuli
        splits_dir: Directory containing train/val/test splits (optional)
        split_name: Which split to evaluate on ("test" or "val")
        device: Device to run on
        num_frames: Number of frames to extract per video
        use_multiframe: Whether to use multiple frames (average features)
    """
    print("=" * 60)
    print("Evaluating Fine-Tuned Model on CAM Test Set")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Split: {split_name}")
    print(f"Device: {device}")
    print(f"Multi-frame: {use_multiframe}")
    print()
    
    # Load dataset
    print("Loading CAM test dataset...")
    dataset = CAMDataset(
        data_root=data_root,
        trial_definitions_file=trial_definitions_file,
        splits_dir=splits_dir,
        split_name=split_name,
        use_actor_filtering=False,  # Use all trials to match original CAM
    )
    print(f"Loaded {len(dataset.trials)} trials")
    print()
    
    # Load fine-tuned model
    print(f"Loading fine-tuned model from: {model_path}")
    try:
        model = CLIPWrapper(
            model_name=model_path,
            device=device,
            num_frames=num_frames,
            aggregation="mean" if use_multiframe else "middle",
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nTrying alternative loading method...")
        # Try loading as HuggingFace model directly
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(model_path)
        processor = CLIPProcessor.from_pretrained(model_path)
        # Wrap in CLIPWrapper
        model = CLIPWrapper(
            model_name=model_path,
            device=device,
            num_frames=num_frames,
            aggregation="mean" if use_multiframe else "middle",
        )
    print()
    
    # Run evaluation
    print("Running evaluation...")
    predictions = []
    correct_labels = []
    skipped_trials = []
    
    for trial in dataset.trials:
        try:
            # Score labels
            output = model.score_labels(
                stimulus_path=trial.stimulus_path,
                candidate_labels=trial.candidate_labels,
                modality=trial.modality,
            )
            
            # Get prediction (highest scoring label)
            predicted_label = max(output.label_scores.items(), key=lambda x: x[1])[0]
            predicted_idx = trial.candidate_labels.index(predicted_label)
            
            predictions.append({
                'trial_id': trial.trial_id,
                'predicted_label': predicted_label,
                'predicted_idx': predicted_idx,
                'correct_label': trial.correct_label,
                'correct_idx': trial.correct_idx,
                'is_correct': predicted_idx == trial.correct_idx,
                'scores': output.label_scores,
            })
            correct_labels.append(trial.correct_label)
        except (ValueError, FileNotFoundError, OSError) as e:
            # Skip trials with missing or corrupted video files
            print(f"Warning: Skipping trial {trial.trial_id}: {e}")
            skipped_trials.append(trial.trial_id)
            continue
    
    if skipped_trials:
        print(f"\nSkipped {len(skipped_trials)} trials due to missing/corrupted files")
        print(f"Evaluating on {len(predictions)} valid trials")
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(predictions, dataset.trials)
    
    # Print results
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Overall Accuracy: {metrics['accuracy']:.2%}")
    print(f"Face Accuracy: {metrics.get('face_accuracy', 'N/A')}")
    print(f"Voice Accuracy: {metrics.get('voice_accuracy', 'N/A')}")
    print()
    print(f"Baseline (zero-shot CLIP): 37.0%")
    print(f"Improvement: {metrics['accuracy'] - 0.37:.2%} ({((metrics['accuracy'] / 0.37) - 1) * 100:.1f}% relative)")
    print()
    
    # Save results
    results_file = Path(model_path).parent / f"cam_evaluation_{split_name}.json"
    results = {
        'model_path': model_path,
        'split': split_name,
        'num_trials': len(dataset.trials),
        'num_valid_trials': len(predictions),
        'num_skipped_trials': len(skipped_trials),
        'skipped_trial_ids': skipped_trials,
        'metrics': metrics,
        'predictions': predictions,
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    print()
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned CLIP model on CAM test set")
    parser.add_argument('--model_path', type=str, required=True, help='Path to fine-tuned model directory')
    parser.add_argument('--trial_definitions', type=str, default='data/cam_trial_definitions_20concepts.json', help='Path to CAM trial definitions')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of CAM stimuli')
    parser.add_argument('--splits_dir', type=str, help='Directory containing train/val/test splits (optional)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='Which split to evaluate on')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu, cuda, mps)')
    parser.add_argument('--num_frames', type=int, default=8, help='Number of frames per video')
    parser.add_argument('--use_multiframe', action='store_true', default=True, help='Use multiple frames (average features)')
    parser.add_argument('--single_frame', action='store_true', help='Use only middle frame')
    
    args = parser.parse_args()
    
    use_multiframe = args.use_multiframe and not args.single_frame
    
    metrics = evaluate_finetuned_model(
        model_path=args.model_path,
        trial_definitions_file=args.trial_definitions,
        data_root=args.data_root,
        splits_dir=args.splits_dir,
        split_name=args.split,
        device=args.device,
        num_frames=args.num_frames,
        use_multiframe=use_multiframe,
    )
    
    print("=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()


