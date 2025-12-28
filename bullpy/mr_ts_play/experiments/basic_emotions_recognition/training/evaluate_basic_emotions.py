#!/usr/bin/env python3
"""
Evaluate fine-tuned CLIP model on basic emotions test set (4-option forced-choice).
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import json
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import dataset
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "cam_human_like" / "training"))
from task_specific_dataset import collate_trial_batch

from finetune_basic_emotions import BasicEmotionDataset, BASIC_EMOTIONS


def evaluate_basic_emotions(
    model_path: str,
    trial_definitions_file: str,
    data_root: str,
    device: str = "cpu",
    num_frames: int = 16,
):
    """
    Evaluate fine-tuned CLIP model on basic emotions test set.
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Load model
    print(f"Loading model from: {model_path}")
    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    
    # Create dataset
    print(f"Loading test trials from: {trial_definitions_file}")
    test_dataset = BasicEmotionDataset(
        data_root=data_root,
        trial_definitions_file=trial_definitions_file,
        num_frames=num_frames,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_trial_batch,
    )
    
    # Evaluate
    print("Evaluating model...")
    correct = 0
    total = 0
    
    # Per-emotion statistics
    emotion_correct = defaultdict(int)
    emotion_total = defaultdict(int)
    
    # Confusion matrix
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    predictions = []
    
    with torch.no_grad():
        for frames_batch, candidate_labels_batch, correct_indices in tqdm(test_loader, desc="Evaluating"):
            batch_size_actual = len(frames_batch)
            
            for video_idx in range(batch_size_actual):
                try:
                    video_frames = frames_batch[video_idx]
                    candidate_labels = candidate_labels_batch[video_idx]
                    correct_idx = correct_indices[video_idx].item()
                    
                    # Get ground truth emotion
                    ground_truth = candidate_labels[correct_idx]
                    
                    # Process frames
                    image_inputs = processor(images=video_frames, return_tensors="pt").to(device)
                    
                    # Process all 4 candidate labels (forced-choice format)
                    prompted_labels = [f"a photo of a person feeling {label}" for label in candidate_labels]
                    text_inputs = processor(
                        text=prompted_labels,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).to(device)
                    
                    # Get embeddings
                    image_features = model.get_image_features(**image_inputs)
                    text_features = model.get_text_features(**text_inputs)
                    
                    # Aggregate video features
                    video_features = image_features.mean(dim=0, keepdim=True)
                    
                    # Normalize
                    video_features = F.normalize(video_features, dim=-1)
                    text_features = F.normalize(text_features, dim=-1)
                    
                    # Compute similarity
                    logits = video_features @ text_features.t()
                    
                    # Get predicted index
                    predicted_idx = logits.argmax(dim=-1).item()
                    predicted_emotion = candidate_labels[predicted_idx]
                    
                    # Update statistics
                    if predicted_idx == correct_idx:
                        correct += 1
                        emotion_correct[ground_truth] += 1
                    total += 1
                    emotion_total[ground_truth] += 1
                    
                    # Update confusion matrix
                    confusion_matrix[ground_truth][predicted_emotion] += 1
                    
                    # Store prediction
                    predictions.append({
                        "ground_truth": ground_truth,
                        "predicted": predicted_emotion,
                        "correct": predicted_idx == correct_idx,
                    })
                except Exception as e:
                    print(f"Error evaluating video {video_idx}: {e}", flush=True)
                    continue
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0.0
    
    # Per-emotion accuracy
    per_emotion_accuracy = {}
    for emotion in BASIC_EMOTIONS:
        if emotion_total[emotion] > 0:
            per_emotion_accuracy[emotion] = emotion_correct[emotion] / emotion_total[emotion]
        else:
            per_emotion_accuracy[emotion] = 0.0
    
    # Create confusion matrix as list of lists
    confusion_matrix_list = []
    confusion_matrix_list.append(["Ground Truth \\ Predicted"] + BASIC_EMOTIONS)
    for true_emotion in BASIC_EMOTIONS:
        row = [true_emotion]
        for pred_emotion in BASIC_EMOTIONS:
            row.append(confusion_matrix[true_emotion][pred_emotion])
        confusion_matrix_list.append(row)
    
    results = {
        "accuracy": accuracy,
        "total_trials": total,
        "correct_predictions": correct,
        "per_emotion_accuracy": per_emotion_accuracy,
        "per_emotion_total": dict(emotion_total),
        "confusion_matrix": confusion_matrix_list,
        "predictions": predictions,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned CLIP model on basic emotions test set"
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to fine-tuned model directory'
    )
    parser.add_argument(
        '--trial_definitions',
        type=str,
        required=True,
        help='Path to test trial definitions JSON file'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Root directory of video files'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to evaluate on: auto (detect), cpu, cuda, or mps (default: auto)'
    )
    parser.add_argument(
        '--num_frames',
        type=int,
        default=16,
        help='Number of frames per video (default: 16)'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Output file for evaluation results (JSON)'
    )
    
    args = parser.parse_args()
    
    # Auto-detect device if 'auto' or not specified
    if args.device == 'auto' or args.device is None:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        elif torch.cuda.is_available():
            args.device = 'cuda'
        else:
            args.device = 'cpu'
        print(f"Auto-detected device: {args.device}")
    
    # Evaluate
    results = evaluate_basic_emotions(
        model_path=args.model_path,
        trial_definitions_file=args.trial_definitions,
        data_root=args.data_root,
        device=args.device,
        num_frames=args.num_frames,
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {results['accuracy']:.2%}")
    print(f"Total Trials: {results['total_trials']}")
    print(f"Correct Predictions: {results['correct_predictions']}")
    print("\nPer-Emotion Accuracy:")
    for emotion in BASIC_EMOTIONS:
        acc = results['per_emotion_accuracy'][emotion]
        total = results['per_emotion_total'][emotion]
        print(f"  {emotion:12s}: {acc:.2%} ({results['per_emotion_total'][emotion]} trials)")
    
    print("\nConfusion Matrix:")
    for row in results['confusion_matrix']:
        print("  " + "  ".join(str(x) for x in row))
    
    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

