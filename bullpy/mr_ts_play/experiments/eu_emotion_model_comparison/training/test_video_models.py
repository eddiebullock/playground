#!/usr/bin/env python3
"""
Test fine-tuned video models (I3D, TimeSformer) on EU-Emotion test set.

Usage:
    python experiments/eu_emotion_model_comparison/training/test_video_models.py \
        --model i3d \
        --model_path models/i3d_emotion_finetuned_task_specific/best_model.pth \
        --test_trials data/trial_definitions/eu_emotion_test.json \
        --data_root /path/to/EU_emotions
"""

import argparse
import json
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.eu_emotion_model_comparison.training.finetune_video_models_task_specific import (
    create_video_model,
    get_model_feature_dim,
    extract_frames_for_video,
    preprocess_i3d_video,
    collate_video_batch,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_finetuned_model(model_type: str, model_path: str, device: str):
    """Load fine-tuned video model."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create base model
    model_result = create_video_model(model_type, device)
    if isinstance(model_result, tuple) and len(model_result) == 2:
        if model_result[1]:  # TimeSformer
            model_tuple, _ = model_result
            model, processor = model_tuple
        else:  # I3D
            model = model_result[0]
            processor = None
    else:
        model = model_result
        processor = None
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create and load score projection
    feature_dim = get_model_feature_dim(model_type, model)
    score_proj = nn.Linear(feature_dim, 4).to(device)
    score_proj.load_state_dict(checkpoint['score_proj_state_dict'])
    score_proj.eval()
    
    logger.info(f"Loaded fine-tuned {model_type} model from {model_path}")
    logger.info(f"Best validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    
    return model, score_proj, processor


def test_model(
    model,
    score_proj,
    test_trials: str,
    data_root: str,
    model_type: str,
    device: str,
    processor=None,
):
    """Test fine-tuned model on test set."""
    # Load test trials
    with open(test_trials, 'r') as f:
        data = json.load(f)
    trials = data['trials']
    
    logger.info(f"Testing on {len(trials)} trials")
    
    correct = 0
    total = 0
    
    model.eval()
    score_proj.eval()
    
    with torch.no_grad():
        for trial in tqdm(trials, desc="Testing"):
            video_path = trial['stimulus_path']
            full_path = Path(data_root) / video_path
            
            if not full_path.exists():
                logger.warning(f"Video not found: {full_path}")
                continue
            
            # Get candidate labels
            if 'candidate_labels' in trial:
                candidate_labels = trial['candidate_labels']
            else:
                # Generate candidate labels if not present
                correct_label = trial.get('correct_label', trial.get('emotion', 'unknown'))
                # For testing, we need to know all possible emotions
                # This is a simplified version - in practice, candidate_labels should be in trials
                candidate_labels = [correct_label] + ['Unknown'] * 3
            
            correct_label = trial.get('correct_label', trial.get('emotion', 'unknown'))
            correct_idx = candidate_labels.index(correct_label) if correct_label in candidate_labels else 0
            
            try:
                # Extract frames
                frames = extract_frames_for_video(
                    str(full_path),
                    num_frames=16 if model_type == "i3d" else 8
                )
                
                if not frames:
                    continue
                
                # Process video
                if model_type == "i3d":
                    video_tensor = preprocess_i3d_video(frames, device)
                    if video_tensor is None:
                        continue
                    features = model(video_tensor)
                    video_features = features.squeeze(0)
                
                elif model_type == "timesformer":
                    inputs = processor(images=frames, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1]
                    video_features = hidden_states.mean(dim=1).squeeze(0)
                
                else:
                    continue
                
                # Get predictions
                emotion_scores = score_proj(video_features.unsqueeze(0))
                predicted_idx = emotion_scores.argmax(dim=1).item()
                
                if predicted_idx == correct_idx:
                    correct += 1
                total += 1
            
            except Exception as e:
                logger.warning(f"Error processing {full_path}: {e}")
                continue
    
    accuracy = 100 * correct / total if total > 0 else 0.0
    logger.info(f"\n{'='*60}")
    logger.info(f"Test Results:")
    logger.info(f"  Correct: {correct}/{total}")
    logger.info(f"  Accuracy: {accuracy:.2f}%")
    logger.info(f"{'='*60}")
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned video models")
    parser.add_argument('--model', type=str, required=True,
                       choices=['i3d', 'timesformer'],
                       help='Video model type')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to fine-tuned model checkpoint')
    parser.add_argument('--test_trials', type=str, required=True,
                       help='Path to test trials JSON file')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of EU-Emotion dataset')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to test on')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    
    # Load model
    model, score_proj, processor = load_finetuned_model(
        args.model, args.model_path, device
    )
    
    # Test model
    accuracy = test_model(
        model,
        score_proj,
        args.test_trials,
        args.data_root,
        args.model,
        device,
        processor,
    )
    
    return accuracy


if __name__ == "__main__":
    main()
