#!/usr/bin/env python3
"""
Create train/val splits from audio training trials.

This script splits the training data into train/val for fine-tuning,
ensuring stratified split (same emotion distribution).
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import random


def create_stratified_split(trials, train_ratio=0.8, seed=42):
    """
    Create stratified train/val split ensuring same emotion distribution.
    
    Args:
        trials: List of trial dictionaries
        train_ratio: Ratio of data for training (default: 0.8)
        seed: Random seed for reproducibility
    
    Returns:
        train_trials, val_trials
    """
    random.seed(seed)
    
    # Group trials by emotion
    trials_by_emotion = defaultdict(list)
    for trial in trials:
        emotion = trial.get('correct_label', trial.get('emotion', 'unknown'))
        trials_by_emotion[emotion].append(trial)
    
    print(f"Found {len(trials_by_emotion)} emotions")
    
    # Split each emotion's trials
    train_trials = []
    val_trials = []
    
    for emotion, emotion_trials in trials_by_emotion.items():
        # Shuffle trials for this emotion
        shuffled = emotion_trials.copy()
        random.shuffle(shuffled)
        
        # Split
        split_idx = int(len(shuffled) * train_ratio)
        train_trials.extend(shuffled[:split_idx])
        val_trials.extend(shuffled[split_idx:])
    
    # Final shuffle
    random.shuffle(train_trials)
    random.shuffle(val_trials)
    
    # Check emotion distribution
    train_emotions = defaultdict(int)
    val_emotions = defaultdict(int)
    
    for trial in train_trials:
        emotion = trial.get('correct_label', trial.get('emotion', 'unknown'))
        train_emotions[emotion] += 1
    
    for trial in val_trials:
        emotion = trial.get('correct_label', trial.get('emotion', 'unknown'))
        val_emotions[emotion] += 1
    
    print(f"\nEmotion distribution:")
    print(f"  Train: {len(train_emotions)} emotions")
    print(f"  Val: {len(val_emotions)} emotions")
    
    # Check for missing emotions
    train_only = set(train_emotions.keys()) - set(val_emotions.keys())
    val_only = set(val_emotions.keys()) - set(train_emotions.keys())
    
    if train_only:
        print(f"  ⚠️  Emotions only in train: {train_only}")
    if val_only:
        print(f"  ⚠️  Emotions only in val: {val_only}")
    
    return train_trials, val_trials


def main():
    parser = argparse.ArgumentParser(
        description="Create train/val splits from audio training trials"
    )
    parser.add_argument(
        '--train_trials',
        type=str,
        default='data/trial_definitions/eu_emotion_audio_train.json',
        help='Path to training trials JSON file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/trial_definitions',
        help='Output directory for splits'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Ratio of data for training (default: 0.8)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Load training trials
    train_file = Path(args.train_trials)
    if not train_file.exists():
        raise FileNotFoundError(f"Training trials file not found: {train_file}")
    
    print(f"Loading training trials from: {train_file}")
    with open(train_file, 'r') as f:
        data = json.load(f)
    
    trials = data.get('trials', [])
    print(f"Loaded {len(trials)} training trials")
    
    # Create train/val split
    print(f"\nCreating stratified train/val split (ratio: {args.train_ratio})...")
    train_final, val_trials = create_stratified_split(
        trials,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    
    print(f"\nSplit results:")
    print(f"  Train: {len(train_final)} trials")
    print(f"  Val: {len(val_trials)} trials")
    
    # Save splits
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_output = output_dir / "eu_emotion_audio_train.json"
    val_output = output_dir / "eu_emotion_audio_val.json"
    
    with open(train_output, 'w') as f:
        json.dump({'trials': train_final}, f, indent=2)
    
    with open(val_output, 'w') as f:
        json.dump({'trials': val_trials}, f, indent=2)
    
    print(f"\nSaved splits:")
    print(f"  Train: {train_output}")
    print(f"  Val: {val_output}")


if __name__ == "__main__":
    main()
