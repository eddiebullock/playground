#!/usr/bin/env python3
"""
Create basic emotion trial definitions for 7-way classification.

This script:
1. Loads existing trial definitions (CAM or EU-Emotion)
2. Maps fine-grained emotions to basic emotions using mapping files
3. Creates 7-way classification trials (all 7 basic emotions as candidates, no foils)
4. Creates train/test splits (80/20) with actor independence
5. Saves trial definitions with basic emotion labels

Key difference from forced-choice:
- Model selects from all 7 basic emotions (not 4 options)
- No foil selection needed
- Standard multi-class classification
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Basic emotion categories (Ekman's 6 + neutral)
BASIC_EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]


def load_emotion_mapping(mapping_file: str) -> Dict[str, str]:
    """Load emotion mapping from JSON file."""
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    return mapping


def normalize_emotion_name(emotion: str) -> str:
    """Normalize emotion name for matching (lowercase, strip whitespace)."""
    return emotion.lower().strip()


def map_to_basic_emotion(
    fine_grained_emotion: str,
    mapping: Dict[str, str]
) -> str:
    """
    Map fine-grained emotion to basic emotion.
    
    Args:
        fine_grained_emotion: Original fine-grained emotion name
        mapping: Dictionary mapping fine-grained -> basic emotions
    
    Returns:
        Basic emotion category
    """
    normalized = normalize_emotion_name(fine_grained_emotion)
    
    # Try exact match first
    if normalized in mapping:
        return mapping[normalized]
    
    # Try with " low intensity" suffix stripped (for EU-Emotion)
    if " low intensity" in normalized:
        base_emotion = normalized.replace(" low intensity", "").strip()
        if base_emotion in mapping:
            return mapping[base_emotion]
    
    # Try case-insensitive match
    for key, value in mapping.items():
        if normalize_emotion_name(key) == normalized:
            return value
    
    # Default to neutral if not found
    print(f"Warning: Could not map '{fine_grained_emotion}' to basic emotion, defaulting to 'neutral'")
    return "neutral"


def create_actor_independent_split(
    trials: List[Dict],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Create actor-independent train/test split.
    
    Ensures no actor appears in both train and test sets.
    """
    random.seed(seed)
    
    # Group trials by actor
    actor_trials = defaultdict(list)
    for trial in trials:
        actor = trial.get('actor', 'unknown')
        actor_trials[actor].append(trial)
    
    actors = list(actor_trials.keys())
    print(f"Found {len(actors)} unique actors")
    
    # Shuffle actors
    random.shuffle(actors)
    
    # Split actors
    split_idx = int(len(actors) * train_ratio)
    train_actors = set(actors[:split_idx])
    test_actors = set(actors[split_idx:])
    
    print(f"Train actors: {len(train_actors)}")
    print(f"Test actors: {len(test_actors)}")
    
    # Assign trials to splits based on actor
    train_trials = []
    test_trials = []
    
    for trial in trials:
        actor = trial.get('actor', 'unknown')
        if actor in train_actors:
            train_trials.append(trial)
        elif actor in test_actors:
            test_trials.append(trial)
        else:
            # Unknown actor, assign randomly
            if random.random() < train_ratio:
                train_trials.append(trial)
            else:
                test_trials.append(trial)
    
    return train_trials, test_trials


def convert_trials_to_basic_emotions(
    trials: List[Dict],
    emotion_mapping: Dict[str, str],
    dataset_type: str
) -> List[Dict]:
    """
    Convert fine-grained emotion trials to basic emotion trials.
    
    Args:
        trials: List of trial dictionaries with fine-grained emotions
        emotion_mapping: Dictionary mapping fine-grained -> basic emotions
        dataset_type: "cam" or "eu_emotion"
    
    Returns:
        List of trial dictionaries with basic emotion labels
    """
    basic_trials = []
    
    for trial in trials:
        # Get original fine-grained emotion
        if dataset_type == "cam":
            fine_grained = trial.get('correct_label') or trial.get('concept', '')
        else:  # eu_emotion
            fine_grained = trial.get('correct_label') or trial.get('emotion', '')
        
        if not fine_grained:
            print(f"Warning: Trial {trial.get('trial_id', 'unknown')} has no emotion label, skipping")
            continue
        
        # Map to basic emotion
        basic_emotion = map_to_basic_emotion(fine_grained, emotion_mapping)
        
        # Create new trial with basic emotion labels
        # All 7 basic emotions are candidate labels (7-way classification)
        basic_trial = {
            "trial_id": trial.get('trial_id', f"basic_trial_{len(basic_trials)+1:03d}"),
            "stimulus_path": trial.get('stimulus_path', ''),
            "modality": trial.get('modality', 'face'),
            "fine_grained_emotion": fine_grained,  # Keep original for reference
            "basic_emotion": basic_emotion,  # Correct basic emotion label
            "all_candidate_labels": BASIC_EMOTIONS.copy(),  # All 7 options
            "correct_label": basic_emotion,  # Correct answer
            "correct_idx": BASIC_EMOTIONS.index(basic_emotion),  # Index of correct label
        }
        
        # Preserve actor and scenario_id if available
        if 'actor' in trial:
            basic_trial['actor'] = trial['actor']
        if 'scenario_id' in trial:
            basic_trial['scenario_id'] = trial['scenario_id']
        
        basic_trials.append(basic_trial)
    
    return basic_trials


def main():
    parser = argparse.ArgumentParser(
        description="Create basic emotion trial definitions for 7-way classification"
    )
    parser.add_argument(
        '--dataset_type',
        type=str,
        choices=['cam', 'eu_emotion'],
        required=True,
        help='Dataset type: cam or eu_emotion'
    )
    parser.add_argument(
        '--input_trials',
        type=str,
        required=True,
        help='Path to input trial definitions JSON file'
    )
    parser.add_argument(
        '--mapping_file',
        type=str,
        required=True,
        help='Path to emotion mapping JSON file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for basic emotion trial definitions'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Train/test split ratio (default: 0.8)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for splitting (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Load input trials
    print(f"Loading trials from: {args.input_trials}")
    with open(args.input_trials, 'r') as f:
        input_data = json.load(f)
    
    trials = input_data.get('trials', [])
    if not trials:
        # Try if the file is a list directly
        if isinstance(input_data, list):
            trials = input_data
        else:
            raise ValueError(f"No trials found in {args.input_trials}")
    
    print(f"Loaded {len(trials)} trials")
    
    # Load emotion mapping
    print(f"Loading emotion mapping from: {args.mapping_file}")
    emotion_mapping = load_emotion_mapping(args.mapping_file)
    print(f"Loaded mapping for {len(emotion_mapping)} emotions")
    
    # Convert to basic emotions
    print("\nConverting trials to basic emotions...")
    basic_trials = convert_trials_to_basic_emotions(
        trials,
        emotion_mapping,
        args.dataset_type
    )
    print(f"Converted {len(basic_trials)} trials to basic emotions")
    
    # Check basic emotion distribution
    basic_emotion_counts = defaultdict(int)
    for trial in basic_trials:
        basic_emotion_counts[trial['basic_emotion']] += 1
    
    print("\nBasic emotion distribution:")
    for emotion in BASIC_EMOTIONS:
        count = basic_emotion_counts[emotion]
        print(f"  {emotion}: {count} ({count/len(basic_trials)*100:.1f}%)")
    
    # Create train/test split with actor independence
    print(f"\nCreating actor-independent train/test split (ratio: {args.train_ratio})...")
    train_trials, test_trials = create_actor_independent_split(
        basic_trials,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    
    print(f"Train trials: {len(train_trials)}")
    print(f"Test trials: {len(test_trials)}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save trial definitions
    dataset_prefix = "cam" if args.dataset_type == "cam" else "eu_emotion"
    
    train_output = output_dir / f"{dataset_prefix}_basic_emotions_train.json"
    test_output = output_dir / f"{dataset_prefix}_basic_emotions_test.json"
    all_output = output_dir / f"{dataset_prefix}_basic_emotions_all.json"
    
    # Save train trials
    with open(train_output, 'w') as f:
        json.dump({"trials": train_trials}, f, indent=2)
    print(f"\nSaved train trials to: {train_output}")
    
    # Save test trials
    with open(test_output, 'w') as f:
        json.dump({"trials": test_trials}, f, indent=2)
    print(f"Saved test trials to: {test_output}")
    
    # Save all trials
    with open(all_output, 'w') as f:
        json.dump({"trials": basic_trials}, f, indent=2)
    print(f"Saved all trials to: {all_output}")
    
    print("\nBasic emotion trial generation complete!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

