#!/usr/bin/env python3
"""
Create basic emotion trial definitions for 4-option forced-choice (matching complex emotion experiments).

This script:
1. Loads existing trial definitions (CAM or EU-Emotion)
2. Maps fine-grained emotions to basic emotions using mapping files
3. Creates 4-option forced-choice trials (1 correct + 3 foils from other basic emotions)
4. Creates train/test splits (80/20) with actor independence
5. Saves trial definitions with basic emotion labels

Key change: Now uses 4-option forced-choice (like complex emotion experiments):
- Model selects from 4 candidate labels (1 correct + 3 foils)
- Foils are selected from the other 6 basic emotions
- Easier discrimination task than 7-way classification
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
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


def select_foils_for_basic_emotion(
    target_emotion: str,
    num_foils: int = 3,
    seed: Optional[int] = None
) -> List[str]:
    """
    Select foil basic emotions for a target basic emotion.
    
    Selects 3 foils from the other 6 basic emotions (excluding target).
    This matches the forced-choice format used in complex emotion experiments.
    
    Args:
        target_emotion: Target basic emotion (one of BASIC_EMOTIONS)
        num_foils: Number of foils to select (default: 3)
        seed: Random seed for reproducibility
    
    Returns:
        List of foil basic emotion names
    """
    if seed is not None:
        random.seed(seed)
    
    # Get all basic emotions except target
    candidates = [e for e in BASIC_EMOTIONS if e != target_emotion]
    
    # Randomly sample foils
    if len(candidates) >= num_foils:
        foils = random.sample(candidates, num_foils)
    else:
        # Should never happen (we have 6 candidates for 3 foils)
        foils = candidates.copy()
        while len(foils) < num_foils:
            foils.append(random.choice(candidates))
        foils = foils[:num_foils]
    
    return foils


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
    Create actor-independent train/test split for CAM dataset.
    
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


def create_random_split(
    trials: List[Dict],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Create simple random train/test split (for EU-Emotion which has no actors).
    
    This matches the approach used in create_eu_emotion_trials.py
    """
    random.seed(seed)
    
    # Shuffle trials
    shuffled = trials.copy()
    random.shuffle(shuffled)
    
    # Split
    split_idx = int(len(shuffled) * train_ratio)
    train_trials = shuffled[:split_idx]
    test_trials = shuffled[split_idx:]
    
    return train_trials, test_trials


def create_stratified_split(
    trials: List[Dict],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Create stratified train/test split ensuring same emotion distribution.
    
    Groups trials by emotion and splits each group proportionally.
    This ensures train and test have similar emotion distributions.
    
    Args:
        trials: List of trial dictionaries with 'correct_label' field
        train_ratio: Proportion of trials for training
        seed: Random seed
    
    Returns:
        Tuple of (train_trials, test_trials)
    """
    random.seed(seed)
    
    # Group trials by emotion (correct_label)
    emotion_trials = defaultdict(list)
    for trial in trials:
        emotion = trial.get('correct_label', 'neutral')
        emotion_trials[emotion].append(trial)
    
    print(f"Found {len(emotion_trials)} unique emotions for stratified split")
    
    train_trials = []
    test_trials = []
    
    # Split each emotion's trials proportionally
    for emotion, emotion_trial_list in emotion_trials.items():
        # Shuffle this emotion's trials
        random.shuffle(emotion_trial_list)
        
        # Calculate split point for this emotion
        split_idx = int(len(emotion_trial_list) * train_ratio)
        
        # Ensure at least 1 trial in each split if possible
        if split_idx == 0 and len(emotion_trial_list) > 1:
            split_idx = 1
        elif split_idx == len(emotion_trial_list) and len(emotion_trial_list) > 1:
            split_idx = len(emotion_trial_list) - 1
        
        train_trials.extend(emotion_trial_list[:split_idx])
        test_trials.extend(emotion_trial_list[split_idx:])
        
        print(f"  {emotion}: {len(emotion_trial_list[:split_idx])} train, {len(emotion_trial_list[split_idx:])} test")
    
    # Shuffle final splits to avoid ordering bias
    random.shuffle(train_trials)
    random.shuffle(test_trials)
    
    return train_trials, test_trials




def convert_trials_to_basic_emotions(
    trials: List[Dict],
    emotion_mapping: Dict[str, str],
    dataset_type: str
) -> List[Dict]:
    """
    Convert fine-grained emotion trials to basic emotion trials.
    
    For CAM dataset, filters out voice modality files (T files) which are corrupted.
    Only uses face modality files (V files).
    
    Args:
        trials: List of trial dictionaries with fine-grained emotions
        emotion_mapping: Dictionary mapping fine-grained -> basic emotions
        dataset_type: "cam" or "eu_emotion"
    
    Returns:
        List of trial dictionaries with basic emotion labels (face only for CAM)
    """
    basic_trials = []
    skipped_voice = 0
    
    for trial in trials:
        # For CAM dataset, filter out voice modality files (T files are corrupted)
        if dataset_type == "cam":
            stimulus_path = trial.get('stimulus_path', '')
            modality = trial.get('modality', '')
            
            # Skip if it's a voice file
            # T files have pattern: {scenario_id}{actor}T{emotion}.mov (e.g., 1900201A6Tappealing.mov)
            # V files have pattern: {scenario_id}{actor}V{emotion}.mov (e.g., 1900201A6Vappealing.mov)
            is_voice_file = False
            
            # Check modality field first
            if modality == 'voice':
                is_voice_file = True
            # Also check filename pattern (in case modality field is missing)
            elif stimulus_path:
                filename = Path(stimulus_path).name
                # Pattern: filename contains T followed by emotion name (before .mov)
                # Simple heuristic: if filename has 'T' and ends with .mov, check if T is before emotion
                # More reliable: check if filename matches pattern *T*.mov (T before emotion)
                if filename.endswith('.mov'):
                    # Remove .mov and check if T appears (likely before emotion name)
                    name_without_ext = filename.replace('.mov', '')
                    # If T is in the filename and it's not part of "Trial" or other words
                    # The pattern is: {scenario}{actor}T{emotion} or {scenario}{actor}V{emotion}
                    # So we check if there's a T that's likely the modality code
                    if 'T' in name_without_ext and 'V' not in name_without_ext:
                        # Additional check: T should be followed by letters (emotion name)
                        # This distinguishes from T being part of a word
                        t_index = name_without_ext.find('T')
                        if t_index > 0 and t_index < len(name_without_ext) - 1:
                            # T found and there's text after it (likely emotion)
                            is_voice_file = True
            
            if is_voice_file:
                skipped_voice += 1
                continue
        
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
        
        # Select 3 foils from other basic emotions (4-option forced-choice)
        trial_seed = hash(trial.get('trial_id', str(len(basic_trials)))) % (2**31)
        foils = select_foils_for_basic_emotion(
            target_emotion=basic_emotion,
            num_foils=3,
            seed=trial_seed
        )
        
        # Create candidate labels: target + foils (4 options total)
        candidate_labels = [basic_emotion] + foils
        
        # Randomize order (matching complex emotion experiment format)
        random.seed(trial_seed)
        random.shuffle(candidate_labels)
        
        # Find correct index after shuffling
        correct_idx = candidate_labels.index(basic_emotion)
        
        # Create new trial with 4-option forced-choice format
        basic_trial = {
            "trial_id": trial.get('trial_id', f"basic_trial_{len(basic_trials)+1:03d}"),
            "stimulus_path": trial.get('stimulus_path', ''),
            "modality": trial.get('modality', 'face'),
            "fine_grained_emotion": fine_grained,  # Keep original for reference
            "correct_label": basic_emotion,  # Correct basic emotion label
            "candidate_labels": candidate_labels,  # 4 options: 1 correct + 3 foils
            "correct_idx": correct_idx,  # Index of correct label (0-3)
        }
        
        # Preserve actor and scenario_id if available
        if 'actor' in trial:
            basic_trial['actor'] = trial['actor']
        if 'scenario_id' in trial:
            basic_trial['scenario_id'] = trial['scenario_id']
        
        basic_trials.append(basic_trial)
    
    if dataset_type == "cam" and skipped_voice > 0:
        print(f"  Filtered out {skipped_voice} voice modality trials (T files are corrupted)")
    
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
        basic_emotion_counts[trial['correct_label']] += 1
    
    print("\nBasic emotion distribution:")
    for emotion in BASIC_EMOTIONS:
        count = basic_emotion_counts[emotion]
        print(f"  {emotion}: {count} ({count/len(basic_trials)*100:.1f}%)")
    
    # Create train/test split
    # CAM uses actor-independent split, EU-Emotion uses random split (no actors)
    if args.dataset_type == "cam":
        print(f"\nCreating actor-independent train/test split (ratio: {args.train_ratio})...")
        train_trials, test_trials = create_actor_independent_split(
            basic_trials,
            train_ratio=args.train_ratio,
            seed=args.seed
        )
    else:  # eu_emotion
        print(f"\nCreating stratified train/test split (ratio: {args.train_ratio})...")
        print("  (Ensures same emotion distribution in train and test)")
        train_trials, test_trials = create_stratified_split(
            basic_trials,
            train_ratio=args.train_ratio,
            seed=args.seed
        )
    
    print(f"Train trials: {len(train_trials)}")
    print(f"Test trials: {len(test_trials)}")
    
    # Split train into train/val (use 20% of train for validation)
    # This ensures test set is truly held out
    print(f"\nSplitting train set into train/val (80/20 split of train set)...")
    random.seed(args.seed)
    shuffled_train = train_trials.copy()
    random.shuffle(shuffled_train)
    val_split_idx = int(len(shuffled_train) * 0.8)
    train_final = shuffled_train[:val_split_idx]
    val_trials = shuffled_train[val_split_idx:]
    
    print(f"Final train trials: {len(train_final)}")
    print(f"Validation trials: {len(val_trials)}")
    print(f"Test trials: {len(test_trials)}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save trial definitions
    dataset_prefix = "cam" if args.dataset_type == "cam" else "eu_emotion"
    
    train_output = output_dir / f"{dataset_prefix}_basic_emotions_train.json"
    val_output = output_dir / f"{dataset_prefix}_basic_emotions_val.json"
    test_output = output_dir / f"{dataset_prefix}_basic_emotions_test.json"
    all_output = output_dir / f"{dataset_prefix}_basic_emotions_all.json"
    
    # Save train trials
    with open(train_output, 'w') as f:
        json.dump({"trials": train_final}, f, indent=2)
    print(f"\nSaved train trials to: {train_output}")
    
    # Save validation trials
    with open(val_output, 'w') as f:
        json.dump({"trials": val_trials}, f, indent=2)
    print(f"Saved validation trials to: {val_output}")
    
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

