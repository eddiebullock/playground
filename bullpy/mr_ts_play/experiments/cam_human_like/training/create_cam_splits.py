#!/usr/bin/env python3
"""
Create train/test splits for CAM dataset from existing trial definitions.

This script:
1. Loads existing CAM trial definitions
2. Creates train/test splits (80/20) ensuring proper distribution
3. Optionally uses actor-independent splitting
4. Saves split definitions to data/cam_splits/
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import random


def create_actor_independent_split(
    trials: List[Dict],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Create actor-independent train/test split.
    
    Ensures no actor appears in both train and test sets.
    
    Args:
        trials: List of trial dictionaries
        train_ratio: Proportion of actors for training
        seed: Random seed
    
    Returns:
        Tuple of (train_trials, test_trials)
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
    Create random train/test split.
    
    Args:
        trials: List of trial dictionaries
        train_ratio: Proportion of trials for training
        seed: Random seed
    
    Returns:
        Tuple of (train_trials, test_trials)
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


def create_concept_balanced_split(
    trials: List[Dict],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Create train/test split balanced by concept.
    
    Ensures each concept has trials in both train and test sets.
    
    Args:
        trials: List of trial dictionaries
        train_ratio: Proportion of trials for training
        seed: Random seed
    
    Returns:
        Tuple of (train_trials, test_trials)
    """
    random.seed(seed)
    
    # Group trials by concept
    concept_trials = defaultdict(list)
    for trial in trials:
        concept = trial.get('concept') or trial.get('correct_label', 'unknown')
        concept_trials[concept].append(trial)
    
    print(f"Found {len(concept_trials)} concepts")
    
    train_trials = []
    test_trials = []
    
    # Split each concept's trials
    for concept, concept_trial_list in concept_trials.items():
        random.shuffle(concept_trial_list)
        split_idx = int(len(concept_trial_list) * train_ratio)
        train_trials.extend(concept_trial_list[:split_idx])
        test_trials.extend(concept_trial_list[split_idx:])
    
    # Shuffle final splits
    random.shuffle(train_trials)
    random.shuffle(test_trials)
    
    return train_trials, test_trials


def main():
    parser = argparse.ArgumentParser(
        description="Create train/test splits for CAM dataset"
    )
    parser.add_argument(
        '--trial-definitions',
        type=str,
        default='data/cam_trial_definitions_20concepts.json',
        help='Path to CAM trial definitions JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/cam_splits',
        help='Output directory for split files (default: data/cam_splits)'
    )
    parser.add_argument(
        '--split-method',
        type=str,
        default='concept_balanced',
        choices=['random', 'actor_independent', 'concept_balanced'],
        help='Split method (default: concept_balanced)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Proportion of trials for training (default: 0.8)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Load trial definitions
    trial_file = Path(args.trial_definitions)
    if not trial_file.exists():
        print(f"Error: Trial definitions file not found: {trial_file}")
        return
    
    print(f"Loading trial definitions from {trial_file}...")
    with open(trial_file, 'r') as f:
        data = json.load(f)
    
    trials = data.get('trials', [])
    print(f"Loaded {len(trials)} trials")
    
    # Create splits
    if args.split_method == 'actor_independent':
        print("\nCreating actor-independent split...")
        train_trials, test_trials = create_actor_independent_split(
            trials,
            train_ratio=args.train_ratio,
            seed=args.seed
        )
    elif args.split_method == 'concept_balanced':
        print("\nCreating concept-balanced split...")
        train_trials, test_trials = create_concept_balanced_split(
            trials,
            train_ratio=args.train_ratio,
            seed=args.seed
        )
    else:
        print("\nCreating random split...")
        train_trials, test_trials = create_random_split(
            trials,
            train_ratio=args.train_ratio,
            seed=args.seed
        )
    
    print(f"\nTrain trials: {len(train_trials)}")
    print(f"Test trials: {len(test_trials)}")
    
    # Analyze splits
    train_concepts = set(t.get('concept') or t.get('correct_label') for t in train_trials)
    test_concepts = set(t.get('concept') or t.get('correct_label') for t in test_trials)
    print(f"\nTrain concepts: {len(train_concepts)}")
    print(f"Test concepts: {len(test_concepts)}")
    print(f"Overlapping concepts: {len(train_concepts & test_concepts)}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save split files
    train_output = output_dir / "train_trials.json"
    test_output = output_dir / "test_trials.json"
    
    for output_file, trials_list, split_name in [
        (train_output, train_trials, "train"),
        (test_output, test_trials, "test"),
    ]:
        output_data = {
            'trials': trials_list,
            'metadata': {
                'num_trials': len(trials_list),
                'split': split_name,
                'split_method': args.split_method,
                'train_ratio': args.train_ratio,
                'seed': args.seed,
                'source_file': str(trial_file),
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nSaved {split_name} split to {output_file}")
        print(f"  - {len(trials_list)} trials")


if __name__ == "__main__":
    main()




