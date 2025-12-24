#!/usr/bin/env python3
"""
Create CAM trial definitions from ALL available files in the dataset.

This script generates trial definitions following CAM methodology but uses
ALL available valid files (not just 100 pre-defined trials).

For high-impact research: Maximizes data usage while maintaining experimental rigor.

Procedure:
1. Discover all valid CAM files in dataset directory
2. Group by emotion concept and modality
3. Generate forced-choice trials (4 options each)
4. Proper foil selection (semantically different concepts)
5. Create train/test splits (80/20)
6. Output trial definitions in CAM-compatible format

Key Differences from Pre-defined Trials:
- Uses ALL available valid files (not just 100)
- Generates 10-15 trials per concept (if enough files)
- More training data = better model performance
- Still maintains 20 concepts for comparison
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

from experiments.cam_human_like.dataset import _parse_filename
from experiments.cam_human_like.taxonomy import (
    get_foil_candidates,
    validate_trial_foils,
    CAM_TAXONOMY,
)
from experiments.cam_human_like.cam_20_concepts import (
    get_cam_20_concepts,
    is_cam_concept,
)


def discover_all_cam_files(data_root: str, min_file_size_kb: int = 50) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Discover ALL valid CAM files grouped by concept and modality.
    
    Args:
        data_root: Root directory of CAM dataset
        min_file_size_kb: Minimum file size to consider valid (default: 50KB)
    
    Returns:
        Dictionary mapping concept -> modality -> list of file info dicts
        Format: {
            "concept_name": {
                "face": [{"path": "...", "actor": "...", "scenario_id": "..."}, ...],
                "voice": [{"path": "...", "actor": "...", "scenario_id": "..."}, ...],
            }
        }
    """
    data_path = Path(data_root)
    concepts = defaultdict(lambda: {"face": [], "voice": []})
    
    print(f"Scanning CAM dataset: {data_root}")
    print(f"Filtering files > {min_file_size_kb}KB...")
    
    valid_files = 0
    skipped_files = 0
    
    for video_file in data_path.rglob("*.mov"):
        # Check file size (skip corrupted files)
        try:
            file_size = video_file.stat().st_size
            if file_size < min_file_size_kb * 1024:
                skipped_files += 1
                continue
        except (OSError, PermissionError):
            skipped_files += 1
            continue
        
        # Parse filename
        parsed = _parse_filename(video_file.name)
        if not parsed:
            skipped_files += 1
            continue
        
        concept = parsed['emotion']
        modality_code = parsed['modality']  # 'V' or 'T'
        
        # Convert modality code to dictionary key
        # V = Visual (face), T = Textual (voice)
        modality = "face" if modality_code == "V" else "voice"
        
        # Only include CAM 20 concepts
        if not is_cam_concept(concept):
            continue
        
        concepts[concept][modality].append({
            'path': str(video_file.relative_to(data_path)),
            'actor': parsed['actor'],
            'scenario_id': parsed['scenario_id'],
            'modality': modality_code,  # Keep original 'V' or 'T' in the data
            'full_path': str(video_file),
        })
        
        valid_files += 1
    
    print(f"Found {valid_files} valid files")
    print(f"Skipped {skipped_files} files (corrupted or invalid)")
    print(f"Concepts found: {len(concepts)}")
    
    return dict(concepts)


def select_foils(
    target_concept: str,
    all_concepts: Set[str],
    num_foils: int = 3,
    seed: Optional[int] = None
) -> List[str]:
    """
    Select foils (wrong answers) for a trial.
    
    Uses CAM taxonomy to select semantically different concepts.
    
    Args:
        target_concept: The correct emotion concept
        all_concepts: Set of all available concepts
        num_foils: Number of foils to select (default: 3)
        seed: Random seed for reproducibility
    
    Returns:
        List of foil concept names
    """
    if seed is not None:
        random.seed(seed)
    
    # Get foil candidates from taxonomy
    foil_candidates = get_foil_candidates(target_concept, all_concepts)
    
    # If not enough candidates, use all concepts except target
    if len(foil_candidates) < num_foils:
        foil_candidates = [c for c in all_concepts if c != target_concept]
    
    # Randomly sample foils
    if len(foil_candidates) >= num_foils:
        foils = random.sample(foil_candidates, num_foils)
    else:
        # Not enough candidates, use what we have and repeat if needed
        foils = foil_candidates.copy()
        while len(foils) < num_foils:
            foils.append(random.choice(list(all_concepts - {target_concept})))
        foils = foils[:num_foils]
    
    return foils


def generate_trials_for_concept(
    concept: str,
    face_stimuli: List[Dict],
    voice_stimuli: List[Dict],
    all_concepts: Set[str],
    num_trials: int = 10,
    seed: Optional[int] = None,
) -> List[Dict]:
    """
    Generate trials for an emotion concept with counterbalanced face/voice distribution.
    
    Args:
        concept: Emotion concept (target label)
        face_stimuli: List of face stimuli for this concept
        voice_stimuli: List of voice stimuli for this concept
        all_concepts: Set of all available concepts for foil selection
        num_trials: Number of trials per concept (default: 10)
        seed: Random seed for reproducibility
    
    Returns:
        List of trial dictionaries
    """
    if seed is not None:
        random.seed(seed)
    
    # Determine face/voice distribution (counterbalanced)
    if not voice_stimuli or len(voice_stimuli) == 0:
        # Only face available
        num_face = num_trials
        num_voice = 0
    elif not face_stimuli or len(face_stimuli) == 0:
        # Only voice available
        num_face = 0
        num_voice = num_trials
    else:
        # Both available: counterbalanced (roughly 50/50)
        num_face = num_trials // 2
        num_voice = num_trials - num_face
    
    # Sample stimuli
    sampled_face = random.sample(face_stimuli, min(num_face, len(face_stimuli))) if face_stimuli and num_face > 0 else []
    sampled_voice = random.sample(voice_stimuli, min(num_voice, len(voice_stimuli))) if voice_stimuli and num_voice > 0 else []
    
    # Adjust if we don't have enough
    if len(sampled_face) < num_face and len(voice_stimuli) > 0:
        num_voice = num_trials - len(sampled_face)
        sampled_voice = random.sample(voice_stimuli, min(num_voice, len(voice_stimuli))) if voice_stimuli else []
    
    if len(sampled_voice) < num_voice and len(face_stimuli) > 0:
        num_face = num_trials - len(sampled_voice)
        sampled_face = random.sample(face_stimuli, min(num_face, len(face_stimuli))) if face_stimuli else []
    
    # Generate trials
    trials = []
    trial_id = 1
    
    for stimulus in sampled_face + sampled_voice:
        # Determine modality
        mod = "face" if stimulus in sampled_face else "voice"
        
        # Generate foils
        foils = select_foils(concept, all_concepts, num_foils=3, seed=seed + trial_id)
        
        # Create candidate labels: target + foils
        candidate_labels = [concept] + foils
        
        # Randomize order (CAM methodology: options presented in random order)
        random.shuffle(candidate_labels)
        
        # Find correct index
        correct_idx = candidate_labels.index(concept)
        
        trial = {
            'trial_id': f'trial_{trial_id:03d}',
            'stimulus_path': stimulus['path'],
            'modality': mod,
            'correct_label': concept,
            'candidate_labels': candidate_labels,
            'correct_idx': correct_idx,
            'actor': stimulus.get('actor', 'unknown'),
            'scenario_id': stimulus.get('scenario_id', 'unknown'),
            'concept': concept,
        }
        
        trials.append(trial)
        trial_id += 1
    
    return trials


def create_train_test_split(
    all_trials: List[Dict],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Create train/test split from all trials.
    
    Ensures each concept has trials in both splits (concept-balanced).
    
    Args:
        all_trials: List of all trial dictionaries
        train_ratio: Proportion of trials for training (default: 0.8)
        seed: Random seed
    
    Returns:
        Tuple of (train_trials, test_trials)
    """
    random.seed(seed)
    
    # Group trials by concept
    concept_trials = defaultdict(list)
    for trial in all_trials:
        concept = trial.get('concept', trial.get('correct_label'))
        concept_trials[concept].append(trial)
    
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
        description="Create CAM trial definitions from ALL available files (high-impact methodology)"
    )
    parser.add_argument(
        '--cam-dir',
        type=str,
        required=True,
        help='Root directory of CAM dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/cam_replication',
        help='Output directory for trial definitions'
    )
    parser.add_argument(
        '--trials-per-concept',
        type=int,
        default=10,
        help='Number of trials per concept (default: 10, can increase if more files available)'
    )
    parser.add_argument(
        '--min-file-size-kb',
        type=int,
        default=50,
        help='Minimum file size in KB to consider valid (default: 50)'
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
    
    random.seed(args.seed)
    
    print("=" * 70)
    print("CAM Trial Generation from ALL Available Files")
    print("=" * 70)
    print(f"\nDataset: {args.cam_dir}")
    print(f"Trials per concept: {args.trials_per_concept}")
    print(f"Min file size: {args.min_file_size_kb}KB")
    print()
    
    # Discover all files
    concepts_data = discover_all_cam_files(args.cam_dir, args.min_file_size_kb)
    
    # Get CAM 20 concepts
    cam_20_concepts = set(get_cam_20_concepts())
    available_concepts = set(concepts_data.keys())
    
    print(f"\nCAM 20 concepts: {len(cam_20_concepts)}")
    print(f"Available concepts: {len(available_concepts)}")
    print(f"Concepts: {sorted(available_concepts)}")
    
    # Generate trials for each concept
    all_trials = []
    
    for concept in sorted(cam_20_concepts):
        if concept not in concepts_data:
            print(f"⚠️  Warning: No files found for concept '{concept}'")
            continue
        
        face_stimuli = concepts_data[concept].get('face', [])
        voice_stimuli = concepts_data[concept].get('voice', [])
        
        print(f"\n{concept}:")
        print(f"  Face files: {len(face_stimuli)}")
        print(f"  Voice files: {len(voice_stimuli)}")
        
        # Generate trials
        concept_trials = generate_trials_for_concept(
            concept,
            face_stimuli,
            voice_stimuli,
            cam_20_concepts,
            num_trials=args.trials_per_concept,
            seed=args.seed
        )
        
        print(f"  Generated: {len(concept_trials)} trials")
        all_trials.extend(concept_trials)
    
    print(f"\nTotal trials generated: {len(all_trials)}")
    
    # Create train/test split
    train_trials, test_trials = create_train_test_split(
        all_trials,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    
    print(f"Train trials: {len(train_trials)}")
    print(f"Test trials: {len(test_trials)}")
    
    # Save trial definitions
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = output_dir / "cam_trial_definitions_train_all_files.json"
    test_file = output_dir / "cam_trial_definitions_test_all_files.json"
    
    with open(train_file, 'w') as f:
        json.dump({'trials': train_trials}, f, indent=2)
    
    with open(test_file, 'w') as f:
        json.dump({'trials': test_trials}, f, indent=2)
    
    print(f"\n✅ Trial definitions saved:")
    print(f"  Train: {train_file}")
    print(f"  Test: {test_file}")
    
    # Summary statistics
    print(f"\nSummary:")
    print(f"  Total concepts: {len(cam_20_concepts)}")
    print(f"  Total trials: {len(all_trials)}")
    print(f"  Train trials: {len(train_trials)} ({len(train_trials)/len(all_trials)*100:.1f}%)")
    print(f"  Test trials: {len(test_trials)} ({len(test_trials)/len(all_trials)*100:.1f}%)")
    
    # Per-concept breakdown
    train_by_concept = defaultdict(int)
    test_by_concept = defaultdict(int)
    
    for trial in train_trials:
        concept = trial.get('concept', trial.get('correct_label'))
        train_by_concept[concept] += 1
    
    for trial in test_trials:
        concept = trial.get('concept', trial.get('correct_label'))
        test_by_concept[concept] += 1
    
    print(f"\nTrials per concept (train/test):")
    for concept in sorted(cam_20_concepts):
        train_count = train_by_concept.get(concept, 0)
        test_count = test_by_concept.get(concept, 0)
        print(f"  {concept}: {train_count}/{test_count}")


if __name__ == '__main__':
    main()

