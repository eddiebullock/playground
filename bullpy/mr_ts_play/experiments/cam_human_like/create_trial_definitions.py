#!/usr/bin/env python3
"""
Utility script to create CAM trial definitions from dataset structure.

This script generates trial definitions following CAM Face-Voice Battery methodology:
1. Loading videos from the dataset
2. Grouping by emotion concept
3. Generating 5 trials per concept with counterbalanced face/voice distribution
4. Selecting foils using CAM taxonomy (different groups, appropriate difficulty)
5. Creating trial definitions in the required format

CAM Methodology:
- 5 trials per emotion concept
- Counterbalanced: 3 face + 2 voice OR 2 face + 3 voice per concept
- 4 options per trial: 1 target + 3 foils
- Foils from different emotion groups than target
- Foils from same or adjacent CAM levels
- Foils match valence/interpersonal theme
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import random

# Add parent directory to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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


def discover_stimuli_by_concept(data_root: str) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Discover stimuli grouped by emotion concept (concept = emotion label).
    
    Groups videos by emotion concept, then by modality (face/voice).
    
    Args:
        data_root: Root directory of CAM stimuli
    
    Returns:
        Dictionary mapping concept -> modality -> list of video info dicts
        Format: {
            "emotion_concept": {
                "face": [{"path": "...", "actor": "...", "scenario_id": "..."}, ...],
                "voice": [{"path": "...", "actor": "...", "scenario_id": "..."}, ...],
            }
        }
    """
    data_path = Path(data_root)
    concepts = defaultdict(lambda: {"face": [], "voice": []})
    
    for video_file in data_path.rglob("*.mov"):
        parsed = _parse_filename(video_file.name)
        if not parsed:
            continue
        
        concept = parsed['emotion']  # Concept = emotion label
        modality = parsed['modality']  # "face" or "voice"
        
        concepts[concept][modality].append({
            'path': str(video_file.relative_to(data_path)),
            'actor': parsed['actor'],
            'scenario_id': parsed['scenario_id'],
            'modality': modality,
        })
    
    return dict(concepts)


def generate_trials_for_concept(
    concept: str,
    face_stimuli: List[Dict],
    voice_stimuli: List[Dict],
    all_emotions: Set[str],
    num_trials: int = 5,
    seed: Optional[int] = None,
) -> List[Dict]:
    """
    Generate 5 trials for an emotion concept with counterbalanced face/voice distribution.
    
    CAM methodology: 5 trials per concept, either 3 face + 2 voice or 2 face + 3 voice.
    
    Args:
        concept: Emotion concept (target label)
        face_stimuli: List of face (visual) stimuli for this concept
        voice_stimuli: List of voice (audio) stimuli for this concept
        all_emotions: Set of all available emotions for foil selection
        num_trials: Number of trials per concept (default 5)
        seed: Random seed for reproducibility
    
    Returns:
        List of trial dictionaries
    """
    if seed is not None:
        random.seed(seed)
    
    # Determine face/voice distribution (counterbalanced)
    # Randomly choose 3+2 or 2+3 distribution
    num_face = 3 if random.random() < 0.5 else 2
    num_voice = num_trials - num_face
    
    # Sample stimuli
    sampled_face = random.sample(face_stimuli, min(num_face, len(face_stimuli)))
    sampled_voice = random.sample(voice_stimuli, min(num_voice, len(voice_stimuli)))
    
    # If we don't have enough stimuli, use what we have
    if len(sampled_face) < num_face:
        num_voice = num_trials - len(sampled_face)
        sampled_voice = random.sample(voice_stimuli, min(num_voice, len(voice_stimuli)))
    
    if len(sampled_voice) < num_voice:
        num_face = num_trials - len(sampled_voice)
        sampled_face = random.sample(face_stimuli, min(num_face, len(face_stimuli)))
    
    # Generate trials
    trials = []
    
    for stimulus in sampled_face + sampled_voice:
        # Generate foils using CAM taxonomy
        # Original CAM: foils from levels 4 and 5 (Golan et al., 2006)
        foils = get_foil_candidates(concept, all_emotions, num_foils=3, foil_levels=[4, 5])
        
        # Validate foils
        is_valid, errors = validate_trial_foils(concept, foils)
        if not is_valid:
            print(f"Warning: Foil validation failed for concept '{concept}': {errors}")
            # Still proceed, but log the issue
        
        # Create candidate labels: target + foils
        candidate_labels = [concept] + foils
        
        # Randomize order (CAM methodology: options presented in random order)
        random.shuffle(candidate_labels)
        
        # Find correct index
        correct_idx = candidate_labels.index(concept)
        
        trial = {
            'stimulus_path': stimulus['path'],
            'modality': stimulus['modality'],
            'correct_label': concept,
            'candidate_labels': candidate_labels,
            'correct_idx': correct_idx,
            'actor': stimulus['actor'],
            'scenario_id': stimulus['scenario_id'],
            'concept': concept,
        }
        
        trials.append(trial)
    
    return trials


def validate_all_trials(trials: List[Dict]) -> Tuple[bool, List[str]]:
    """
    Validate all trials conform to CAM rules.
    
    Args:
        trials: List of trial dictionaries
    
    Returns:
        Tuple of (all_valid, list_of_errors)
    """
    errors = []
    
    # Group by concept
    concept_trials = defaultdict(list)
    for trial in trials:
        concept = trial.get('concept')
        if concept:
            concept_trials[concept].append(trial)
    
    # Check: 5 trials per concept
    for concept, concept_trial_list in concept_trials.items():
        if len(concept_trial_list) != 5:
            errors.append(
                f"Concept '{concept}' has {len(concept_trial_list)} trials, expected 5"
            )
        
        # Check face/voice counterbalancing
        face_count = sum(1 for t in concept_trial_list if t['modality'] == 'face')
        voice_count = sum(1 for t in concept_trial_list if t['modality'] == 'voice')
        
        if not ((face_count == 3 and voice_count == 2) or (face_count == 2 and voice_count == 3)):
            errors.append(
                f"Concept '{concept}': face={face_count}, voice={voice_count} "
                f"(expected 3+2 or 2+3)"
            )
    
    # Check each trial's foil selection
    for trial in trials:
        target = trial['correct_label']
        foils = [label for label in trial['candidate_labels'] if label != target]
        
        is_valid, foil_errors = validate_trial_foils(target, foils)
        if not is_valid:
            errors.append(f"Trial {trial.get('trial_id', 'unknown')}: {foil_errors}")
    
    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(
        description="Create CAM trial definitions from dataset using CAM taxonomy"
    )
    parser.add_argument(
        '--data-root',
        type=str,
        required=True,
        help='Root directory of CAM stimuli'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSON file path'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--trials-per-concept',
        type=int,
        default=5,
        help='Number of trials per concept (default: 5)'
    )
    parser.add_argument(
        '--min-stimuli-per-concept',
        type=int,
        default=5,
        help='Minimum stimuli required per concept (default: 5)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate trials against CAM rules'
    )
    parser.add_argument(
        '--cam-20-only',
        action='store_true',
        help='Filter to only the 20 original CAM concepts (Golan et al., 2006)'
    )
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print(f"Discovering stimuli from {args.data_root}...")
    concepts_data = discover_stimuli_by_concept(args.data_root)
    
    # Filter to CAM 20 concepts if requested
    if args.cam_20_only:
        cam_20 = get_cam_20_concepts()
        original_count = len(concepts_data)
        concepts_data = {
            concept: modalities
            for concept, modalities in concepts_data.items()
            if is_cam_concept(concept)
        }
        print(f"Filtered to {len(concepts_data)} CAM concepts (from {original_count} total)")
        print(f"CAM 20 concepts: {sorted(cam_20)}")
        
        # Check which CAM concepts are missing
        found_concepts = set(concepts_data.keys())
        missing = [c for c in cam_20 if c.lower() not in [fc.lower() for fc in found_concepts]]
        if missing:
            print(f"Warning: Missing CAM concepts in dataset: {missing}")
    else:
        print(f"Found {len(concepts_data)} emotion concepts")
    
    # Get all unique emotions for foil generation
    all_emotions = set(concepts_data.keys())
    print(f"Found {len(all_emotions)} unique emotions/concepts")
    
    # Generate trials for each concept
    all_trials = []
    trial_id = 1
    
    concepts_with_insufficient_stimuli = []
    
    for concept, modalities in concepts_data.items():
        face_stimuli = modalities['face']
        voice_stimuli = modalities['voice']
        total_stimuli = len(face_stimuli) + len(voice_stimuli)
        
        if total_stimuli < args.min_stimuli_per_concept:
            concepts_with_insufficient_stimuli.append(concept)
            continue
        
        # Generate trials for this concept
        concept_trials = generate_trials_for_concept(
            concept=concept,
            face_stimuli=face_stimuli,
            voice_stimuli=voice_stimuli,
            all_emotions=all_emotions,
            num_trials=args.trials_per_concept,
            seed=args.seed + trial_id,  # Different seed per concept for reproducibility
        )
        
        # Add trial IDs
        for trial in concept_trials:
            trial['trial_id'] = f"trial_{trial_id:03d}"
            trial_id += 1
        
        all_trials.extend(concept_trials)
    
    if concepts_with_insufficient_stimuli:
        print(f"\nWarning: {len(concepts_with_insufficient_stimuli)} concepts have "
              f"< {args.min_stimuli_per_concept} stimuli and were skipped")
    
    print(f"\nGenerated {len(all_trials)} trials across {len(concepts_data) - len(concepts_with_insufficient_stimuli)} concepts")
    
    # Validate trials if requested
    if args.validate:
        print("\nValidating trials against CAM rules...")
        all_valid, errors = validate_all_trials(all_trials)
        
        if all_valid:
            print("✓ All trials pass validation")
        else:
            print(f"✗ Found {len(errors)} validation errors:")
            for error in errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")
    
    # Randomize trial order (CAM methodology: trials presented in random order)
    random.shuffle(all_trials)
    
    # Save trial definitions
    output_data = {
        'trials': all_trials,
        'metadata': {
            'num_trials': len(all_trials),
            'num_concepts': len(concepts_data) - len(concepts_with_insufficient_stimuli),
            'trials_per_concept': args.trials_per_concept,
            'seed': args.seed,
            'foil_selection': 'CAM taxonomy-based',
            'counterbalancing': '3 face + 2 voice OR 2 face + 3 voice per concept',
            'cam_20_only': args.cam_20_only,
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nTrial definitions saved to {args.output}")
    print(f"  - {len(all_trials)} trials")
    print(f"  - {len(concepts_data) - len(concepts_with_insufficient_stimuli)} concepts")
    print(f"  - {args.trials_per_concept} trials per concept")
    print(f"  - Foils selected using CAM taxonomy")


if __name__ == "__main__":
    main()

