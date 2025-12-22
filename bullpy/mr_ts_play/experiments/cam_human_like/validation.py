"""
Trial validation functions for CAM Face-Voice Battery.

Validates that generated trials conform to CAM methodology rules:
- 5 trials per concept
- Counterbalanced face/voice distribution (3+2 or 2+3)
- Foils from different emotion groups than target
- Correct number of options (4: 1 target + 3 foils)
"""

from typing import List, Dict, Tuple
from collections import defaultdict

from .dataset import CAMTrial
from .taxonomy import validate_trial_foils


def validate_trial_structure(trial: CAMTrial) -> Tuple[bool, List[str]]:
    """
    Validate a single trial's structure.
    
    Args:
        trial: CAM trial to validate
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check: 4 candidate labels
    if len(trial.candidate_labels) != 4:
        errors.append(
            f"Trial {trial.trial_id}: Expected 4 candidate labels, got {len(trial.candidate_labels)}"
        )
    
    # Check: correct_idx in valid range
    if trial.correct_idx not in range(4):
        errors.append(
            f"Trial {trial.trial_id}: correct_idx must be 0-3, got {trial.correct_idx}"
        )
    
    # Check: correct_label matches candidate_labels[correct_idx]
    if trial.correct_idx < len(trial.candidate_labels):
        if trial.candidate_labels[trial.correct_idx] != trial.correct_label:
            errors.append(
                f"Trial {trial.trial_id}: correct_label '{trial.correct_label}' "
                f"does not match candidate_labels[{trial.correct_idx}] "
                f"'{trial.candidate_labels[trial.correct_idx]}'"
            )
    
    # Check: modality is valid
    if trial.modality not in ["face", "voice"]:
        errors.append(
            f"Trial {trial.trial_id}: modality must be 'face' or 'voice', got '{trial.modality}'"
        )
    
    # Check: no duplicate labels
    if len(trial.candidate_labels) != len(set(trial.candidate_labels)):
        errors.append(
            f"Trial {trial.trial_id}: Duplicate labels in candidate_labels"
        )
    
    return len(errors) == 0, errors


def validate_trial_foils_cam(trial: CAMTrial) -> Tuple[bool, List[str]]:
    """
    Validate that trial's foils follow CAM methodology.
    
    Args:
        trial: CAM trial to validate
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    foils = [label for label in trial.candidate_labels if label != trial.correct_label]
    return validate_trial_foils(trial.correct_label, foils)


def validate_concept_trials(trials: List[CAMTrial], concept: str) -> Tuple[bool, List[str]]:
    """
    Validate trials for a specific concept.
    
    Checks:
    - Exactly 5 trials per concept
    - Counterbalanced face/voice distribution (3+2 or 2+3)
    
    Args:
        trials: List of all trials
        concept: Concept name to validate
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    concept_trials = [t for t in trials if t.concept == concept]
    errors = []
    
    # Check: 5 trials per concept
    if len(concept_trials) != 5:
        errors.append(
            f"Concept '{concept}': Expected 5 trials, got {len(concept_trials)}"
        )
    
    # Check: counterbalanced face/voice distribution
    face_count = sum(1 for t in concept_trials if t.modality == "face")
    voice_count = sum(1 for t in concept_trials if t.modality == "voice")
    
    if not ((face_count == 3 and voice_count == 2) or (face_count == 2 and voice_count == 3)):
        errors.append(
            f"Concept '{concept}': Face={face_count}, Voice={voice_count} "
            f"(expected 3+2 or 2+3)"
        )
    
    return len(errors) == 0, errors


def validate_all_trials_cam(trials: List[CAMTrial]) -> Tuple[bool, List[str]]:
    """
    Comprehensive validation of all trials against CAM rules.
    
    Args:
        trials: List of all CAM trials
    
    Returns:
        Tuple of (all_valid, list_of_all_errors)
    """
    all_errors = []
    
    # Group by concept
    concepts = set(trial.concept for trial in trials if trial.concept)
    
    # Validate each trial's structure
    for trial in trials:
        is_valid, errors = validate_trial_structure(trial)
        if not is_valid:
            all_errors.extend(errors)
        
        # Validate foil selection
        is_valid, errors = validate_trial_foils_cam(trial)
        if not is_valid:
            all_errors.extend(errors)
    
    # Validate concept-level constraints
    for concept in concepts:
        is_valid, errors = validate_concept_trials(trials, concept)
        if not is_valid:
            all_errors.extend(errors)
    
    return len(all_errors) == 0, all_errors


def print_validation_report(trials: List[CAMTrial]) -> None:
    """
    Print a comprehensive validation report.
    
    Args:
        trials: List of all CAM trials
    """
    print("\n" + "="*60)
    print("CAM TRIAL VALIDATION REPORT")
    print("="*60)
    
    # Overall statistics
    concepts = set(trial.concept for trial in trials if trial.concept)
    print(f"\nTotal trials: {len(trials)}")
    print(f"Total concepts: {len(concepts)}")
    print(f"Average trials per concept: {len(trials) / len(concepts) if concepts else 0:.1f}")
    
    # Modality distribution
    face_trials = sum(1 for t in trials if t.modality == "face")
    voice_trials = sum(1 for t in trials if t.modality == "voice")
    print(f"\nModality distribution:")
    print(f"  Face trials: {face_trials} ({face_trials/len(trials)*100:.1f}%)")
    print(f"  Voice trials: {voice_trials} ({voice_trials/len(trials)*100:.1f}%)")
    
    # Concept-level statistics
    print(f"\nConcept-level statistics:")
    concept_trial_counts = defaultdict(int)
    for trial in trials:
        if trial.concept:
            concept_trial_counts[trial.concept] += 1
    
    for concept, count in sorted(concept_trial_counts.items()):
        concept_trials = [t for t in trials if t.concept == concept]
        face_count = sum(1 for t in concept_trials if t.modality == "face")
        voice_count = sum(1 for t in concept_trials if t.modality == "voice")
        print(f"  {concept}: {count} trials (face={face_count}, voice={voice_count})")
    
    # Validation
    print(f"\nValidation:")
    all_valid, errors = validate_all_trials_cam(trials)
    
    if all_valid:
        print("  ✓ All trials pass CAM validation")
    else:
        print(f"  ✗ Found {len(errors)} validation errors:")
        for error in errors[:20]:  # Show first 20 errors
            print(f"    - {error}")
        if len(errors) > 20:
            print(f"    ... and {len(errors) - 20} more errors")
    
    print("="*60 + "\n")





