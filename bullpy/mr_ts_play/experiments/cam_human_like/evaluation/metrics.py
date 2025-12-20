"""
Evaluation metrics for CAM Face-Voice Battery.

Implements metrics matching the original CAM analysis methodology:
- Overall accuracy: proportion of correct trials
- Per-emotion accuracy: accuracy for each emotion label
- Per-concept accuracy: accuracy for each emotion concept
- Confusion matrices: detailed error analysis
- Concept recognition: 4/5 correct items = concept passed (original CAM criterion)

These metrics allow comparison with human performance reported in Golan et al. (2006).
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from collections import defaultdict

from ..trials.forced_choice import ForcedChoiceTrial


@dataclass
class EvaluationResults:
    """
    Comprehensive evaluation results for CAM experiment.
    
    Matches the metrics reported in the original CAM paper:
    - Facial scale score (face trials only)
    - Vocal scale score (voice trials only)
    - Overall score (all trials)
    - Number of concepts passed (4/5 correct items)
    """
    overall_accuracy: float
    face_accuracy: float
    voice_accuracy: float
    per_emotion_accuracy: Dict[str, float]
    per_concept_accuracy: Dict[str, float]
    concept_recognition_rate: float  # Proportion of concepts passed
    confusion_matrix: pd.DataFrame
    num_trials: int
    num_face_trials: int
    num_voice_trials: int
    num_concepts: int
    metadata: Dict = field(default_factory=dict)


def compute_accuracy(trials: List[ForcedChoiceTrial]) -> float:
    """
    Compute overall accuracy across all trials.
    
    This corresponds to the "CAM overall score" in the original paper,
    calculated as the number of correct answers across the whole battery.
    
    Args:
        trials: List of trial results
    
    Returns:
        Accuracy as proportion (0.0 to 1.0)
    """
    if not trials:
        return 0.0
    
    correct = sum(1 for t in trials if t.is_correct)
    return correct / len(trials)


def compute_per_emotion_accuracy(trials: List[ForcedChoiceTrial]) -> Dict[str, float]:
    """
    Compute accuracy for each emotion label.
    
    This provides detailed breakdown of model performance across different
    mental states, allowing identification of which emotions are easier/harder.
    
    Args:
        trials: List of trial results
    
    Returns:
        Dictionary mapping emotion labels to accuracy scores
    """
    emotion_counts = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for trial in trials:
        emotion = trial.trial.correct_label
        emotion_counts[emotion]['total'] += 1
        if trial.is_correct:
            emotion_counts[emotion]['correct'] += 1
    
    return {
        emotion: counts['correct'] / counts['total'] if counts['total'] > 0 else 0.0
        for emotion, counts in emotion_counts.items()
    }


def compute_per_concept_accuracy(trials: List[ForcedChoiceTrial]) -> Dict[str, float]:
    """
    Compute accuracy for each emotion concept.
    
    In the original CAM, concepts are groups of related emotions.
    Each concept has 5 items (trials), and concept-level accuracy
    shows how well the model recognizes each concept overall.
    
    Args:
        trials: List of trial results
    
    Returns:
        Dictionary mapping concept names to accuracy scores
    """
    concept_counts = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for trial in trials:
        concept = trial.trial.concept
        if concept:
            concept_counts[concept]['total'] += 1
            if trial.is_correct:
                concept_counts[concept]['correct'] += 1
    
    return {
        concept: counts['correct'] / counts['total'] if counts['total'] > 0 else 0.0
        for concept, counts in concept_counts.items()
    }


def compute_confusion_matrix(
    trials: List[ForcedChoiceTrial],
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Compute confusion matrix for predictions.
    
    Rows = true labels (correct answers)
    Columns = predicted labels
    
    This allows detailed error analysis to see which emotions
    are confused with each other.
    
    Args:
        trials: List of trial results
        normalize: If True, normalize rows to show proportions
    
    Returns:
        DataFrame with confusion matrix
    """
    # Get all unique labels
    all_labels = set()
    for trial in trials:
        all_labels.add(trial.trial.correct_label)
        all_labels.add(trial.predicted_label)
    
    all_labels = sorted(list(all_labels))
    label_to_idx = {label: i for i, label in enumerate(all_labels)}
    
    # Build confusion matrix
    n = len(all_labels)
    cm = np.zeros((n, n), dtype=float)
    
    for trial in trials:
        true_idx = label_to_idx[trial.trial.correct_label]
        pred_idx = label_to_idx[trial.predicted_label]
        cm[true_idx, pred_idx] += 1
    
    # Normalize rows if requested
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        cm = cm / row_sums
    
    # Create DataFrame
    cm_df = pd.DataFrame(
        cm,
        index=all_labels,
        columns=all_labels,
    )
    
    return cm_df


def compute_concept_recognition_rate(trials: List[ForcedChoiceTrial]) -> Tuple[float, Dict[str, bool]]:
    """
    Compute concept recognition rate using CAM criterion.
    
    Original CAM methodology:
    - A concept is considered "recognized" if at least 4 out of 5 items
      are answered correctly
    - This corresponds to the "number of concepts passed" metric
    
    Args:
        trials: List of trial results
    
    Returns:
        Tuple of (recognition_rate, concept_passed_dict)
        recognition_rate: Proportion of concepts that passed (0.0 to 1.0)
        concept_passed_dict: Dictionary mapping concepts to True/False
    """
    concept_trials = defaultdict(list)
    
    for trial in trials:
        concept = trial.trial.concept
        if concept:
            concept_trials[concept].append(trial)
    
    concept_passed = {}
    
    for concept, concept_trial_list in concept_trials.items():
        correct_count = sum(1 for t in concept_trial_list if t.is_correct)
        # CAM criterion: 4/5 correct = concept passed
        concept_passed[concept] = (correct_count >= 4)
    
    if not concept_passed:
        return 0.0, {}
    
    recognition_rate = sum(concept_passed.values()) / len(concept_passed)
    
    return recognition_rate, concept_passed


def evaluate_trials(trials: List[ForcedChoiceTrial]) -> EvaluationResults:
    """
    Comprehensive evaluation of trial results.
    
    Computes all metrics matching the original CAM analysis:
    - Overall accuracy
    - Facial scale score (face trials)
    - Vocal scale score (voice trials)
    - Per-emotion accuracy
    - Per-concept accuracy
    - Concept recognition rate
    - Confusion matrix
    
    Args:
        trials: List of trial results
    
    Returns:
        EvaluationResults with all computed metrics
    """
    if not trials:
        raise ValueError("Cannot evaluate empty trial list")
    
    # Overall accuracy
    overall_accuracy = compute_accuracy(trials)
    
    # Split by modality
    face_trials = [t for t in trials if t.trial.modality == "face"]
    voice_trials = [t for t in trials if t.trial.modality == "voice"]
    
    face_accuracy = compute_accuracy(face_trials) if face_trials else 0.0
    voice_accuracy = compute_accuracy(voice_trials) if voice_trials else 0.0
    
    # Per-emotion and per-concept accuracy
    per_emotion_accuracy = compute_per_emotion_accuracy(trials)
    per_concept_accuracy = compute_per_concept_accuracy(trials)
    
    # Concept recognition rate
    concept_recognition_rate, concept_passed = compute_concept_recognition_rate(trials)
    
    # Confusion matrix
    confusion_matrix = compute_confusion_matrix(trials)
    
    # Get unique concepts
    concepts = set(t.trial.concept for t in trials if t.trial.concept)
    
    return EvaluationResults(
        overall_accuracy=overall_accuracy,
        face_accuracy=face_accuracy,
        voice_accuracy=voice_accuracy,
        per_emotion_accuracy=per_emotion_accuracy,
        per_concept_accuracy=per_concept_accuracy,
        concept_recognition_rate=concept_recognition_rate,
        confusion_matrix=confusion_matrix,
        num_trials=len(trials),
        num_face_trials=len(face_trials),
        num_voice_trials=len(voice_trials),
        num_concepts=len(concepts),
        metadata={
            'concept_passed': concept_passed,
        }
    )


