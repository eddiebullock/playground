"""
Forced-choice trial logic for CAM Face-Voice Battery.

Implements the 4-option forced-choice decision procedure from the original
CAM methodology (Golan et al., 2006).

In the original CAM:
- Participants view/hear a stimulus (video with audio)
- Four numbered adjectives (1-4) are presented
- Participants press 1, 2, 3, or 4 to select their answer
- No feedback is given
- Response time is unrestricted

This module replicates this procedure computationally:
- Model processes stimulus and scores all four candidate labels
- Decision is restricted to the four trial options (no other labels considered)
- Highest-scoring option is selected as the model's choice
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

from ..dataset import CAMTrial
from ..models.base import ModelWrapper, ModelOutput


@dataclass
class ForcedChoiceTrial:
    """
    Represents a single forced-choice trial with model prediction.
    
    Contains:
    - The original CAM trial definition
    - Model scores for each candidate label
    - Model's predicted choice (index 0-3)
    - Whether prediction is correct
    """
    trial: CAMTrial
    model_scores: Dict[str, float]  # Scores for each candidate label
    predicted_idx: int  # Index of predicted label (0-3)
    predicted_label: str  # The predicted label string
    is_correct: bool  # Whether prediction matches correct answer
    confidence: float  # Confidence score (e.g., softmax of scores)
    
    def __post_init__(self):
        """Validate trial structure."""
        if self.predicted_idx not in range(4):
            raise ValueError(f"predicted_idx must be 0-3, got {self.predicted_idx}")
        if self.predicted_label != self.trial.candidate_labels[self.predicted_idx]:
            raise ValueError("predicted_label mismatch with predicted_idx")


def run_forced_choice_trial(
    trial: CAMTrial,
    model: ModelWrapper,
    temperature: float = 1.0,
) -> ForcedChoiceTrial:
    """
    Run a single CAM forced-choice trial.
    
    This function implements the core CAM decision logic:
    1. Model processes stimulus and scores all four candidate labels
    2. Decision is restricted to the four trial options (forced-choice)
    3. Highest-scoring option is selected
    
    This matches the human participant procedure where:
    - Stimulus is presented
    - Four options are shown
    - Participant must choose one of the four (no "other" option)
    
    Args:
        trial: CAM trial definition with stimulus and candidate labels
        model: Model wrapper that can score labels against stimuli
        temperature: Temperature scaling for scores (1.0 = no scaling)
                    Higher temperature = softer distribution
    
    Returns:
        ForcedChoiceTrial with model prediction and correctness
    """
    # Get model scores for all candidate labels
    model_output = model.score_labels(
        stimulus_path=trial.stimulus_path,
        candidate_labels=trial.candidate_labels,
        modality=trial.modality,
    )
    
    scores = model_output.label_scores
    
    # Ensure all candidate labels have scores
    for label in trial.candidate_labels:
        if label not in scores:
            raise ValueError(f"Model did not return score for label: {label}")
    
    # Extract scores in order of candidate_labels
    score_values = [scores[label] for label in trial.candidate_labels]
    
    # Apply temperature scaling (optional calibration)
    if temperature != 1.0:
        score_values = [s / temperature for s in score_values]
    
    # Select highest-scoring option (forced-choice decision)
    predicted_idx = int(np.argmax(score_values))
    predicted_label = trial.candidate_labels[predicted_idx]
    
    # Compute confidence (softmax probability of predicted label)
    score_array = np.array(score_values)
    exp_scores = np.exp(score_array - np.max(score_array))  # Numerical stability
    probs = exp_scores / exp_scores.sum()
    confidence = float(probs[predicted_idx])
    
    # Check correctness
    is_correct = (predicted_idx == trial.correct_idx)
    
    return ForcedChoiceTrial(
        trial=trial,
        model_scores=scores,
        predicted_idx=predicted_idx,
        predicted_label=predicted_label,
        is_correct=is_correct,
        confidence=confidence,
    )


def run_batch_trials(
    trials: List[CAMTrial],
    model: ModelWrapper,
    temperature: float = 1.0,
    verbose: bool = True,
) -> List[ForcedChoiceTrial]:
    """
    Run multiple forced-choice trials in batch.
    
    Args:
        trials: List of CAM trial definitions
        model: Model wrapper
        temperature: Temperature scaling for scores
        verbose: Whether to print progress
    
    Returns:
        List of ForcedChoiceTrial results
    """
    results = []
    
    for i, trial in enumerate(trials):
        if verbose and (i + 1) % 10 == 0:
            print(f"Processing trial {i + 1}/{len(trials)}...")
        
        result = run_forced_choice_trial(trial, model, temperature)
        results.append(result)
    
    return results









