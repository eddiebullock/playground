"""
Evaluation metrics for CAM Face-Voice Battery.

Implements metrics matching the original CAM analysis:
- Overall accuracy
- Per-emotion accuracy
- Per-concept accuracy
- Confusion matrices
- Concept-level recognition (4/5 correct = concept passed)
"""

from .metrics import (
    compute_accuracy,
    compute_per_emotion_accuracy,
    compute_per_concept_accuracy,
    compute_confusion_matrix,
    compute_concept_recognition_rate,
    EvaluationResults,
)

__all__ = [
    'compute_accuracy',
    'compute_per_emotion_accuracy',
    'compute_per_concept_accuracy',
    'compute_confusion_matrix',
    'compute_concept_recognition_rate',
    'EvaluationResults',
]


