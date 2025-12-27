"""
Evaluation scripts for basic emotions LLM augmentation (reused from llm_augmented_emotion_recognition).
"""

import sys
from pathlib import Path

# Import evaluation from existing LLM augmentation experiment
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "llm_augmented_emotion_recognition" / "evaluation"))

from three_way_comparison import run_three_way_comparison
from metrics import compute_metrics

__all__ = ['run_three_way_comparison', 'compute_metrics']

