"""
Evaluation components for LLM-augmented emotion recognition.
"""

from .three_way_comparison import run_three_way_comparison
from .metrics import compute_metrics, save_results

__all__ = ['run_three_way_comparison', 'compute_metrics', 'save_results']

