"""
Module: imbalance_handler.py
Purpose: Handle class imbalance using configurable strategies (resampling, class weights, SMOTE, etc.)
"""
from typing import Tuple, Optional
import pandas as pd
import numpy as np

# Import imblearn for advanced resampling methods
try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

def handle_imbalance(X: pd.DataFrame, y: pd.Series, method: str = 'none', random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply the selected class imbalance handling method.
    Args:
        X: Features DataFrame
        y: Target Series
        method: 'none', 'oversample', 'undersample', 'smote', 'class_weight'
        random_state: for reproducibility
    Returns:
        X_res, y_res: Resampled features and target
    """
    if method == 'none':
        return X, y
    elif method == 'oversample':
        if not IMBLEARN_AVAILABLE:
            raise ImportError("imblearn not available. Install with: pip install imbalanced-learn")
        sampler = RandomOverSampler(random_state=random_state)
        X_res, y_res = sampler.fit_resample(X, y)
        return X_res, y_res
    elif method == 'undersample':
        if not IMBLEARN_AVAILABLE:
            raise ImportError("imblearn not available. Install with: pip install imbalanced-learn")
        sampler = RandomUnderSampler(random_state=random_state)
        X_res, y_res = sampler.fit_resample(X, y)
        return X_res, y_res
    elif method == 'smote':
        if not IMBLEARN_AVAILABLE:
            raise ImportError("imblearn not available. Install with: pip install imbalanced-learn")
        sampler = SMOTE(random_state=random_state)
        X_res, y_res = sampler.fit_resample(X, y)
        return X_res, y_res
    elif method == 'class_weight':
        # No resampling, just return as is (handled in model)
        return X, y
    else:
        raise ValueError(f"Unknown imbalance handling method: {method}") 