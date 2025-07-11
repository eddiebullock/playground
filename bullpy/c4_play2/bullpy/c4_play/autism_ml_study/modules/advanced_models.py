"""
Module: advanced_models.py
Purpose: Provide a unified interface for advanced models (XGBoost, LightGBM, MLP, SVM, etc.)
"""
from typing import Any, Dict
import numpy as np

# Import model classes
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

def get_model(model_name: str, model_params: Dict[str, Any] = None):
    """
    Return an instantiated model based on the name and parameters.
    Args:
        model_name: 'xgboost', 'lightgbm', 'mlp', 'svm', 'random_forest', 'logistic_regression'
        model_params: dict of model-specific parameters
    Returns:
        Instantiated model
    """
    if model_params is None:
        model_params = {}
    
    if model_name == 'xgboost':
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available. Install with: pip install xgboost")
        return XGBClassifier(**model_params)
    elif model_name == 'lightgbm':
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available. Install with: pip install lightgbm")
        return LGBMClassifier(**model_params)
    elif model_name == 'mlp':
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available. Install with: pip install scikit-learn")
        return MLPClassifier(**model_params)
    elif model_name == 'svm':
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available. Install with: pip install scikit-learn")
        return SVC(**model_params)
    elif model_name == 'random_forest':
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available. Install with: pip install scikit-learn")
        return RandomForestClassifier(**model_params)
    elif model_name == 'logistic_regression':
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available. Install with: pip install scikit-learn")
        return LogisticRegression(**model_params)
    else:
        raise ValueError(f"Unknown model: {model_name}") 