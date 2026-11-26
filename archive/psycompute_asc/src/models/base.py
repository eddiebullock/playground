"""
Base model class for all machine learning models.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
import joblib
from pathlib import Path
import json
from typing import Dict, Tuple, Any, Optional

class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, model_name: str, assessment_type: str):
        """
        Initialize the base model.
        
        Args:
            model_name: Name of the model
            assessment_type: Type of assessment ('AQ', 'SQ', or 'EQ')
        """
        self.model_name = model_name
        self.assessment_type = assessment_type
        self.model = None
        self.scaler = StandardScaler()
        self.metrics = {}
        self.best_params = None
        self.feature_importance = None
        self.X_train = None
        self.y_train = None
    
    @abstractmethod
    def _initialize_model(self) -> Any:
        """Initialize the specific model implementation."""
        pass
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame containing the data
            
        Returns:
            Tuple of (X, y) where X is the feature matrix and y is the target vector
        """
        # Get question columns
        question_cols = [col for col in df.columns 
                        if col.startswith(f'{self.assessment_type}_') 
                        and not col.endswith('_score') 
                        and col != 'asd_diagnosis']
        
        # Fill missing values with median
        df[question_cols] = df[question_cols].fillna(df[question_cols].median())
        
        # Prepare features and target
        X = df[question_cols].values
        y = df['asd_diagnosis'].values.astype(int)
        
        # Split data and store training data
        self.X_train, X_test, self.y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return self.X_train, self.y_train
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       y_prob: np.ndarray) -> Dict:
        """
        Compute various performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            
        Returns:
            Dictionary containing various metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_true, y_prob),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        # Compute specificity
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        metrics['roc_curve'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
        
        # Class distribution
        metrics['class_distribution'] = {
            'train': {
                'class_0': int(np.sum(self.y_train == 0)),
                'class_1': int(np.sum(self.y_train == 1))
            },
            'test': {
                'class_0': int(np.sum(y_true == 0)),
                'class_1': int(np.sum(y_true == 1))
            }
        }
        
        return metrics
    
    def _to_serializable(self, obj: Any) -> Any:
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def save_model(self, output_dir: str):
        """Save the trained model and metrics."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, output_path / f"{self.model_name}_model.joblib")
        
        # Save scaler
        joblib.dump(self.scaler, output_path / f"{self.model_name}_scaler.joblib")
        
        # Save metrics and parameters
        metadata = {
            'metrics': self._to_serializable(self.metrics),
            'best_params': self._to_serializable(self.best_params),
            'feature_importance': self._to_serializable(self.feature_importance)
        }
        
        with open(output_path / f"{self.model_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
    
    def load_model(self, model_dir: str):
        """Load a trained model and metadata."""
        model_path = Path(model_dir)
        
        # Load model and scaler
        self.model = joblib.load(model_path / f"{self.model_name}_model.joblib")
        self.scaler = joblib.load(model_path / f"{self.model_name}_scaler.joblib")
        
        # Load metadata
        with open(model_path / f"{self.model_name}_metadata.json", 'r') as f:
            metadata = json.load(f)
            self.metrics = metadata['metrics']
            self.best_params = metadata['best_params']
            self.feature_importance = metadata['feature_importance'] 