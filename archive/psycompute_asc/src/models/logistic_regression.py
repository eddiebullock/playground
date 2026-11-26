"""
Advanced logistic regression model implementation.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from .base import BaseModel

class AdvancedLogisticRegression(BaseModel):
    """Advanced logistic regression model with hyperparameter tuning and feature selection."""
    
    def __init__(self, assessment_type: str):
        """
        Initialize the advanced logistic regression model.
        
        Args:
            assessment_type: Type of assessment ('AQ', 'SQ', or 'EQ')
        """
        super().__init__('logistic_regression', assessment_type)
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model with default parameters."""
        self.model = LogisticRegression(
            random_state=42,
            max_iter=2000,
            class_weight='balanced',
            solver='liblinear'
        )
    
    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Train the model with hyperparameter tuning and feature selection.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary containing model metrics
        """
        # Define hyperparameter grid with more focused values
        param_grid = {
            'classifier__C': [0.1, 1, 10],  # Regular C values
            'classifier__penalty': ['l2'],  # Only L2 regularization
            'classifier__solver': ['liblinear'],
            'classifier__class_weight': [
                'balanced',
                {0: 1, 1: 5},  # Moderate weight to minority class
                {0: 1, 1: 10}  # Higher weight to minority class
            ]
        }
        
        # Create pipeline with feature selection
        pipeline = Pipeline([
            ('feature_selection', SelectFromModel(
                LogisticRegression(
                    random_state=42,
                    class_weight='balanced',
                    solver='liblinear',
                    max_iter=2000
                ),
                threshold='mean'  # Less aggressive feature selection
            )),
            ('classifier', self.model)
        ])
        
        # Setup cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Perform grid search with f1 as the scoring metric
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='f1',  # Changed back to f1 for better balance
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        grid_search.fit(X, y)
        
        # Store best model and parameters
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        # Get feature importance
        self.feature_importance = self.model.named_steps['classifier'].coef_[0]
        
        # Make predictions
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        # Use a lower threshold for positive class prediction
        threshold = 0.3  # Lowered from 0.7 to catch more positive cases
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Compute metrics
        self.metrics = self.compute_metrics(y, y_pred, y_pred_proba)
        
        return self.metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        return (y_pred_proba >= 0.3).astype(int)  # Using same threshold as training
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability estimates.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability estimates for each class
        """
        return self.model.predict_proba(X) 