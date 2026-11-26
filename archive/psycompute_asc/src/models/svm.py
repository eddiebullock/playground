"""
Advanced SVM model implementation.
"""

import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from .base import BaseModel

class AdvancedSVM(BaseModel):
    """Advanced SVM model with hyperparameter tuning and feature selection."""
    
    def __init__(self, assessment_type: str):
        """
        Initialize the advanced SVM model.
        
        Args:
            assessment_type: Type of assessment ('AQ', 'SQ', or 'EQ')
        """
        super().__init__('svm', assessment_type)
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model with default parameters."""
        self.model = SVC(
            random_state=42,
            probability=True,
            class_weight='balanced',
            cache_size=1000,  # Increased cache size for faster training
            max_iter=1000,    # Reduced max iterations
            tol=1e-3         # Increased tolerance for faster convergence
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
        # Define hyperparameter grid with focused values
        param_grid = {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['rbf'],  # Only RBF kernel for classification
            'classifier__gamma': ['scale', 'auto'],
            'classifier__class_weight': [
                'balanced',
                {0: 1, 1: 5},  # Moderate weight to minority class
                {0: 1, 1: 10}  # Higher weight to minority class
            ]
        }
        
        # Create pipeline with feature selection using LinearSVC
        pipeline = Pipeline([
            ('feature_selection', SelectFromModel(
                LinearSVC(
                    random_state=42,
                    class_weight='balanced',
                    max_iter=1000,
                    tol=1e-3
                ),
                threshold='mean'  # Less aggressive feature selection
            )),
            ('classifier', self.model)
        ])
        
        # Setup cross-validation with early stopping
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Perform grid search with f1 as the scoring metric
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1,
            error_score='raise'  # Raise error if any fold fails
        )
        
        # Fit the model
        grid_search.fit(X, y)
        
        # Store best model and parameters
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        # Get feature importance from the feature selection step
        self.feature_importance = np.abs(self.model.named_steps['feature_selection'].estimator_.coef_[0])
        
        # Make predictions
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        # Use a lower threshold for positive class prediction
        threshold = 0.3  # Lower threshold to catch more positive cases
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