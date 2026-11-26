"""
Tree-based models implementation including Decision Tree, Random Forest, and Gradient Boosting.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from .base import BaseModel

class AdvancedDecisionTree(BaseModel):
    """Advanced Decision Tree model with hyperparameter tuning and feature selection."""
    
    def __init__(self, assessment_type: str):
        """
        Initialize the advanced decision tree model.
        
        Args:
            assessment_type: Type of assessment ('AQ', 'SQ', or 'EQ')
        """
        super().__init__('decision_tree', assessment_type)
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model with default parameters."""
        self.model = DecisionTreeClassifier(
            random_state=42,
            class_weight='balanced'
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
        param_grid = {
            'classifier__max_depth': [3, 5, 7, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__class_weight': ['balanced', {0: 1, 1: 5}, {0: 1, 1: 10}]
        }
        
        pipeline = Pipeline([
            ('feature_selection', SelectFromModel(
                DecisionTreeClassifier(random_state=42, class_weight='balanced'),
                threshold='mean'
            )),
            ('classifier', self.model)
        ])
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.feature_importance = self.model.named_steps['classifier'].feature_importances_
        
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        threshold = 0.3
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        self.metrics = self.compute_metrics(y, y_pred, y_pred_proba)
        
        return self.metrics

class AdvancedRandomForest(BaseModel):
    """Advanced Random Forest model with hyperparameter tuning and feature selection."""
    
    def __init__(self, assessment_type: str):
        """
        Initialize the advanced random forest model.
        
        Args:
            assessment_type: Type of assessment ('AQ', 'SQ', or 'EQ')
        """
        super().__init__('random_forest', assessment_type)
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model with default parameters."""
        self.model = RandomForestClassifier(
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
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
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [5, 10, None],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2],
            'classifier__class_weight': ['balanced', {0: 1, 1: 5}, {0: 1, 1: 10}]
        }
        
        pipeline = Pipeline([
            ('feature_selection', SelectFromModel(
                RandomForestClassifier(random_state=42, class_weight='balanced'),
                threshold='mean'
            )),
            ('classifier', self.model)
        ])
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.feature_importance = self.model.named_steps['classifier'].feature_importances_
        
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        threshold = 0.3
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        self.metrics = self.compute_metrics(y, y_pred, y_pred_proba)
        
        return self.metrics

class AdvancedGradientBoosting(BaseModel):
    """Advanced Gradient Boosting model with hyperparameter tuning and feature selection."""
    
    def __init__(self, assessment_type: str):
        """
        Initialize the advanced gradient boosting model.
        
        Args:
            assessment_type: Type of assessment ('AQ', 'SQ', or 'EQ')
        """
        super().__init__('gradient_boosting', assessment_type)
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model with default parameters."""
        self.model = GradientBoostingClassifier(
            random_state=42
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
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__max_depth': [3, 5],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2],
            'classifier__subsample': [0.8, 1.0]
        }
        
        pipeline = Pipeline([
            ('feature_selection', SelectFromModel(
                GradientBoostingClassifier(random_state=42),
                threshold='mean'
            )),
            ('classifier', self.model)
        ])
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.feature_importance = self.model.named_steps['classifier'].feature_importances_
        
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        threshold = 0.3
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        self.metrics = self.compute_metrics(y, y_pred, y_pred_proba)
        
        return self.metrics 