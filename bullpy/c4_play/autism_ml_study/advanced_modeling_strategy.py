#!/usr/bin/env python3
"""
Advanced modeling strategy for autism diagnosis prediction.
Addresses class imbalance and implements multiple approaches to improve F1 score.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

SAMPLE_SIZE = 10000  # Number of rows to use for prototyping (set to None for full data)

class AdvancedAutismPredictor:
    def __init__(self, data_path: str):
        """Initialize the predictor with data."""
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.results = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess the data."""
        print("Loading and preprocessing data...")
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        
        # Subsample for prototyping
        if SAMPLE_SIZE is not None and len(self.df) > SAMPLE_SIZE:
            self.df = self.df.sample(n=SAMPLE_SIZE, random_state=42)
            print(f"Subsampled to {SAMPLE_SIZE} rows for prototyping.")
        
        # Create target variable (same as in comprehensive_analysis.py)
        diagnosis_cols = [col for col in self.df.columns if col.startswith('diagnosis_')]
        autism_diag_cols = [col for col in self.df.columns if col.startswith('autism_diagnosis_')]
        
        self.df['autism_any'] = 0
        if autism_diag_cols:
            for col in autism_diag_cols:
                mask = self.df[col].isin([1.0, 2.0, 3.0])
                self.df.loc[mask, 'autism_any'] = 1
        
        # Create questionnaire scores
        for questionnaire in ['spq', 'eq', 'sqr', 'aq']:
            cols = [col for col in self.df.columns if col.startswith(f'{questionnaire}_')]
            if cols:
                self.df[f'{questionnaire}_total'] = self.df[cols].sum(axis=1)
                self.df[f'{questionnaire}_mean'] = self.df[cols].mean(axis=1)
        
        # Create interaction features
        questionnaire_totals = ['spq_total', 'eq_total', 'sqr_total', 'aq_total']
        available_totals = [col for col in questionnaire_totals if col in self.df.columns]
        
        if len(available_totals) >= 2:
            # Create ratio features
            self.df['aq_eq_ratio'] = self.df['aq_total'] / (self.df['eq_total'] + 1e-8)
            self.df['spq_sqr_ratio'] = self.df['spq_total'] / (self.df['sqr_total'] + 1e-8)
            
            # Create interaction features
            for i, col1 in enumerate(available_totals):
                for col2 in available_totals[i+1:]:
                    self.df[f'{col1}_{col2}_interaction'] = self.df[col1] * self.df[col2]
        
        # Select features for modeling
        feature_cols = []
        
        # Questionnaire totals and means
        for questionnaire in ['spq', 'eq', 'sqr', 'aq']:
            if f'{questionnaire}_total' in self.df.columns:
                feature_cols.append(f'{questionnaire}_total')
            if f'{questionnaire}_mean' in self.df.columns:
                feature_cols.append(f'{questionnaire}_mean')
        
        # Interaction features
        interaction_cols = [col for col in self.df.columns if 'interaction' in col or 'ratio' in col]
        feature_cols.extend(interaction_cols)
        
        # Demographics (if available)
        demo_cols = ['age', 'sex', 'education']
        for col in demo_cols:
            if col in self.df.columns:
                feature_cols.append(col)
        
        # Remove any missing values
        self.df = self.df.dropna(subset=feature_cols + ['autism_any'])
        
        # Prepare X and y
        self.X = self.df[feature_cols]
        self.y = self.df['autism_any']
        self.feature_names = feature_cols
        
        print(f"Final dataset shape: {self.X.shape}")
        print(f"Class distribution: {self.y.value_counts().to_dict()}")
        print(f"Features: {self.feature_names}")
        
    def evaluate_baseline_models(self):
        """Evaluate baseline models without any imbalance handling."""
        print("\n" + "="*60)
        print("BASELINE MODEL EVALUATION")
        print("="*60)
        
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1)
        }
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in models.items():
            print(f"\nEvaluating {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X, self.y, cv=skf, scoring='f1')
            print(f"  CV F1 Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # Train on full dataset for detailed metrics
            model.fit(self.X, self.y)
            y_pred = model.predict(self.X)
            y_pred_proba = model.predict_proba(self.X)[:, 1]
            
            # Calculate metrics
            f1 = f1_score(self.y, y_pred)
            auc = roc_auc_score(self.y, y_pred_proba)
            
            print(f"  F1 Score: {f1:.4f}")
            print(f"  AUROC: {auc:.4f}")
            
            # Store results
            self.results[f'{name}_baseline'] = {
                'cv_f1_mean': cv_scores.mean(),
                'cv_f1_std': cv_scores.std(),
                'f1_score': f1,
                'auc': auc,
                'model': model
            }
    
    def evaluate_imbalance_handling_strategies(self):
        """Evaluate different class imbalance handling strategies."""
        print("\n" + "="*60)
        print("CLASS IMBALANCE HANDLING STRATEGIES")
        print("="*60)
        
        strategies = {
            'SMOTE': SMOTE(random_state=42),
            'ADASYN': ADASYN(random_state=42),
            'Random Under-sampling': RandomUnderSampler(random_state=42),
            'Class Weights': 'class_weight'
        }
        
        base_models = {
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for strategy_name, strategy in strategies.items():
            print(f"\n--- {strategy_name} ---")
            
            for model_name, base_model in base_models.items():
                print(f"  {model_name}:")
                
                if strategy == 'class_weight':
                    # Use class weights
                    params = base_model.get_params()
                    if hasattr(base_model, 'class_weight'):
                        # Only set if not already set
                        if params.get('class_weight', None) != 'balanced':
                            params['class_weight'] = 'balanced'
                        model = base_model.__class__(**params)
                    elif isinstance(base_model, xgb.XGBClassifier):
                        # For XGBoost, use scale_pos_weight if not already set
                        if params.get('scale_pos_weight', None) is None:
                            scale_pos_weight = len(self.y[self.y == 0]) / len(self.y[self.y == 1])
                            params['scale_pos_weight'] = scale_pos_weight
                        model = xgb.XGBClassifier(**params)
                    else:
                        continue
                else:
                    # Use resampling
                    X_resampled, y_resampled = strategy.fit_resample(self.X, self.y)
                    model = base_model
                    X_train, y_train = X_resampled, y_resampled
                
                # Cross-validation
                if strategy == 'class_weight':
                    cv_scores = cross_val_score(model, self.X, self.y, cv=skf, scoring='f1')
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1')
                
                print(f"    CV F1 Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
                # Train on full dataset
                if strategy == 'class_weight':
                    model.fit(self.X, self.y)
                    y_pred = model.predict(self.X)
                    y_pred_proba = model.predict_proba(self.X)[:, 1]
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(self.X)
                    y_pred_proba = model.predict_proba(self.X)[:, 1]
                
                f1 = f1_score(self.y, y_pred)
                auc = roc_auc_score(self.y, y_pred_proba)
                from sklearn.metrics import precision_score, recall_score
                precision = precision_score(self.y, y_pred)
                recall = recall_score(self.y, y_pred)
                print(f"    F1 Score: {f1:.4f}")
                print(f"    Precision: {precision:.4f}")
                print(f"    Recall: {recall:.4f}")
                print(f"    AUROC: {auc:.4f}")
                
                # Store results
                self.results[f'{model_name}_{strategy_name}'] = {
                    'cv_f1_mean': cv_scores.mean(),
                    'cv_f1_std': cv_scores.std(),
                    'f1_score': f1,
                    'auc': auc,
                    'precision': precision,
                    'recall': recall,
                    'strategy': strategy_name,
                    'model': model
                }
    
    def create_ensemble_model(self):
        """Create an ensemble model combining multiple strategies."""
        print("\n" + "="*60)
        print("ENSEMBLE MODEL CREATION")
        print("="*60)
        
        # Get the best performing models from previous evaluations
        best_models = []
        
        # Find models with highest F1 scores
        f1_scores = {name: result['f1_score'] for name, result in self.results.items()}
        best_model_names = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        print("Top 3 models by F1 score:")
        for name, f1 in best_model_names:
            print(f"  {name}: {f1:.4f}")
            best_models.append(self.results[name]['model'])
        
        # Create ensemble predictions
        ensemble_preds = []
        ensemble_probs = []
        
        for model in best_models:
            if hasattr(model, 'predict_proba'):
                ensemble_probs.append(model.predict_proba(self.X)[:, 1])
            else:
                ensemble_probs.append(model.predict(self.X))
        
        # Average predictions
        ensemble_prob = np.mean(ensemble_probs, axis=0)
        ensemble_pred = (ensemble_prob > 0.5).astype(int)
        
        # Calculate ensemble metrics
        ensemble_f1 = f1_score(self.y, ensemble_pred)
        ensemble_auc = roc_auc_score(self.y, ensemble_prob)
        
        print(f"\nEnsemble Model Performance:")
        print(f"  F1 Score: {ensemble_f1:.4f}")
        print(f"  AUROC: {ensemble_auc:.4f}")
        
        self.results['Ensemble'] = {
            'f1_score': ensemble_f1,
            'auc': ensemble_auc,
            'models': best_models
        }
    
    def analyze_feature_importance(self):
        """Analyze feature importance for the best model."""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Find the best model
        best_model_name = max(self.results.items(), key=lambda x: x[1]['f1_score'])[0]
        best_model = self.results[best_model_name]['model']
        
        print(f"Analyzing feature importance for: {best_model_name}")
        
        # Get feature importance
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
        elif hasattr(best_model, 'coef_'):
            importances = np.abs(best_model.coef_[0])
        else:
            print("  Model doesn't support feature importance analysis")
            return
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 most important features:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance - {best_model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive report of all results."""
        print("\n" + "="*60)
        print("COMPREHENSIVE RESULTS REPORT")
        print("="*60)
        
        # Create results summary
        results_summary = []
        for name, result in self.results.items():
            def fmt(val):
                if isinstance(val, float):
                    return f"{val:.4f}"
                return str(val)
            results_summary.append({
                'Model': name,
                'F1 Score': fmt(result['f1_score']),
                'AUROC': fmt(result['auc']),
                'CV F1 Mean': fmt(result.get('cv_f1_mean', 'N/A')),
                'CV F1 Std': fmt(result.get('cv_f1_std', 'N/A')),
                'Precision': fmt(result.get('precision', 'N/A')),
                'Recall': fmt(result.get('recall', 'N/A'))
            })
        
        results_df = pd.DataFrame(results_summary)
        print("\nModel Performance Summary:")
        print(results_df.to_string(index=False))
        
        # Find best model
        best_model_name = max(self.results.items(), key=lambda x: x[1]['f1_score'])[0]
        best_f1 = self.results[best_model_name]['f1_score']
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Best F1 Score: {best_f1:.4f}")
        
        # Save results
        results_df.to_csv('model_comparison_results.csv', index=False)
        print(f"\nResults saved to: model_comparison_results.csv")
        
        return results_df

    def evaluate_on_holdout(self):
        """Evaluate the best model on a true holdout set (20% of original data)."""
        print("\n" + "="*60)
        print("EVALUATION ON TRUE HOLDOUT SET")
        print("="*60)
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
        # Use the original data (before resampling)
        X = self.X
        y = self.y
        X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        # Find the best model
        best_model_name = max(self.results.items(), key=lambda x: x[1]['f1_score'])[0]
        best_model = self.results[best_model_name]['model']
        # Retrain on train set only
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_holdout)
        y_pred_proba = best_model.predict_proba(X_holdout)[:, 1]
        f1 = f1_score(y_holdout, y_pred)
        precision = precision_score(y_holdout, y_pred)
        recall = recall_score(y_holdout, y_pred)
        auc = roc_auc_score(y_holdout, y_pred_proba)
        print(f"Holdout F1 Score: {f1:.4f}")
        print(f"Holdout Precision: {precision:.4f}")
        print(f"Holdout Recall: {recall:.4f}")
        print(f"Holdout AUROC: {auc:.4f}")
        return {'f1': f1, 'precision': precision, 'recall': recall, 'auc': auc}
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting Advanced Autism Prediction Analysis")
        print("="*60)
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Evaluate baseline models
        self.evaluate_baseline_models()
        
        # Evaluate imbalance handling strategies
        self.evaluate_imbalance_handling_strategies()
        
        # Create ensemble model
        self.create_ensemble_model()
        
        # Analyze feature importance
        self.analyze_feature_importance()
        
        # Generate comprehensive report
        results_df = self.generate_comprehensive_report()
        
        # Evaluate on true holdout set
        self.evaluate_on_holdout()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("Next steps:")
        print("1. Review model_comparison_results.csv")
        print("2. Check feature_importance.png")
        print("3. Implement the best performing model")
        print("4. Consider ensemble approach for production")
        
        return results_df

def main():
    """Main function to run the analysis."""
    data_path = "processed_data_with_target.csv"
    
    predictor = AdvancedAutismPredictor(data_path)
    results = predictor.run_complete_analysis()
    
    return results

if __name__ == "__main__":
    main() 