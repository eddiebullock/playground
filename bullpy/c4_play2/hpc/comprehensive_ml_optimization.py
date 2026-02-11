#!/usr/bin/env python3
"""
Comprehensive ML Optimization Script for Autism Classification
Includes: Hyperparameter tuning, feature selection, threshold optimization, and ensemble methods
"""

import os
import sys
import yaml
import json
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# ML imports
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, classification_report, accuracy_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Try to import CatBoost (optional)
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available. Install with: pip install catboost")

# Utility imports
import joblib
import warnings
warnings.filterwarnings('ignore')

def setup_logging(log_dir):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"comprehensive_ml_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

class ComprehensiveMLOptimizer:
    """Comprehensive ML optimization with all advanced techniques"""
    
    def __init__(self, config_path="hpc_config_comprehensive.yaml"):
        """Initialize the optimizer"""
        # If config not found, try in current directory
        if not os.path.exists(config_path):
            # Try without path
            if os.path.exists(os.path.basename(config_path)):
                config_path = os.path.basename(config_path)
        self.config = self.load_config(config_path)
        self.logger = setup_logging(self.config['output']['logs_dir'])
        self.setup_directories()
        self.load_data()
        self.all_results = {}
        # Used for checkpointing partial results so timeouts still produce artifacts
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _checkpoint(self, model_key: str | None = None):
        """
        Save partial results + any completed model(s) to disk.
        This is intentionally lightweight so it can run after each model.
        """
        results_dir = self.config['output']['results_dir']
        models_dir = self.config['output']['models_dir']
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

        # Save model (if requested and present)
        if self.config['output'].get('save_models', True) and model_key and model_key in self.all_results:
            model_obj = self.all_results[model_key].get('model')
            if model_obj is not None:
                model_path = os.path.join(models_dir, f"{model_key}_optimized_{self.run_id}.joblib")
                joblib.dump(model_obj, model_path)
                self.logger.info(f"[checkpoint] Model saved: {model_path}")

        # Save scaler + feature selector once (overwrites with same run_id, cheap + useful)
        scaler_path = os.path.join(models_dir, f"scaler_{self.run_id}.joblib")
        joblib.dump(self.scaler, scaler_path)
        if hasattr(self, 'feature_selector') and self.feature_selector:
            selector_path = os.path.join(models_dir, f"feature_selector_{self.run_id}.joblib")
            joblib.dump(self.feature_selector, selector_path)

        # Save summary JSON/CSV without embedding large objects
        summary = {}
        for key, result in self.all_results.items():
            if key == 'ensemble':
                # ensemble doesn't have best_params/cv_score in our structure
                summary[key] = {
                    'test_f1': float(result.get('test_f1', 0.0)),
                    'test_auc': float(result.get('test_auc', 0.0)),
                    'optimized_f1': float(result.get('optimized_f1', 0.0)),
                    'best_threshold': float(result.get('best_threshold', 0.5)),
                    'base_models': result.get('base_models', []),
                }
                continue

            summary[key] = {
                'best_params': result.get('best_params', {}),
                'cv_score': float(result.get('cv_score', 0.0)),
                'test_f1': float(result.get('test_f1', 0.0)),
                'test_auc': float(result.get('test_auc', 0.0)),
                'test_accuracy': float(result.get('test_accuracy', 0.0)),
                'test_precision': float(result.get('test_precision', 0.0)),
                'test_recall': float(result.get('test_recall', 0.0)),
                'optimized_f1': float(result.get('optimized_f1', 0.0)),
                'optimized_accuracy': float(result.get('optimized_accuracy', 0.0)),
                'optimized_precision': float(result.get('optimized_precision', 0.0)),
                'optimized_recall': float(result.get('optimized_recall', 0.0)),
                'best_threshold': float(result.get('best_threshold', 0.5)),
            }

        json_path = os.path.join(results_dir, f"comprehensive_partial_{self.run_id}.json")
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # CSV snapshot (sorted by optimized F1 where present)
        rows = []
        for k, v in summary.items():
            if k == 'ensemble':
                continue
            rows.append({
                'Model': k,
                'CV_F1': v.get('cv_score', 0.0),
                'Test_F1': v.get('test_f1', 0.0),
                'Optimized_F1': v.get('optimized_f1', 0.0),
                'Test_AUC': v.get('test_auc', 0.0),
                'Best_Threshold': v.get('best_threshold', 0.5),
            })
        if rows:
            df = pd.DataFrame(rows).sort_values('Optimized_F1', ascending=False)
            csv_path = os.path.join(results_dir, f"comprehensive_partial_{self.run_id}.csv")
            df.to_csv(csv_path, index=False)

        self.logger.info(f"[checkpoint] Partial results saved: {json_path}")
        
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def setup_directories(self):
        """Create necessary directories"""
        for dir_name in ['results', 'models', 'logs', 'plots']:
            os.makedirs(self.config['output'][f'{dir_name}_dir'], exist_ok=True)
    
    def load_data(self):
        """Load and preprocess the dataset"""
        self.logger.info("Loading dataset...")
        
        data_file = self.config['data']['input_file']
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        self.df = pd.read_csv(data_file)
        self.logger.info(f"Dataset loaded: {self.df.shape}")
        
        # Remove AQ features to prevent data leakage
        aq_features = [col for col in self.df.columns if 'aq' in col.lower()]
        exclude_features = [self.config['data']['target_column']] + aq_features
        
        # Prepare features and target
        self.X = self.df.drop(columns=exclude_features, errors='ignore')
        self.y = self.df[self.config['data']['target_column']]
        
        self.logger.info(f"Features after AQ exclusion: {self.X.shape[1]}")
        if aq_features:
            self.logger.info(f"Excluded AQ features: {aq_features}")
        
        # Handle missing values
        imputer = SimpleImputer(strategy=self.config['feature_engineering']['imputation_strategy'])
        self.X_imputed = pd.DataFrame(
            imputer.fit_transform(self.X), 
            columns=self.X.columns
        )
        
        # Convert object columns to numeric
        for col in self.X_imputed.columns:
            if self.X_imputed[col].dtype == 'object':
                self.X_imputed[col] = pd.Categorical(self.X_imputed[col]).codes
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_imputed, self.y,
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=self.y if self.config['data']['stratify'] else None
        )
        
        # Apply scaling
        self.scaler = StandardScaler()
        self.X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns
        )
        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns
        )
        
        self.logger.info(f"Training set: {self.X_train_scaled.shape}")
        self.logger.info(f"Test set: {self.X_test_scaled.shape}")
        self.logger.info(f"Class distribution - Train: {self.y_train.value_counts().to_dict()}")
        self.logger.info(f"Class distribution - Test: {self.y_test.value_counts().to_dict()}")
    
    def optimize_feature_selection(self):
        """Optimize feature selection"""
        self.logger.info("Optimizing feature selection...")
        
        k_values = self.config['feature_engineering']['feature_selection']['selectkbest_k']
        best_k = None
        best_score = 0
        best_selector = None
        
        for k in k_values:
            if k <= self.X_train_scaled.shape[1]:
                self.logger.info(f"Testing SelectKBest with k={k}")
                
                selector = SelectKBest(score_func=f_classif, k=k)
                X_train_selected = selector.fit_transform(self.X_train_scaled, self.y_train)
                X_test_selected = selector.transform(self.X_test_scaled)
                
                # Quick evaluation with Random Forest
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X_train_selected, self.y_train)
                
                y_probs = rf.predict_proba(X_test_selected)[:, 1]
                auc = roc_auc_score(self.y_test, y_probs)
                
                self.logger.info(f"k={k}: AUC={auc:.4f}")
                
                if auc > best_score:
                    best_score = auc
                    best_k = k
                    best_selector = selector
        
        if best_selector:
            self.X_train_selected = pd.DataFrame(
                best_selector.transform(self.X_train_scaled),
                columns=self.X_train_scaled.columns[best_selector.get_support()]
            )
            self.X_test_selected = pd.DataFrame(
                best_selector.transform(self.X_test_scaled),
                columns=self.X_test_scaled.columns[best_selector.get_support()]
            )
            self.feature_selector = best_selector
            self.logger.info(f"Best feature selection: k={best_k}, AUC={best_score:.4f}")
        else:
            self.X_train_selected = self.X_train_scaled
            self.X_test_selected = self.X_test_scaled
            self.feature_selector = None
            self.logger.info("Using all features")
        
        return best_k, best_score
    
    def optimize_threshold(self, y_true, y_proba):
        """Find optimal threshold for F1 score"""
        prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_f1 = f1_scores[best_idx]
        return best_threshold, best_f1
    
    def optimize_model(self, model_name, model_class, param_grid, use_scaled=True):
        """Optimize a single model"""
        self.logger.info(f"Optimizing {model_name}...")
        
        X_train = self.X_train_selected if hasattr(self, 'X_train_selected') else (self.X_train_scaled if use_scaled else self.X_train)
        X_test = self.X_test_selected if hasattr(self, 'X_test_selected') else (self.X_test_scaled if use_scaled else self.X_test)
        
        # Use CV folds from config (default 3 for speed)
        cv_folds = self.config.get('cross_validation', {}).get('inner_cv', 3)
        
        # Log parameter grid size for debugging
        import itertools
        param_combinations = len(list(itertools.product(*param_grid.values())))
        self.logger.info(f"{model_name} parameter grid: {param_combinations} combinations, {cv_folds}-fold CV = {param_combinations * cv_folds} total fits")
        
        # Create model instance
        model = model_class(random_state=42)
        
        # Grid search - use available CPUs efficiently
        import os
        n_jobs = min(8, os.cpu_count() or 1)  # Limit to 8 to match SLURM request
        
        # Special handling for XGBoost - set n_jobs in model, not GridSearchCV
        if model_name == 'XGBoost':
            model.set_params(n_jobs=n_jobs)
            grid_n_jobs = 1  # XGBoost handles parallelism internally
        else:
            grid_n_jobs = n_jobs
        
        self.logger.info(f"Using {grid_n_jobs} jobs for GridSearchCV, {cv_folds} CV folds")
        
        grid_search = GridSearchCV(
            model, param_grid, 
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring='f1',
            n_jobs=grid_n_jobs,
            verbose=2  # Increased verbosity for debugging
        )
        
        self.logger.info(f"Starting grid search fit...")
        import sys
        sys.stdout.flush()  # Force output flush
        
        try:
            grid_search.fit(X_train, self.y_train)
            self.logger.info(f"Grid search completed for {model_name}")
        except Exception as e:
            self.logger.error(f"Error during grid search for {model_name}: {str(e)}")
            raise
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Predictions
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Metrics with default threshold
        f1 = f1_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_proba)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        
        # Threshold optimization
        best_threshold, optimized_f1 = self.optimize_threshold(self.y_test, y_proba)
        y_pred_optimized = (y_proba >= best_threshold).astype(int)
        
        optimized_accuracy = accuracy_score(self.y_test, y_pred_optimized)
        optimized_precision = precision_score(self.y_test, y_pred_optimized, zero_division=0)
        optimized_recall = recall_score(self.y_test, y_pred_optimized, zero_division=0)
        
        results = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'test_f1': f1,
            'test_auc': auc,
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'optimized_f1': optimized_f1,
            'optimized_accuracy': optimized_accuracy,
            'optimized_precision': optimized_precision,
            'optimized_recall': optimized_recall,
            'best_threshold': best_threshold,
            'y_proba': y_proba
        }
        
        self.logger.info(f"{model_name} - F1: {f1:.4f}, Optimized F1: {optimized_f1:.4f}, AUC: {auc:.4f}")
        
        return results
    
    def optimize_all_models(self):
        """Optimize all models"""
        self.logger.info("="*80)
        self.logger.info("STARTING COMPREHENSIVE MODEL OPTIMIZATION")
        self.logger.info("="*80)
        
        # Optimize feature selection first
        self.optimize_feature_selection()
        
        # Random Forest
        rf_results = self.optimize_model(
            'Random Forest',
            RandomForestClassifier,
            self.config['random_forest']
        )
        self.all_results['random_forest'] = rf_results
        self._checkpoint('random_forest')
        
        # XGBoost
        xgb_results = self.optimize_model(
            'XGBoost',
            XGBClassifier,
            self.config['xgboost']
        )
        self.all_results['xgboost'] = xgb_results
        self._checkpoint('xgboost')
        
        # LightGBM
        lgb_results = self.optimize_model(
            'LightGBM',
            LGBMClassifier,
            self.config['lightgbm']
        )
        self.all_results['lightgbm'] = lgb_results
        self._checkpoint('lightgbm')
        
        # Gradient Boosting
        gb_results = self.optimize_model(
            'Gradient Boosting',
            GradientBoostingClassifier,
            self.config['gradient_boosting']
        )
        self.all_results['gradient_boosting'] = gb_results
        self._checkpoint('gradient_boosting')
        
        # Extra Trees
        et_results = self.optimize_model(
            'Extra Trees',
            ExtraTreesClassifier,
            self.config['extra_trees']
        )
        self.all_results['extra_trees'] = et_results
        self._checkpoint('extra_trees')
        
        # CatBoost (if available)
        if CATBOOST_AVAILABLE and 'catboost' in self.config:
            cb_results = self.optimize_model(
                'CatBoost',
                CatBoostClassifier,
                self.config['catboost'],
                use_scaled=False  # CatBoost doesn't need scaling
            )
            self.all_results['catboost'] = cb_results
            self._checkpoint('catboost')
        
        # Logistic Regression
        lr_results = self.optimize_model(
            'Logistic Regression',
            LogisticRegression,
            self.config['logistic_regression']
        )
        self.all_results['logistic_regression'] = lr_results
        self._checkpoint('logistic_regression')
        
        return self.all_results
    
    def create_ensemble(self):
        """Create voting ensemble from best models"""
        self.logger.info("Creating voting ensemble...")
        
        X_train = self.X_train_selected if hasattr(self, 'X_train_selected') else self.X_train_scaled
        X_test = self.X_test_selected if hasattr(self, 'X_test_selected') else self.X_test_scaled
        
        # Get top 5 models by optimized F1
        sorted_models = sorted(self.all_results.items(), key=lambda x: x[1]['optimized_f1'], reverse=True)
        top_models = sorted_models[:5]
        
        estimators = []
        for name, result in top_models:
            estimators.append((name.replace('_', ' ').title(), result['model']))
        
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft'
        )
        
        voting_clf.fit(X_train, self.y_train)
        y_pred = voting_clf.predict(X_test)
        y_proba = voting_clf.predict_proba(X_test)[:, 1]
        
        f1 = f1_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_proba)
        
        # Threshold optimization
        best_threshold, optimized_f1 = self.optimize_threshold(self.y_test, y_proba)
        y_pred_optimized = (y_proba >= best_threshold).astype(int)
        optimized_f1_final = f1_score(self.y_test, y_pred_optimized)
        
        ensemble_results = {
            'model': voting_clf,
            'test_f1': f1,
            'test_auc': auc,
            'optimized_f1': optimized_f1_final,
            'best_threshold': best_threshold,
            'base_models': [name for name, _ in top_models]
        }
        
        self.logger.info(f"Ensemble - F1: {f1:.4f}, Optimized F1: {optimized_f1_final:.4f}, AUC: {auc:.4f}")
        
        return ensemble_results
    
    def save_results(self):
        """Save all results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save models
        if self.config['output']['save_models']:
            for name, result in self.all_results.items():
                model_path = os.path.join(
                    self.config['output']['models_dir'],
                    f"{name}_optimized_{timestamp}.joblib"
                )
                joblib.dump(result['model'], model_path)
                self.logger.info(f"Model saved: {model_path}")
        
        # Save scaler and feature selector
        scaler_path = os.path.join(self.config['output']['models_dir'], f"scaler_{timestamp}.joblib")
        joblib.dump(self.scaler, scaler_path)
        
        if hasattr(self, 'feature_selector') and self.feature_selector:
            selector_path = os.path.join(self.config['output']['models_dir'], f"feature_selector_{timestamp}.joblib")
            joblib.dump(self.feature_selector, selector_path)
        
        # Save results summary
        results_summary = {}
        for name, result in self.all_results.items():
            results_summary[name] = {
                'best_params': result['best_params'],
                'cv_score': float(result['cv_score']),
                'test_f1': float(result['test_f1']),
                'test_auc': float(result['test_auc']),
                'test_accuracy': float(result['test_accuracy']),
                'test_precision': float(result['test_precision']),
                'test_recall': float(result['test_recall']),
                'optimized_f1': float(result['optimized_f1']),
                'optimized_accuracy': float(result['optimized_accuracy']),
                'optimized_precision': float(result['optimized_precision']),
                'optimized_recall': float(result['optimized_recall']),
                'best_threshold': float(result['best_threshold'])
            }
        
        results_path = os.path.join(
            self.config['output']['results_dir'],
            f"comprehensive_results_{timestamp}.json"
        )
        
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        self.logger.info(f"Results saved: {results_path}")
        
        # Create CSV summary
        summary_data = []
        for name, result in results_summary.items():
            summary_data.append({
                'Model': name,
                'CV_F1': result['cv_score'],
                'Test_F1': result['test_f1'],
                'Optimized_F1': result['optimized_f1'],
                'Test_AUC': result['test_auc'],
                'Test_Accuracy': result['test_accuracy'],
                'Optimized_Accuracy': result['optimized_accuracy'],
                'Best_Threshold': result['best_threshold']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Optimized_F1', ascending=False)
        
        csv_path = os.path.join(
            self.config['output']['results_dir'],
            f"comprehensive_results_{timestamp}.csv"
        )
        summary_df.to_csv(csv_path, index=False)
        self.logger.info(f"CSV summary saved: {csv_path}")
        
        # Print summary
        self.logger.info("="*80)
        self.logger.info("OPTIMIZATION SUMMARY")
        self.logger.info("="*80)
        print(summary_df.to_string(index=False))
        self.logger.info("="*80)
        
        return results_path, csv_path
    
    def run_optimization(self):
        """Run complete optimization pipeline"""
        # Optimize all models
        self.optimize_all_models()
        
        # Create ensemble
        ensemble_results = self.create_ensemble()
        self.all_results['ensemble'] = ensemble_results
        
        # Save results
        self.save_results()
        
        return self.all_results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Comprehensive ML Optimization')
    parser.add_argument('--config', default='hpc_config_comprehensive.yaml', help='Configuration file')
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = ComprehensiveMLOptimizer(args.config)
    
    # Run optimization
    results = optimizer.run_optimization()
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ML OPTIMIZATION COMPLETED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()
