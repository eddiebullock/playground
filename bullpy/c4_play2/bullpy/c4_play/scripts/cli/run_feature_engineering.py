#!/usr/bin/env python3
"""
Run feature engineering experiment on a 10,000-sample subset.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
from model_training import ClinicalModelTrainer

def main():
    trainer = ClinicalModelTrainer()
    # Load data subset
    X, y = trainer.load_data('data/processed/features_full.csv', subset=10000)
    # Run feature engineering experiment
    results = trainer.run_feature_engineering_experiment(X, y, top_n=20, add_interactions=True, analyze_aq_eq_sq_spq=True)
    print("\n=== Feature Engineering Experiment Results ===")
    print("Test set metrics:")
    for k, v in results['test_metrics'].items():
        print(f"  {k}: {v:.4f}")
    print(f"Best threshold: {results['best_threshold']:.3f}")
    print("\nAQ/EQ/SQ/SPQ feature coefficients:")
    for k, v in results['aq_eq_sq_spq_coefs'].items():
        print(f"  {k}: {v:.4f}")
    print("\nFeatures used:")
    for f in results['features']:
        print(f"  {f}")

if __name__ == "__main__":
    main() 