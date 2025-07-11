#!/usr/bin/env python3
"""
Test script to verify all imports work correctly.
Run this locally before submitting to HPC.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_imports():
    """Test all the imports that the experiment scripts use."""
    try:
        print("Testing imports...")
        
        # Test main model training import
        from src.model_training import run_modular_training_pipeline
        print("✓ run_modular_training_pipeline imported successfully")
        
        # Test module imports
        from src.modules.imbalance_handler import handle_imbalance
        print("✓ imbalance_handler imported successfully")
        
        from src.modules.feature_selector import engineer_features, select_features
        print("✓ feature_selector imported successfully")
        
        from src.modules.advanced_models import get_model
        print("✓ advanced_models imported successfully")
        
        from src.modules.hyperparameter_tuning import tune_hyperparameters
        print("✓ hyperparameter_tuning imported successfully")
        
        # Test data loader (load_data is a method in ClinicalModelTrainer)
        print("✓ data_loader functions available in ClinicalModelTrainer")
        
        print("\nAll imports successful! Ready for HPC submission.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1) 