#!/usr/bin/env python3
"""
Quick validation script to check if the data file exists and can be loaded.
"""

import sys
from pathlib import Path
import pandas as pd

def validate_data_file():
    """Validate that the data file exists and can be loaded."""
    
    # Data file path from config
    data_path = "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/data_c4_raw.csv"
    
    print("Validating data file...")
    print(f"Path: {data_path}")
    
    # Check if file exists
    if not Path(data_path).exists():
        print("ERROR: Data file not found!")
        print("Please check the file path in experiments/configs/data_config.yaml")
        return False
    
    print("File exists")
    
    # Try to load the data
    try:
        df = pd.read_csv(data_path)
        print("Data loaded successfully")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Check for expected columns
        expected_cols = ['spq', 'eq', 'sqr', 'aq', 'diagnosis', 'age', 'sex', 'autism_diagnosis']
        found_cols = []
        
        for expected in expected_cols:
            matching_cols = [col for col in df.columns if expected in col.lower()]
            if matching_cols:
                found_cols.extend(matching_cols)
                print(f"   Found {len(matching_cols)} columns containing '{expected}'")
        
        if not found_cols:
            print("WARNING: No expected columns found. Check column names.")
        else:
            print(f"   Found {len(found_cols)} relevant columns")
        
        # Check for missing values
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        print(f"   Missing values: {missing_pct:.2f}%")
        
        # Check for autism diagnosis columns (target will be created from these)
        autism_cols = [col for col in df.columns if 'autism_diagnosis' in col]
        if autism_cols:
            print(f"   Found {len(autism_cols)} autism diagnosis columns")
        else:
            print("WARNING: No autism diagnosis columns found")
        
        return True
        
    except Exception as e:
        print(f"ERROR loading data: {str(e)}")
        return False

def main():
    """Main validation function."""
    print("=" * 60)
    print("DATA FILE VALIDATION")
    print("=" * 60)
    
    success = validate_data_file()
    
    print("\n" + "=" * 60)
    if success:
        print("VALIDATION PASSED - Ready to run preprocessing pipeline!")
        print("\nNext steps:")
        print("1. Run: python tests/test_pipeline.py")
        print("2. Run: python scripts/cli/run_preprocessing.py --config experiments/configs/data_config.yaml --verbose")
        print("3. Run: python src/exploratory_analysis.py")
    else:
        print("VALIDATION FAILED - Please fix issues before proceeding")
    print("=" * 60)

if __name__ == "__main__":
    main() 