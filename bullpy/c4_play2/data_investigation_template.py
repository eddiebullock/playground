# Copy and paste this code to investigate the sex distribution issue

import pandas as pd
import numpy as np

print("="*60)
print("INVESTIGATING SEX DISTRIBUTION ISSUE")
print("="*60)

# 1. Check original raw dataset
print("1. ORIGINAL RAW DATASET:")
print("Loading original dataset...")
try:
    df_raw = pd.read_csv('/Users/eb2007/documents/phd/data/data_c4_raw.csv')
    print(f"Original dataset shape: {df_raw.shape}")
    
    # Check what sex columns exist in raw data
    sex_cols_raw = [col for col in df_raw.columns if 'sex' in col.lower()]
    print(f"Sex-related columns in raw data: {sex_cols_raw}")
    
    # Show unique values in sex columns
    for col in sex_cols_raw:
        if col in df_raw.columns:
            print(f"\n{col} unique values:")
            print(df_raw[col].value_counts())
            print(f"Missing values: {df_raw[col].isnull().sum()}")
    
except Exception as e:
    print(f"Error loading raw dataset: {e}")

print("\n" + "="*60)

# 2. Check processed dataset
print("2. PROCESSED DATASET:")
print("Loading processed dataset...")
try:
    df_processed = pd.read_csv('data/processed/data_c4_matched_balanced.csv')
    print(f"Processed dataset shape: {df_processed.shape}")
    
    # Check what sex columns exist in processed data
    sex_cols_processed = [col for col in df_processed.columns if 'sex' in col.lower()]
    print(f"Sex-related columns in processed data: {sex_cols_processed}")
    
    # Show unique values in sex columns
    for col in sex_cols_processed:
        if col in df_processed.columns:
            print(f"\n{col} unique values:")
            print(df_processed[col].value_counts())
            print(f"Missing values: {df_processed[col].isnull().sum()}")
    
except Exception as e:
    print(f"Error loading processed dataset: {e}")

print("\n" + "="*60)

# 3. Compare sex distributions
print("3. COMPARISON:")
if 'df_raw' in locals() and 'df_processed' in locals():
    print("Sex distribution comparison:")
    
    # Find common sex columns or try to identify them
    raw_sex_col = None
    processed_sex_cols = []
    
    # Look for sex columns in raw data
    for col in df_raw.columns:
        if 'sex' in col.lower():
            raw_sex_col = col
            break
    
    # Look for sex columns in processed data
    for col in df_processed.columns:
        if 'sex' in col.lower():
            processed_sex_cols.append(col)
    
    if raw_sex_col:
        print(f"\nRaw dataset - {raw_sex_col}:")
        print(df_raw[raw_sex_col].value_counts())
        print(f"Total: {len(df_raw)}")
    
    if processed_sex_cols:
        print(f"\nProcessed dataset - sex columns:")
        for col in processed_sex_cols:
            print(f"{col}: {df_processed[col].sum()} ({df_processed[col].sum()/len(df_processed)*100:.1f}%)")
        print(f"Total: {len(df_processed)}")

print("\n" + "="*60)
print("POTENTIAL ISSUES TO CHECK:")
print("1. Was the data filtering/balancing process biased?")
print("2. Was there an error in the one-hot encoding?")
print("3. Was the original data already imbalanced?")
print("4. Was there a sampling issue during processing?")
print("5. Are the sex categories being interpreted correctly?")
print("="*60)
