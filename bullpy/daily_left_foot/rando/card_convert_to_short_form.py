#!/usr/bin/env python3
"""
Converting full AQ EQ SQ into short 10 item versions
"""

import pandas as pd
import numpy as np
from pathlib import Path

# file path configs (edit these) 
INPUT_FILE = '/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/CARD_Nov2025(Sheet1).csv' #change this to your path
OUTPUT_FILE = 'card_short_form_scores.csv' #change this to your path  

# Item mappings (0-indexed positions in full-length questionnaires)
AQ_FULL_LENGTH = 50
AQ_SHORT_ITEMS = [4, 19, 26, 27, 30, 31, 35, 36, 40, 44]  # AQ-50 items 5,20,27,28,31,32,36,37,41,45 (1-indexed)

EQ_FULL_LENGTH = 60
EQ_SHORT_ITEMS = [13, 3, 8, 30, 27, 34, 11, 21, 17, 33]

SQR_FULL_LENGTH = 75
SQR_SHORT_ITEMS = [31, 15, 26, 8, 29, 32, 11, 24, 7, 6]

# Reverse scoring positions (1-indexed in the 10-item output)
AQ_REVERSE_POSITIONS = {2, 3, 4, 5, 6, 9}   # Positions 2,3,4,5,6,9 are reverse scored
EQ_REVERSE_POSITIONS = {3, 5, 7, 8, 10}     # Positions 3,5,7,8,10 are reverse scored
SQR_REVERSE_POSITIONS = {2, 5, 8}           # Positions 2,5,8 are reverse scored

# Column names expected in input
TEST_NAME_COL = 'TestName'
ITEMISED_SCORE_COL = 'Itemised Score'


# functions 

def is_adult_questionnaire(test_name):
    """
    Skipping adolescent and child qs
    """
    t = str(test_name).lower()
    if 'adolescent' in t or 'child' in t:
        return False
    if 'aq' in t or 'eq' in t or 'sq' in t:
        return True
    return False


def parse_itemised_score(itemised_str):
    """
    Parse comma-separated itemised score string into list of integers.
    
    Returns list of integers, or empty list if parsing fails.
    """
    if pd.isna(itemised_str) or itemised_str == '':
        return []
    
    try:
        items = [int(x.strip()) for x in str(itemised_str).split(',') if x.strip()]
        return items
    except (ValueError, AttributeError):
        return []


def extract_short_form_items(items, questionnaire_type):
    """
    Extract 10 items from full-length questionnaire.
    """
    if questionnaire_type == 'aq':
        full_length = AQ_FULL_LENGTH
        short_items = AQ_SHORT_ITEMS
    elif questionnaire_type == 'eq':
        full_length = EQ_FULL_LENGTH
        short_items = EQ_SHORT_ITEMS
    elif questionnaire_type == 'sqr':
        full_length = SQR_FULL_LENGTH
        short_items = SQR_SHORT_ITEMS
    else:
        return [np.nan] * 10
    
    item_count = len(items)
    
    # Case 1: Full length - extract using short_items indices
    if item_count >= full_length:
        extracted = [items[i] if i < item_count else np.nan for i in short_items]
        return extracted
    
    # Case 2: EQ-40 - try to extract using EQ-60 indices (some eqs are 40 items long i think)
    elif questionnaire_type == 'eq' and item_count == 40:
        # Try to extract using EQ-60 indices - items beyond index 39 will be NaN
        extracted = [items[i] if i < item_count else np.nan for i in short_items]
        return extracted
    
    # Case 3: Already 10 items - use as-is
    elif item_count == 10:
        return items
    
    # Case 4: Unexpected length - return NaN-filled list
    else:
        return [np.nan] * 10


def score_item(raw_value, reverse=False):
    """
    Convert 0-3 raw response to binary 0/1.
    """
    if pd.isna(raw_value) or raw_value not in [0, 1, 2, 3]:
        return np.nan
    
    if reverse:
        # Reverse: 1 for disagree (0 or 1), 0 for agree (2 or 3)
        return 1 if raw_value in [0, 1] else 0
    else:
        # Forward: 1 for agree (2 or 3), 0 for disagree (0 or 1)
        return 1 if raw_value in [2, 3] else 0


def score_questionnaire(items, questionnaire_type):
    """
    Score a 10-item questionnaire using binary scoring rules.
    """
    if questionnaire_type == 'aq':
        reverse_positions = AQ_REVERSE_POSITIONS
    elif questionnaire_type == 'eq':
        reverse_positions = EQ_REVERSE_POSITIONS
    elif questionnaire_type == 'sqr':
        reverse_positions = SQR_REVERSE_POSITIONS
    else:
        return ([np.nan] * 10, np.nan)
    
    scored_items = []
    for i, raw_val in enumerate(items):
        position = i + 1  # 1-indexed position
        reverse = position in reverse_positions
        scored_items.append(score_item(raw_val, reverse=reverse))
    
    # Calculate total (sum, ignoring NaN)
    total = sum([x for x in scored_items if not pd.isna(x)])
    if all(pd.isna(x) for x in scored_items):
        total = np.nan
    
    return scored_items, total


def add_short_form_columns(df):
    """
    Add short form score columns to the dataframe, keeping all original rows.
    """
    # Start with a copy of the original dataframe
    output_df = df.copy()
    
    # Initialize short form score columns with NaN
    for q_type in ['aq', 'eq', 'sqr']:
        for i in range(1, 11):
            output_df[f'{q_type}_short_form_{i}'] = np.nan
        output_df[f'{q_type}_short_form_total'] = np.nan
    
    unexpected_length_count = 0
    unexpected_lengths_summary = {}  # {q_type: {length: count}}
    
    # Process each row
    for idx, row in output_df.iterrows():
        test_name = str(row[TEST_NAME_COL]).lower()
        
        # Skip adolescent/child versions (but keep the row with NaN scores)
        if 'adolescent' in test_name or 'child' in test_name:
            continue
        
        # Determine questionnaire type
        q_type = None
        if 'aq' in test_name:
            q_type = 'aq'
        elif 'eq' in test_name:
            q_type = 'eq'
        elif 'sq' in test_name:
            q_type = 'sqr'
        else:
            continue
        
        # Parse itemised score
        items = parse_itemised_score(row[ITEMISED_SCORE_COL])
        
        # Check for unexpected length
        expected_lengths = [10]
        if q_type == 'aq':
            expected_lengths.append(AQ_FULL_LENGTH)
        elif q_type == 'eq':
            expected_lengths.append(EQ_FULL_LENGTH)
            expected_lengths.append(40)
        elif q_type == 'sqr':
            expected_lengths.append(SQR_FULL_LENGTH)
        
        if len(items) not in expected_lengths:
            unexpected_length_count += 1
            if q_type not in unexpected_lengths_summary:
                unexpected_lengths_summary[q_type] = {}
            if len(items) not in unexpected_lengths_summary[q_type]:
                unexpected_lengths_summary[q_type][len(items)] = 0
            unexpected_lengths_summary[q_type][len(items)] += 1
        
        # Extract short form items
        short_items = extract_short_form_items(items, q_type)
        
        # Score questionnaire
        scored_items, total = score_questionnaire(short_items, q_type)
        
        # Add scores to this row
        for i, score in enumerate(scored_items):
            output_df.loc[idx, f'{q_type}_short_form_{i+1}'] = score
        output_df.loc[idx, f'{q_type}_short_form_total'] = total
    
    return output_df, unexpected_length_count, unexpected_lengths_summary


def print_summary(output_df, unexpected_length_count, unexpected_lengths_summary):
    """
    Print simplified processing summary.
    """

    print("SCORE DISTRIBUTIONS")
    print("-" * 60)
    
    for q_type, q_name in [('aq', 'AQ'), ('eq', 'EQ'), ('sqr', 'SQ-R')]:
        scores = output_df[f'{q_type}_short_form_total'].dropna()
        if len(scores) > 0:
            print(f"{q_name}:  mean={scores.mean():.2f}, min={scores.min():.0f}, max={scores.max():.0f}")
        else:
            print(f"{q_name}:  No valid data")
    


# MAIN 

def main():
    """Main processing function."""
    
    # Check input file exists
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        print(f"ERROR: Input file not found at {INPUT_FILE}")
        print("Please update INPUT_FILE at the top of the script.")
        return
    
    # Load data
    print(f"Loading data from: {INPUT_FILE}")
    try:
        # Try UTF-8 first, then fall back to latin-1 if needed
        try:
            df = pd.read_csv(INPUT_FILE, encoding='utf-8')
        except UnicodeDecodeError:
            print("  UTF-8 encoding failed, trying latin-1...")
            df = pd.read_csv(INPUT_FILE, encoding='latin-1')
    except Exception as e:
        print(f"ERROR: Failed to load CSV file: {e}")
        return
    
    print(f"Loaded {len(df)} rows")
    
    # Check required columns
    if TEST_NAME_COL not in df.columns:
        print(f"ERROR: Column '{TEST_NAME_COL}' not found.")
        print(f"Available columns: {list(df.columns)}")
        print("Please update the column name constant at the top of the script.")
        return
    
    if ITEMISED_SCORE_COL not in df.columns:
        print(f"ERROR: Column '{ITEMISED_SCORE_COL}' not found.")
        print(f"Available columns: {list(df.columns)}")
        print("Please update the column name constant at the top of the script.")
        return
    
    # Check Itemised Score column has data
    if df[ITEMISED_SCORE_COL].isna().all():
        print(f"ERROR: '{ITEMISED_SCORE_COL}' column contains no valid data.")
        return
    
    rows_before = len(df)
    print(f"\nTotal rows in input: {rows_before}")
    
    # Step 1-4: Add short form score columns to all rows
    print("PROCESSING: creating short form scores")
    
    # Add short form columns (keeps all rows)
    output_df, unexpected_length_count, unexpected_lengths_summary = add_short_form_columns(df)
    
    # Step 5: Save output
    print("SAVING OUTPUT")
    
    output_path = Path(OUTPUT_FILE)
    output_df.to_csv(output_path, index=False)
    print(f"Saved output to: {output_path}")
    print(f"Input shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Output shape: {output_df.shape[0]} rows, {output_df.shape[1]} columns")
    print(f"Added columns: 33 (11 per questionnaire: 10 item scores + 1 total)")
    
    # Step 6: Print summary
    print_summary(output_df, unexpected_length_count, unexpected_lengths_summary)
    
    print("Processing complete!")


if __name__ == '__main__':
    main()
