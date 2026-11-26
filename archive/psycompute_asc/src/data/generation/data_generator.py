"""
Data generation module for creating synthetic test data that mimics real-world autism research datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import random

def generate_realistic_data(n_samples: int = 1000, assessment_type: str = 'AQ') -> pd.DataFrame:
    """
    Generate synthetic test data that mimics real-world autism research datasets.
    
    Args:
        n_samples: Number of samples to generate
        assessment_type: Type of assessment ('AQ', 'SQ', or 'EQ')
        
    Returns:
        DataFrame containing synthetic test data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define assessment-specific parameters based on published research
    assessment_params = {
        'AQ': {
            'n_questions': 50,
            'score_range': (0, 50),
            'typical_mean': 16.4,
            'typical_std': 6.3,
            'asd_mean': 35.8,
            'asd_std': 6.5,
            'reverse_items': [2, 4, 5, 6, 7, 9, 12, 13, 16, 17, 21, 22, 23, 26, 33, 35, 39, 41, 42, 43, 45, 46],
            'subscales': {
                'social': list(range(1, 11)),
                'attention_switch': list(range(11, 21)),
                'attention_detail': list(range(21, 31)),
                'communication': list(range(31, 41)),
                'imagination': list(range(41, 51))
            }
        },
        'EQ': {
            'n_questions': 60,
            'score_range': (0, 80),
            'typical_mean': 42.1,
            'typical_std': 11.2,
            'asd_mean': 20.4,
            'asd_std': 11.4,
            'reverse_items': [1, 6, 19, 22, 25, 26, 35, 36, 37, 38, 41, 42, 43, 44, 52, 54, 55, 57, 58, 59, 60],
            'subscales': {
                'cognitive': list(range(1, 21)),
                'emotional': list(range(21, 41)),
                'social': list(range(41, 61))
            }
        },
        'SQ': {
            'n_questions': 50,
            'score_range': (0, 150),
            'typical_mean': 65.3,
            'typical_std': 15.2,
            'asd_mean': 90.1,
            'asd_std': 18.4,
            'reverse_items': [2, 4, 5, 6, 7, 9, 12, 13, 16, 17, 21, 22, 23, 26, 33, 35, 39, 41, 42, 43, 45, 46],
            'subscales': {
                'patterns': list(range(1, 17)),
                'details': list(range(17, 33)),
                'systems': list(range(33, 51))
            }
        }
    }
    
    params = assessment_params[assessment_type]
    
    # Generate participant IDs
    participant_ids = [f'SUBJ_{i:06d}' for i in range(n_samples)]
    
    # Generate demographic data based on published research
    ages = np.random.normal(35, 12, n_samples).clip(18, 80)
    genders = np.random.choice(['M', 'F', 'Other'], size=n_samples, p=[0.48, 0.48, 0.04])
    ethnicities = np.random.choice(
        ['White', 'Asian', 'Black', 'Hispanic', 'Other'], 
        size=n_samples, 
        p=[0.45, 0.25, 0.15, 0.12, 0.03]
    )
    
    # Generate diagnosis status with realistic ASD prevalence (1.7%)
    asd_status = np.random.choice([0, 1], size=n_samples, p=[0.983, 0.017])
    
    # Generate assessment scores based on diagnosis status
    scores = []
    for status in asd_status:
        if status == 1:  # ASD
            score = np.random.normal(params['asd_mean'], params['asd_std'])
        else:  # Typical
            score = np.random.normal(params['typical_mean'], params['typical_std'])
        scores.append(score)
    
    scores = np.array(scores).clip(params['score_range'][0], params['score_range'][1])
    
    # Generate individual question responses with realistic patterns
    responses = []
    for i, score in enumerate(scores):
        # Generate base responses for each subscale
        raw_responses = np.zeros(params['n_questions'])
        
        for subscale, items in params['subscales'].items():
            # Calculate subscale score based on total score
            subscale_mean = score / len(params['subscales'])
            subscale_std = 0.8  # Natural variation within subscale
            
            # Generate responses for this subscale
            subscale_responses = np.random.normal(subscale_mean, subscale_std, len(items))
            subscale_responses = subscale_responses.clip(0, 4)
            
            # Add some correlation between items in the same subscale
            for j in range(len(items)):
                if j > 0:
                    # Add correlation with previous item
                    subscale_responses[j] = 0.7 * subscale_responses[j-1] + 0.3 * subscale_responses[j]
            
            # Assign responses to the appropriate items
            for j, item_idx in enumerate(items):
                raw_responses[item_idx-1] = subscale_responses[j]
        
        # Add some natural variation
        variation = np.random.normal(0, 0.3, params['n_questions'])
        raw_responses = (raw_responses + variation).clip(0, 4)
        
        # Handle reverse-scored items
        for idx in params['reverse_items']:
            if idx <= len(raw_responses):
                raw_responses[idx-1] = 4 - raw_responses[idx-1]
        
        # Add some random missing values (1% missing rate)
        missing_mask = np.random.random(params['n_questions']) < 0.01
        raw_responses[missing_mask] = np.nan
        
        responses.append(raw_responses)
    
    # Create DataFrame
    df = pd.DataFrame({
        'participant_id': participant_ids,
        'age': ages.round(1),
        'gender': genders,
        'ethnicity': ethnicities,
        'asd_diagnosis': asd_status,
        f'{assessment_type}_total_score': scores.round(1)
    })
    
    # Add individual question responses
    for i in range(params['n_questions']):
        df[f'{assessment_type}_{i+1}'] = [r[i] for r in responses]
    
    # Add metadata columns
    df['date_of_assessment'] = pd.date_range(
        start='2020-01-01', 
        periods=n_samples, 
        freq='D'
    ).strftime('%Y-%m-%d')
    
    df['site_id'] = np.random.choice(['SITE_001', 'SITE_002', 'SITE_003'], size=n_samples)
    df['version'] = '1.0'
    df['language'] = np.random.choice(['English', 'Spanish', 'French'], size=n_samples, p=[0.85, 0.1, 0.05])
    
    return df 

def save_test_data(df: pd.DataFrame, assessment_type: str):
    """
    Save the generated test data to the appropriate location.
    
    Args:
        df: DataFrame containing the test data
        assessment_type: Type of assessment ('AQ', 'SQ', or 'EQ')
    """
    # Create the output directory if it doesn't exist
    output_dir = Path('/Users/eb2007/bulldev/data/raw/Data-downloads/kaggle') / assessment_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the data
    output_file = output_dir / f'{assessment_type.lower()}_test_data.csv'
    df.to_csv(output_file, index=False)
    print(f"Saved test data to {output_file}")

def generate_all_test_datasets(n_samples: int = 1000):
    """
    Generate test datasets for all assessment types.
    
    Args:
        n_samples: Number of samples to generate for each assessment type
    """
    for assessment_type in ['AQ', 'EQ', 'SQ']:
        print(f"\nGenerating {assessment_type} test data...")
        df = generate_realistic_data(n_samples=n_samples, assessment_type=assessment_type)
        save_test_data(df, assessment_type)
        print(f"Generated {n_samples} samples for {assessment_type}")

if __name__ == "__main__":
    generate_all_test_datasets(n_samples=1000) 