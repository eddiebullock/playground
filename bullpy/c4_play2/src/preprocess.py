import pandas as pd
import os

def preprocess_data(input_path, output_path):
    # Load data
    df = pd.read_csv(input_path)

    # Remove test user IDs
    df = df[df['userid'] > 174283]

    # Create autism_target column
    autism_cols = [col for col in df.columns if 'autism_diagnosis' in col]
    diagnosis_cols = [col for col in df.columns if col.startswith('diagnosis_') and not 'autism' in col]
    autism_from_specific = df[autism_cols].fillna(0).ge(1).any(axis=1)
    autism_from_general = df[diagnosis_cols].fillna(0).eq(2).any(axis=1)
    df['autism_target'] = (autism_from_specific | autism_from_general).astype(int)

    # Impute demographic columns with 'unknown'
    demographic_cols = ['sex', 'handedness', 'education', 'occupation', 'country_region']
    for col in demographic_cols:
        df[col] = df[col].fillna('unknown')

    # Impute questionnaire scores with median
    questionnaire_cols = [col for col in df.columns if any(q in col for q in ['spq_', 'eq_', 'sqr_', 'aq_'])]
    df[questionnaire_cols] = df[questionnaire_cols].fillna(df[questionnaire_cols].median())

    # Drop rows with missing questionnaire data
    df = df.dropna(subset=questionnaire_cols)

    # One-hot encode demographic columns
    df = pd.get_dummies(df, columns=demographic_cols, drop_first=True)

    # Impute any remaining NaNs in the DataFrame with 0
    df = df.fillna(0)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}. Shape: {df.shape}")

if __name__ == "__main__":
    # Use the absolute path for input, and save to the processed directory
    input_path = "/Users/eb2007/playground/bullpy/c4_play2/data/raw/data_c4_raw.csv"
    output_path = "/Users/eb2007/playground/bullpy/c4_play2/data/processed/data_c4_processed.csv"
    preprocess_data(input_path, output_path) 