import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier

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

    # Aggregate questionnaire scores
    df['spq_total'] = df[[f'spq_{i}' for i in range(1, 11)]].sum(axis=1)
    df['eq_total'] = df[[f'eq_{i}' for i in range(1, 11)]].sum(axis=1)
    df['sqr_total'] = df[[f'sqr_{i}' for i in range(1, 11)]].sum(axis=1)
    df['aq_total'] = df[[f'aq_{i}' for i in range(1, 11)]].sum(axis=1)

    # D-score (EQ - SQR)
    df['d_score'] = df['eq_total'] - df['sqr_total']

    # Map sex to numeric for interaction (male=0, female=1, other=2, prefer_not_to_say=3, unknown=4)
    sex_map = {'male': 0, 'female': 1, 'other': 2, 'prefer_not_to_say': 3, 'unknown': 4}
    df['sex_num'] = df['sex'].map(sex_map).fillna(4)

    # Interaction features
    df['age_x_aq'] = df['age'] * df['aq_total']
    df['sex_x_eq'] = df['sex_num'] * df['eq_total']
    df['handedness_x_aq'] = df['handedness'].replace('unknown', 0).astype(float) * df['aq_total']
    df['education_x_aq'] = df['education'].replace('unknown', 0).astype(float) * df['aq_total']

    # Drop leaky/non-informative columns (diagnosis columns)
    drop_cols = [col for col in df.columns if col.startswith('diagnosis_') or col.startswith('autism_diagnosis')]
    df = df.drop(columns=drop_cols, errors='ignore')

    # Create is_stem_occupation column
    stem_occupation_codes = {2, 3, 5, 21}
    def is_stem(occupation_code):
        try:
            return int(float(occupation_code) in stem_occupation_codes)
        except:
            return 0
    if 'occupation' in df.columns:
        df['is_stem_occupation'] = df['occupation'].apply(is_stem)
    
    # Remove occupation, country_region, handedness, and education from demographic_cols for one-hot encoding
    demographic_cols = ['sex']

    # Standardize questionnaire features
    questionnaire_cols = [col for col in df.columns if col.startswith(('spq_', 'eq_', 'sqr_', 'aq_'))]
    scaler = StandardScaler()
    if questionnaire_cols:
        df[questionnaire_cols] = scaler.fit_transform(df[questionnaire_cols])

    # Drop rows with missing questionnaire data
    df = df.dropna(subset=questionnaire_cols)

    # One-hot encode demographic columns (sex only)
    df = pd.get_dummies(df, columns=demographic_cols, drop_first=True)

    # Drop columns that are no longer needed, including any occupation/country_region/handedness/education info
    drop_cols = ['userid', 'repeat', 'occupation', 'country_region', 'handedness', 'education']
    drop_cols += [col for col in df.columns if col.startswith('occupation_') or col.startswith('country_region_') or col.startswith('handedness_') or col.startswith('education_')]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Impute any remaining NaNs in the DataFrame with 0
    df = df.fillna(0)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save processed data
    # Note: The numbers in the output CSV represent the processed feature values for each participant/row, including standardized questionnaire scores, engineered features, and the binary is_stem_occupation indicator. Each column is a feature used for modeling.
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}. Shape: {df.shape}")

if __name__ == "__main__":
    # Use the absolute path for input, and save to the processed directory
    input_path = "/Users/eb2007/playground/bullpy/c4_play2/data/raw/data_c4_raw.csv"
    output_path = "/Users/eb2007/playground/bullpy/c4_play2/data/processed/data_c4_processed.csv"
    preprocess_data(input_path, output_path) 