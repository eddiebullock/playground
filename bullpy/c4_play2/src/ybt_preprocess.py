import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocess_ybt_data(input_path, output_path):
    # Load data
    df = pd.read_csv(input_path)

    # Rename columns to match expected names where possible
    df = df.rename(columns={
        'hand': 'handedness',
        'edu': 'education',
        'country': 'country_region',
    })

    # --- AUTISM TARGET CREATION (strict: only explicit autism mention in diagnosis) ---
    autism_keywords = ['autism', 'asd', 'asperger', 'autistic']
    autism_from_diagnosis = df['diagnosis'].astype(str).str.contains('|'.join(autism_keywords), case=False, na=False)
    df['autism_target'] = autism_from_diagnosis.astype(int)

    # Print autism target stats
    print("\n=== Autism target variable analysis ===")
    print(f"total participants: {len(df)}")
    print(f"autism cases: {df['autism_target'].sum()}")
    print(f"non-autism cases: {(df['autism_target'] == 0).sum()}")
    print(f"autism percentage: {df['autism_target'].mean() * 100:.2f}%")
    print("\n=== class imbalance ===")
    print(f"autism: {df['autism_target'].sum()} ({df['autism_target'].mean() * 100:.2f}%)")
    print(f"non-autism: {(df['autism_target'] == 0).sum()} ({(1-df['autism_target'].mean()) * 100:.1f}%)")
    print("\n=== VERIFICATION ===")
    print("Autism cases breakdown:")
    autism_cases = df[df['autism_target'] == 1]
    print(autism_cases['diagnosis'].value_counts().head(10))
    print("\nCases with autism in diagnosis:")
    print(f"Count: {len(autism_cases)}")

    # Impute demographic columns with 'unknown'
    demographic_cols = ['sex', 'handedness', 'education', 'country_region']
    for col in demographic_cols:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')

    # Identify candidate questionnaire columns by name
    candidate_cols = [col for col in df.columns if any(q in col for q in ['eq10_', 'sq10_', 'aq_'])]

    # Map Likert responses to numbers (force unmapped to NaN)
    likert_map = {
        'strongly disagree': 1,
        'slightly disagree': 2,
        'slightly agree': 3,
        'strongly agree': 4
    }
    for col in candidate_cols:
        df[col] = df[col].astype(str).str.strip().str.lower().map(likert_map)

    # Try to convert to numeric, keep only those that are at least 10% non-NaN
    questionnaire_cols = []
    for col in candidate_cols:
        numeric_col = pd.to_numeric(df[col], errors='coerce')
        if numeric_col.notna().mean() > 0.1:
            df[col] = numeric_col
            questionnaire_cols.append(col)

    # Filter out non-data rows: keep only rows where age is numeric
    before_rows = df.shape[0]
    df = df[pd.to_numeric(df['age'], errors='coerce').notna()]
    after_rows = df.shape[0]
    print(f"Rows before filtering for numeric age: {before_rows}, after: {after_rows}")

    # Now impute
    df[questionnaire_cols] = df[questionnaire_cols].fillna(df[questionnaire_cols].median())

    # Aggregate questionnaire scores
    df['eq_total'] = df[[f'eq10_{i}' for i in range(1, 11) if f'eq10_{i}' in questionnaire_cols]].sum(axis=1)
    df['sqr_total'] = df[[f'sq10_{i}' for i in range(1, 11) if f'sq10_{i}' in questionnaire_cols]].sum(axis=1)
    df['aq_total'] = df[[f'aq_{i}' for i in range(1, 11) if f'aq_{i}' in questionnaire_cols]].sum(axis=1)
    
    # Create individual EQ and SQR items to match C4 format
    for i in range(1, 11):
        if f'eq10_{i}' in questionnaire_cols:
            df[f'eq_{i}'] = df[f'eq10_{i}']
        if f'sq10_{i}' in questionnaire_cols:
            df[f'sqr_{i}'] = df[f'sq10_{i}']

    # Ensure age and aggregate columns are numeric
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['eq_total'] = pd.to_numeric(df['eq_total'], errors='coerce')
    df['sqr_total'] = pd.to_numeric(df['sqr_total'], errors='coerce')
    df['aq_total'] = pd.to_numeric(df['aq_total'], errors='coerce')

    # D-score (EQ - SQR)
    df['d_score'] = df['eq_total'] - df['sqr_total']

    # Map sex to numeric for interaction (male=0, female=1, other=2, prefer_not_to_say=3, unknown=4)
    sex_map = {'male': 0, 'female': 1, 'other': 2, 'prefer_not_to_say': 3, 'unknown': 4}
    df['sex_num'] = df['sex'].map(lambda x: sex_map.get(str(x).strip().lower(), 4)) if 'sex' in df.columns else 4

    # Interaction features
    df['age_x_aq'] = df['age'] * df['aq_total']
    df['sex_x_eq'] = df['sex_num'] * df['eq_total']
    if 'handedness' in df.columns:
        df['handedness_x_aq'] = pd.to_numeric(df['handedness'].replace('unknown', 0), errors='coerce').fillna(0) * df['aq_total']
    if 'education' in df.columns:
        df['education_x_aq'] = pd.to_numeric(df['education'].replace('unknown', 0), errors='coerce').fillna(0) * df['aq_total']

    # Drop leaky/non-informative columns (diagnosis columns)
    drop_cols = [col for col in df.columns if col.startswith('diagnosis')]
    df = df.drop(columns=drop_cols, errors='ignore')

    # No occupation column in YBT, so skip is_stem_occupation

    # Remove occupation, country_region, handedness, and education from demographic_cols for one-hot encoding
    demographic_cols = ['sex']

    # Standardize questionnaire features
    scaler = StandardScaler()
    if questionnaire_cols:
        df[questionnaire_cols] = scaler.fit_transform(df[questionnaire_cols])

    # Drop rows with missing questionnaire data
    before_dropna = df.shape[0]
    df = df.dropna(subset=questionnaire_cols)
    after_dropna = df.shape[0]
    print(f"Rows before dropping missing questionnaire data: {before_dropna}, after: {after_dropna}")

    # One-hot encode demographic columns (sex only)
    if 'sex' in df.columns:
        df = pd.get_dummies(df, columns=demographic_cols, drop_first=True)

    # --- Feature Engineering (must match notebook as closely as possible) ---
    # Age group bins (categorical)
    df['age_group'] = pd.cut(df['age'], bins=[0, 18, 30, 45, 60, 100], labels=['0-18', '19-30', '31-45', '46-60', '61+'])

    # Nonlinear transformations
    df['log_aq_total'] = np.log1p(df['aq_total'])
    df['sqrt_age'] = np.sqrt(df['age'])

    # Interaction terms
    df['aq_eq_interaction'] = df['aq_total'] * df['eq_total']
    # No spq_total, so skip spq_aq_interaction
    df['age_x_eq'] = df['age'] * df['eq_total']

    # Questionnaire score ratios
    # No spq_total, so skip aq_spq_ratio
    df['eq_sqr_ratio'] = df['eq_total'] / (df['sqr_total'] + 1e-8)

    # Boolean: high AQ (above 1 std)
    df['high_aq'] = (df['aq_total'] > df['aq_total'].mean() + df['aq_total'].std()).astype(int)

    # One-hot encode new categorical features (age_group)
    df = pd.get_dummies(df, columns=['age_group'], drop_first=True)

    # Drop columns that are no longer needed, including any occupation/country_region/handedness/education info
    drop_cols = ['occupation', 'country_region', 'handedness', 'education']
    drop_cols += [col for col in df.columns if col.startswith('occupation_') or col.startswith('country_region_') or col.startswith('handedness_') or col.startswith('education_')]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Impute any remaining NaNs in the DataFrame with 0
    df = df.fillna(0)

    # --- STANDARDIZE AGGREGATE FEATURES USING C4 MEAN & STD ---
    # Load C4 processed data for means and stds
    c4_path = "/Users/eb2007/playground/bullpy/c4_play2/data/processed/data_c4_processed.csv"
    c4 = pd.read_csv(c4_path)
    agg_cols = ['eq_total', 'sqr_total', 'aq_total', 'd_score']
    for col in agg_cols:
        if col in df.columns and col in c4.columns:
            mean = c4[col].mean()
            std = c4[col].std()
            df[col + '_std'] = (df[col] - mean) / std
            print(f"Standardized {col} in YBT using C4 mean={mean:.4f}, std={std:.4f}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"Processed YBT data saved to {output_path}. Shape: {df.shape}")

if __name__ == "__main__":
    input_path = "/Users/eb2007/playground/bullpy/c4_play2/data/raw/YBT.csv"
    output_path = "/Users/eb2007/playground/bullpy/c4_play2/data/processed/YBT_processed.csv"
    preprocess_ybt_data(input_path, output_path)
