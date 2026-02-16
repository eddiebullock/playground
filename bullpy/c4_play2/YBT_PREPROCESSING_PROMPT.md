# YBT Dataset Preprocessing Prompt for C4 Model Validation

## Objective
Preprocess the YBT dataset (`/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/YBT.csv`) to create exactly 45 features matching the C4 training feature space, enabling proper external validation of the trained C4 models.

## Critical Requirements

### 1. Data Leakage Prevention
- **EXCLUDE all AQ-related features** from the final feature set (AQ items, AQ total, AQ interactions)
- AQ features are excluded in C4 training (`excluded_features` in `feature_info_original.json`)
- Do NOT use AQ features for predictions - they create circularity

### 2. Feature Alignment
- Must create exactly 45 features matching `models/cross_validation/feature_info_original.json`
- Features must be in the EXACT same order as C4 training
- Missing features (SPQ) must be filled with 0, not NaN

### 3. Scaler Application
- Use the saved scaler from C4 training (`scaler_original.joblib`)
- Do NOT refit the scaler on YBT data
- Apply scaler ONLY after feature alignment

## Step-by-Step Preprocessing Pipeline

### STEP 1: Load and Initial Data Inspection

```python
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.preprocessing import StandardScaler

# Load YBT data
ybt_path = '/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/YBT.csv'
df = pd.read_csv(ybt_path)

# Verify questionnaire columns exist
print("Checking questionnaire columns...")
eq_cols = [col for col in df.columns if col.startswith('eq10_')]
sqr_cols = [col for col in df.columns if col.startswith('sq10_')]
aq_cols = [col for col in df.columns if col.startswith('aq_') and len(col) <= 5]  # aq_1 to aq_10

print(f"EQ columns found: {len(eq_cols)} - {eq_cols}")
print(f"SQR columns found: {len(sqr_cols)} - {sqr_cols}")
print(f"AQ columns found: {len(aq_cols)} - {aq_cols}")
```

### STEP 2: Convert Text Responses to Numeric

```python
# YBT uses text responses, need to convert to numeric
response_mapping = {
    'strongly agree': 4,
    'slightly agree': 3,
    'slightly disagree': 2,
    'strongly disagree': 1
}

# Convert all questionnaire columns
all_q_cols = eq_cols + sqr_cols + aq_cols
for col in all_q_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower().map(response_mapping)
        df[col] = pd.to_numeric(df[col], errors='coerce')
```

### STEP 3: Score Questionnaires (Matching C4 Rules)

#### EQ-10 Scoring (Binary 0-1 with reverse-scoring)
```python
# EQ-10: Items 1,2,4,5,6,7,8,9,10: Agree (3,4) = 1 point
#        Item 3: Disagree (1,2) = 1 point (reverse-scored)
eq_reverse_items = [3]

for i in range(1, 11):
    col_name = f'eq10_{i}'
    if col_name in df.columns:
        if i in eq_reverse_items:
            # Reverse: disagree (1,2) = 1, agree (3,4) = 0
            df[col_name] = df[col_name].apply(lambda x: 1 if pd.notna(x) and x in [1, 2] else 0 if pd.notna(x) and x in [3, 4] else np.nan)
        else:
            # Normal: agree (3,4) = 1, disagree (1,2) = 0
            df[col_name] = df[col_name].apply(lambda x: 1 if pd.notna(x) and x in [3, 4] else 0 if pd.notna(x) and x in [1, 2] else np.nan)

# Calculate EQ total
df['eq_total'] = df[eq_cols].sum(axis=1)

# Map eq10_* to eq_* for C4 compatibility
for i in range(1, 11):
    if f'eq10_{i}' in df.columns:
        df[f'eq_{i}'] = df[f'eq10_{i}']
```

#### SQR-10 Scoring (Binary 0-1 with reverse-scoring)
```python
# SQR-10: Items 1,3,5,7,9: Agree (3,4) = 1 point
#         Items 2,4,6,8,10: Disagree (1,2) = 1 point (reverse-scored)
sqr_reverse_items = [2, 4, 6, 8, 10]

for i in range(1, 11):
    col_name = f'sq10_{i}'
    if col_name in df.columns:
        if i in sqr_reverse_items:
            # Reverse: disagree (1,2) = 1, agree (3,4) = 0
            df[col_name] = df[col_name].apply(lambda x: 1 if pd.notna(x) and x in [1, 2] else 0 if pd.notna(x) and x in [3, 4] else np.nan)
        else:
            # Normal: agree (3,4) = 1, disagree (1,2) = 0
            df[col_name] = df[col_name].apply(lambda x: 1 if pd.notna(x) and x in [3, 4] else 0 if pd.notna(x) and x in [1, 2] else np.nan)

# Calculate SQR total
df['sqr_total'] = df[sqr_cols].sum(axis=1)

# Map sq10_* to sqr_* for C4 compatibility
for i in range(1, 11):
    if f'sq10_{i}' in df.columns:
        df[f'sqr_{i}'] = df[f'sq10_{i}']
```

#### AQ-10 Scoring (For Reference Only - Will Be Excluded)
```python
# AQ-10: Items 1,7,8,10: Agree (3,4) = 1 point
#        Items 2,3,4,5,6,9: Disagree (1,2) = 1 point (reverse-scored)
aq_reverse_items = [2, 3, 4, 5, 6, 9]

for i in range(1, 11):
    col_name = f'aq_{i}'
    if col_name in df.columns:
        if i in aq_reverse_items:
            df[col_name] = df[col_name].apply(lambda x: 1 if pd.notna(x) and x in [1, 2] else 0 if pd.notna(x) and x in [3, 4] else np.nan)
        else:
            df[col_name] = df[col_name].apply(lambda x: 1 if pd.notna(x) and x in [3, 4] else 0 if pd.notna(x) and x in [1, 2] else np.nan)

# Calculate AQ total (for reference only)
df['aq_total'] = df[aq_cols].sum(axis=1)
```

#### SPQ-10 (Missing in YBT - Fill with 0)
```python
# YBT does NOT have SPQ - create all SPQ features as 0
for i in range(1, 11):
    df[f'spq_{i}'] = 0

df['spq_total'] = 0
```

### STEP 4: Create Target Variable

```python
# Create autism_target from diagnosis columns
if 'diagnosis' in df.columns:
    # Check if 'autism' appears in diagnosis column (case-insensitive)
    df['autism_target'] = df['diagnosis'].astype(str).str.contains('autism', case=False, na=False).astype(int)
    
    # Also check diagnosis_yes_no if available
    if 'diagnosis_yes_no' in df.columns:
        diagnosis_yes = (df['diagnosis_yes_no'].astype(str).str.lower().str.strip() == 'yes').astype(int)
        # Combine: autism in diagnosis OR (diagnosis_yes AND autism in diagnosis)
        df['autism_target'] = ((df['autism_target'] == 1) | (diagnosis_yes == 1)).astype(int)
else:
    print("WARNING: No diagnosis column found - cannot create autism_target")
    df['autism_target'] = 0

print(f"Autism target distribution: {df['autism_target'].value_counts().to_dict()}")
```

### STEP 5: Demographic Feature Engineering

```python
# Age: Use 'age' column, ensure numeric
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['age'] = df['age'].fillna(df['age'].median())

# Sex: Map to numeric (matching C4 encoding)
# C4 uses: 1=male, 2=female, 3=other, 4=prefer not to say
sex_mapping = {
    'male': 1,
    'female': 2,
    'other': 3,
    'prefer not to say': 4,
    'i prefer not to say': 4,
    'i do not know': 4
}
if 'sex' in df.columns:
    df['sex'] = df['sex'].astype(str).str.strip().str.lower().map(sex_mapping).fillna(4)
    df['sex_num'] = df['sex'].map({1: 0, 2: 1, 3: 2, 4: 3}).fillna(0).astype(int)
else:
    df['sex'] = 4  # Unknown
    df['sex_num'] = 0

# Age groups (matching C4 bins)
df['age_group_19-30'] = ((df['age'] >= 19) & (df['age'] <= 30)).astype(int)
df['age_group_31-45'] = ((df['age'] >= 31) & (df['age'] <= 45)).astype(int)
df['age_group_46-60'] = ((df['age'] >= 46) & (df['age'] <= 60)).astype(int)
df['age_group_61+'] = (df['age'] >= 61).astype(int)

# sqrt_age
df['sqrt_age'] = np.sqrt(np.clip(df['age'], a_min=0, a_max=None))
```

### STEP 6: Questionnaire Totals and Interactions

```python
# d_score (SQR - EQ, matching C4)
df['d_score'] = df['sqr_total'] - df['eq_total']

# age_x_eq interaction
df['age_x_eq'] = df['age'] * df['eq_total']

# eq_sqr_ratio
df['eq_sqr_ratio'] = df['eq_total'] / (df['sqr_total'].replace(0, np.nan) + 1e-8)
df['eq_sqr_ratio'] = df['eq_sqr_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
```

### STEP 7: Occupation Feature

```python
# is_stem_occupation (binary flag)
if 'Q549' in df.columns or 'occupation' in df.columns:
    occupation_col = 'Q549' if 'Q549' in df.columns else 'occupation'
    df['is_stem_occupation'] = df[occupation_col].astype(str).str.contains(
        'science|technology|engineering|math|computer|software|data|research', 
        case=False, na=False
    ).astype(int)
else:
    df['is_stem_occupation'] = 0
```

### STEP 8: Feature Alignment to C4 Schema

```python
# Load C4 feature schema
feature_info_path = '/Users/eb2007/playground/bullpy/c4_play2/models/cross_validation/feature_info_original.json'
with open(feature_info_path, 'r') as f:
    feature_info = json.load(f)

c4_feature_names = feature_info['feature_names']  # 45 features
excluded_features = set(feature_info.get('excluded_features', []))

print(f"C4 expects {len(c4_feature_names)} features")
print(f"Excluded features (will not be used): {excluded_features}")

# Build aligned feature matrix
X_ybt = pd.DataFrame(index=df.index)
missing_features = []

for feat_name in c4_feature_names:
    if feat_name in df.columns:
        X_ybt[feat_name] = df[feat_name]
    else:
        # Missing feature - fill with 0
        X_ybt[feat_name] = 0
        missing_features.append(feat_name)

# Ensure correct order
X_ybt = X_ybt[c4_feature_names]

print(f"Aligned feature matrix shape: {X_ybt.shape}")
print(f"Missing features filled with 0: {len(missing_features)}")
if missing_features:
    print(f"  Missing: {missing_features[:10]}...")

# Handle missing values
X_ybt = X_ybt.fillna(0)
X_ybt = X_ybt.apply(pd.to_numeric, errors='coerce').fillna(0)
```

### STEP 9: Apply C4 Scaler (NO REFITTING)

```python
# Load saved scaler from C4 training
scaler_path = '/Users/eb2007/playground/bullpy/c4_play2/models/cross_validation/scaler_original.joblib'
scaler = joblib.load(scaler_path)

# Apply scaler (fitted on C4, NOT refit on YBT)
X_ybt_scaled = scaler.transform(X_ybt.values)

print(f"Scaled feature matrix shape: {X_ybt_scaled.shape}")
print("✅ Applied C4 scaler (no refitting)")
```

### STEP 10: Load Models and Generate Predictions

```python
# Load trained models
models_dir = '/Users/eb2007/playground/bullpy/c4_play2/models/cross_validation'
models = {
    'Logistic Regression': joblib.load(f'{models_dir}/logistic_regression_original.joblib'),
    'Random Forest': joblib.load(f'{models_dir}/random_forest_original.joblib'),
    'Gradient Boosting': joblib.load(f'{models_dir}/gradient_boosting_original.joblib'),
    'XGBoost': joblib.load(f'{models_dir}/xgboost_original.joblib'),
    'LightGBM': joblib.load(f'{models_dir}/lightgbm_original.joblib'),
}

# Generate predictions
predictions = {}
probabilities = {}

for name, model in models.items():
    y_proba = model.predict_proba(X_ybt_scaled)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    predictions[name] = y_pred
    probabilities[name] = y_proba
    print(f"{name}: Generated predictions for {len(y_pred)} samples")
```

### STEP 11: Evaluate Performance (if ground truth available)

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

if 'autism_target' in df.columns:
    y_true = df['autism_target'].values
    
    results = {}
    for name in models.keys():
        y_pred = predictions[name]
        y_proba = probabilities[name]
        
        results[name] = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_proba)
        }
    
    results_df = pd.DataFrame(results).T
    print("\nExternal Validation Results on YBT:")
    print(results_df.round(4).sort_values('auc', ascending=False))
    
    # Save results
    results_df.to_csv('/Users/eb2007/playground/bullpy/c4_play2/data/processed/ybt_external_validation_results.csv')
else:
    print("No ground truth labels available - predictions only")
```

### STEP 12: Save Predictions

```python
# Save predictions
pred_df = pd.DataFrame({
    'userid': df.index if 'userid' not in df.columns else df['userid']
})

for name in models.keys():
    pred_df[f'proba_{name.replace(" ", "_").lower()}'] = probabilities[name]
    pred_df[f'pred_{name.replace(" ", "_").lower()}'] = predictions[name]

pred_df.to_csv('/Users/eb2007/playground/bullpy/c4_play2/data/processed/ybt_external_predictions.csv', index=False)
print("Predictions saved")
```

## Validation Checklist

- [ ] All 45 C4 features created in exact order
- [ ] SPQ features filled with 0 (YBT doesn't have SPQ)
- [ ] EQ and SQR items mapped correctly (eq10_* → eq_*, sq10_* → sqr_*)
- [ ] Questionnaire scoring matches C4 rules (binary 0-1 with reverse-scoring)
- [ ] AQ features NOT included in final feature set (data leakage prevention)
- [ ] Scaler applied from C4 (no refitting)
- [ ] Missing values handled (filled with 0 or median)
- [ ] Feature order matches C4 exactly
- [ ] All features are numeric
- [ ] No infinite values

## Expected Outputs

1. **Preprocessed YBT dataset** with 45 features matching C4
2. **External validation metrics** (if ground truth available)
3. **Predictions** from all 5 models
4. **Feature alignment report** showing which features were missing/filled

## Notes

- **SPQ = Sensory Perception Quotient** (not Schizotypal Personality Questionnaire)
- **SQR = Systemizing Quotient-Revised** (not Social Responsiveness Scale)
- YBT is missing SPQ compared to C4 - this is expected and handled by filling with 0
- AQ features are excluded to prevent data leakage (AQ is a direct autism screening tool)
