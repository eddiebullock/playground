# CARD Dataset Preprocessing and C4 Model Validation Prompt

## Objective
Preprocess the CARD dataset (`/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/CARD_Nov2025.xlsx`) to create exactly 45 features matching the C4 training feature space, enabling proper external validation of the original C4-trained models (with SPQ).

## Critical Requirements

### 1. Data Structure Understanding
- **CARD has one row per questionnaire entry** (not per participant)
- **Multiple questionnaires per participant** (indicated by `VolunteerID` column)
- **Itemized scores are CSV strings** in the `itemised score` column
- **Must aggregate to one row per participant** before preprocessing

### 2. Data Leakage Prevention
- **EXCLUDE all AQ-related features** from the final feature set (AQ items, AQ total, AQ interactions)
- AQ features are excluded in C4 training (`excluded_features` in `feature_info_original.json`)
- Do NOT use AQ features for predictions - they create circularity

### 3. Feature Alignment
- Must create exactly 45 features matching `models/cross_validation/feature_info_original.json`
- Features must be in the EXACT same order as C4 training
- **SPQ features MUST be included** (unlike YBT, CARD has SPQ data)

### 4. Scaler Application
- Use the saved scaler from C4 training (`scaler_original.joblib`)
- Do NOT refit the scaler on CARD data
- Apply scaler ONLY after feature alignment

### 5. File Access
- File may be password-protected: `£ddie4ever!`
- Use `msoffcrypto-tool` library for decryption if needed

## Step-by-Step Preprocessing Pipeline

### STEP 1: Load CARD Dataset and Understand Structure

```python
import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Paths
CARD_PATH = '/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/CARD_Nov2025.xlsx'
ARTIFACT_DIR = '/Users/eb2007/playground/bullpy/c4_play2/models/cross_validation'
FEATURE_INFO_PATH = os.path.join(ARTIFACT_DIR, 'feature_info_original.json')
SCALER_PATH = os.path.join(ARTIFACT_DIR, 'scaler_original.joblib')

# Excel password (if needed)
EXCEL_PASSWORD = '£ddie4ever!'

# Load CARD dataset
print("="*80)
print("LOADING CARD DATASET")
print("="*80)

if CARD_PATH.lower().endswith(('.xlsx', '.xls')):
    try:
        df_card = pd.read_excel(CARD_PATH, engine='openpyxl')
        print("✅ Successfully loaded file (no password required)")
    except Exception as e:
        if 'BadZipFile' in str(type(e).__name__) or 'encrypted' in str(e).lower():
            print("⚠️  File appears to be password-protected.")
            print("Attempting to decrypt...")
            try:
                import msoffcrypto
                import io
                decrypted = io.BytesIO()
                with open(CARD_PATH, 'rb') as f:
                    office_file = msoffcrypto.OfficeFile(f)
                    office_file.load_key(password=EXCEL_PASSWORD)
                    office_file.decrypt(decrypted)
                    decrypted.seek(0)
                    df_card = pd.read_excel(decrypted, engine='openpyxl')
                print("✅ Successfully decrypted and loaded file")
            except ImportError:
                print("Installing msoffcrypto-tool...")
                import subprocess
                import sys
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'msoffcrypto-tool'])
                raise Exception("Please re-run after installation")
        else:
            raise

print(f"\nCARD dataset shape (before aggregation): {df_card.shape}")
print(f"Columns: {list(df_card.columns)}")

# Identify key columns
volunteer_id_col = None
for col in df_card.columns:
    if 'volunteer' in col.lower() and 'id' in col.lower():
        volunteer_id_col = col
        break

if volunteer_id_col is None:
    # Try alternatives
    for col in df_card.columns:
        if 'id' in col.lower() and df_card[col].nunique() < len(df_card) * 0.5:
            volunteer_id_col = col
            break

print(f"\nVolunteer ID column: {volunteer_id_col}")
print(f"Unique participants: {df_card[volunteer_id_col].nunique()}")
print(f"Total questionnaire entries: {len(df_card)}")
print(f"Average entries per participant: {len(df_card) / df_card[volunteer_id_col].nunique():.2f}")
```

### STEP 2: Parse Itemized Scores from CSV Column

```python
print("\n" + "="*80)
print("PARSING ITEMIZED SCORES FROM CSV COLUMN")
print("="*80)

# Find itemised score column
itemised_col = None
for col in df_card.columns:
    if 'itemis' in col.lower() or ('item' in col.lower() and 'score' in col.lower()):
        itemised_col = col
        break

if itemised_col is None:
    raise ValueError("Could not find 'itemised score' column!")

print(f"Found itemised score column: '{itemised_col}'")

# Parse CSV itemized scores
def parse_itemised_scores(itemised_str, test_name):
    """
    Parse CSV itemized scores and return as dictionary
    Format: "item1,item2,item3,..." or "1,2,3,..."
    """
    if pd.isna(itemised_str):
        return {}
    
    try:
        # Split by comma
        items = str(itemised_str).split(',')
        # Clean whitespace and convert to numeric
        items = [float(x.strip()) for x in items if x.strip()]
        
        # Create item dictionary based on test name
        item_dict = {}
        if 'aq' in test_name.lower():
            for i, val in enumerate(items[:10], 1):  # AQ-10
                item_dict[f'aq_{i}'] = val
        elif 'eq' in test_name.lower():
            for i, val in enumerate(items[:10], 1):  # EQ-10
                item_dict[f'eq_{i}'] = val
        elif 'sq' in test_name.lower() or 'sqr' in test_name.lower():
            for i, val in enumerate(items[:10], 1):  # SQR-10
                item_dict[f'sqr_{i}'] = val
        elif 'spq' in test_name.lower():
            for i, val in enumerate(items[:10], 1):  # SPQ-10
                item_dict[f'spq_{i}'] = val
        
        return item_dict
    except Exception as e:
        print(f"Warning: Could not parse itemised scores: {e}")
        return {}

# Find test name column
test_name_col = None
for col in df_card.columns:
    if 'test' in col.lower() or 'questionnaire' in col.lower() or 'name' in col.lower():
        test_name_col = col
        break

if test_name_col is None:
    raise ValueError("Could not find test name column!")

print(f"Found test name column: '{test_name_col}'")
print(f"\nUnique test types: {df_card[test_name_col].unique()}")

# Parse itemized scores for each row
print("\nParsing itemized scores...")
parsed_items = []
for idx, row in df_card.iterrows():
    test_name = str(row[test_name_col]).lower()
    itemised_str = row[itemised_col]
    parsed = parse_itemised_scores(itemised_str, test_name)
    parsed_items.append(parsed)

# Add parsed items as new columns
for item_key in set().union(*parsed_items):
    df_card[item_key] = np.nan

for idx, parsed in enumerate(parsed_items):
    for key, val in parsed.items():
        df_card.loc[idx, key] = val

print(f"✅ Parsed itemized scores")
print(f"   Created columns: {[k for k in set().union(*parsed_items)]}")
```

### STEP 3: Aggregate Multiple Questionnaires Per Participant

```python
print("\n" + "="*80)
print("AGGREGATING MULTIPLE QUESTIONNAIRES PER PARTICIPANT")
print("="*80)

# Strategy: For each participant, collect all questionnaire items
# Priority: Use adult versions (AQ, EQ, SQ, SPQ) over child/adolescent versions
# If multiple entries exist, prefer most recent (if date available) or last row

# Identify date column (if available)
date_col = None
for col in df_card.columns:
    if any(kw in col.lower() for kw in ['date', 'time', 'modified', 'created']):
        date_col = col
        break

if date_col:
    print(f"Found date column: '{date_col}' - will use most recent entry")
    try:
        df_card[date_col] = pd.to_datetime(df_card[date_col], errors='coerce')
        df_card = df_card.sort_values(by=[volunteer_id_col, date_col])
    except:
        print("  Warning: Could not parse as datetime")
        date_col = None

# Group by participant and aggregate questionnaire items
print("\nAggregating questionnaires per participant...")

# Define questionnaire priority (adult > adolescent > child)
test_priority = {
    'aq': 1, 'adolescent aq': 2, 'child aq': 3,
    'eq': 1, 'adolescent eq': 2, 'child eq': 3, 'childeq': 3,
    'sq': 1, 'sqr': 1, 'adolescent sq': 2, 'child sq': 3, 'childsq': 3,
    'spq': 1, 'child_spq': 3
}

def get_priority(test_name):
    test_lower = str(test_name).lower()
    for key, priority in test_priority.items():
        if key in test_lower:
            return priority
    return 99  # Unknown tests get lowest priority

df_card['test_priority'] = df_card[test_name_col].apply(get_priority)

# Aggregate: For each participant, take best available questionnaire for each type
participant_data = []

for volunteer_id in df_card[volunteer_id_col].unique():
    participant_rows = df_card[df_card[volunteer_id_col] == volunteer_id].copy()
    
    # Start with demographic info (take from first row)
    participant_row = participant_rows.iloc[0].copy()
    
    # Collect questionnaire items (prefer higher priority tests)
    questionnaire_items = {}
    
    # Group by test type and take highest priority
    for test_type in ['aq', 'eq', 'sq', 'sqr', 'spq']:
        matching_rows = participant_rows[
            participant_rows[test_name_col].str.lower().str.contains(test_type, na=False)
        ]
        if len(matching_rows) > 0:
            # Take highest priority (lowest number)
            best_row = matching_rows.loc[matching_rows['test_priority'].idxmin()]
            
            # Extract items for this questionnaire
            if test_type == 'aq':
                for i in range(1, 11):
                    col = f'aq_{i}'
                    if col in best_row.index and pd.notna(best_row[col]):
                        questionnaire_items[col] = best_row[col]
            elif test_type == 'eq':
                for i in range(1, 11):
                    col = f'eq_{i}'
                    if col in best_row.index and pd.notna(best_row[col]):
                        questionnaire_items[col] = best_row[col]
            elif test_type in ['sq', 'sqr']:
                for i in range(1, 11):
                    col = f'sqr_{i}'
                    if col in best_row.index and pd.notna(best_row[col]):
                        questionnaire_items[col] = best_row[col]
            elif test_type == 'spq':
                for i in range(1, 11):
                    col = f'spq_{i}'
                    if col in best_row.index and pd.notna(best_row[col]):
                        questionnaire_items[col] = best_row[col]
    
    # Add questionnaire items to participant row
    for key, val in questionnaire_items.items():
        participant_row[key] = val
    
    participant_data.append(participant_row)

df_card_aggregated = pd.DataFrame(participant_data)

# Rename volunteer ID to userid
if volunteer_id_col != 'userid':
    df_card_aggregated = df_card_aggregated.rename(columns={volunteer_id_col: 'userid'})

print(f"\n✅ Aggregated dataset:")
print(f"   Participants: {len(df_card_aggregated)}")
print(f"   Columns: {len(df_card_aggregated.columns)}")
```

### STEP 4: Score Questionnaires (Matching C4 Rules)

```python
print("\n" + "="*80)
print("SCORING QUESTIONNAIRES (MATCHING C4 RULES)")
print("="*80)

# SPQ-10 Scoring (Continuous 0-3 scale, range 0-30)
# C4 uses: 1->3, 2->2, 3->1, 4->0 (so: score = 4 - raw_value)
print("\nSPQ-10 Scoring...")
spq_cols = [f'spq_{i}' for i in range(1, 11)]
for col in spq_cols:
    if col in df_card_aggregated.columns:
        # Convert to numeric
        df_card_aggregated[col] = pd.to_numeric(df_card_aggregated[col], errors='coerce')
        # C4 scoring: if values are 1-4, convert to 0-3 scale
        # Check current range
        if df_card_aggregated[col].notna().any():
            min_val = df_card_aggregated[col].min()
            max_val = df_card_aggregated[col].max()
            if min_val >= 1 and max_val <= 4:
                # Convert 1,2,3,4 to 3,2,1,0
                df_card_aggregated[col] = 4 - df_card_aggregated[col]
            elif min_val >= 0 and max_val <= 3:
                # Already in correct format
                pass

df_card_aggregated['spq_total'] = df_card_aggregated[spq_cols].sum(axis=1)
print(f"  ✅ SPQ-10 scored: total range {df_card_aggregated['spq_total'].min():.0f}-{df_card_aggregated['spq_total'].max():.0f}")

# EQ-10 Scoring (Binary 0-1 with reverse-scoring)
print("\nEQ-10 Scoring...")
eq_cols = [f'eq_{i}' for i in range(1, 11)]
eq_reverse_items = [3]  # Item 3 is reverse-scored

for i in range(1, 11):
    col = f'eq_{i}'
    if col in df_card_aggregated.columns:
        df_card_aggregated[col] = pd.to_numeric(df_card_aggregated[col], errors='coerce')
        if i in eq_reverse_items:
            # Reverse: disagree (1,2) = 1, agree (3,4) = 0
            df_card_aggregated[col] = df_card_aggregated[col].apply(
                lambda x: 1 if pd.notna(x) and x in [1, 2] 
                else 0 if pd.notna(x) and x in [3, 4] 
                else np.nan
            )
        else:
            # Normal: agree (3,4) = 1, disagree (1,2) = 0
            df_card_aggregated[col] = df_card_aggregated[col].apply(
                lambda x: 1 if pd.notna(x) and x in [3, 4] 
                else 0 if pd.notna(x) and x in [1, 2] 
                else np.nan
            )

df_card_aggregated['eq_total'] = df_card_aggregated[eq_cols].sum(axis=1)
print(f"  ✅ EQ-10 scored: total range {df_card_aggregated['eq_total'].min():.0f}-{df_card_aggregated['eq_total'].max():.0f}")

# SQR-10 Scoring (Binary 0-1 with reverse-scoring)
print("\nSQR-10 Scoring...")
sqr_cols = [f'sqr_{i}' for i in range(1, 11)]
sqr_reverse_items = [2, 4, 6, 8, 10]

for i in range(1, 11):
    col = f'sqr_{i}'
    if col in df_card_aggregated.columns:
        df_card_aggregated[col] = pd.to_numeric(df_card_aggregated[col], errors='coerce')
        if i in sqr_reverse_items:
            # Reverse: disagree (1,2) = 1, agree (3,4) = 0
            df_card_aggregated[col] = df_card_aggregated[col].apply(
                lambda x: 1 if pd.notna(x) and x in [1, 2] 
                else 0 if pd.notna(x) and x in [3, 4] 
                else np.nan
            )
        else:
            # Normal: agree (3,4) = 1, disagree (1,2) = 0
            df_card_aggregated[col] = df_card_aggregated[col].apply(
                lambda x: 1 if pd.notna(x) and x in [3, 4] 
                else 0 if pd.notna(x) and x in [1, 2] 
                else np.nan
            )

df_card_aggregated['sqr_total'] = df_card_aggregated[sqr_cols].sum(axis=1)
print(f"  ✅ SQR-10 scored: total range {df_card_aggregated['sqr_total'].min():.0f}-{df_card_aggregated['sqr_total'].max():.0f}")

# AQ-10 Scoring (For reference only - will be excluded)
print("\nAQ-10 Scoring (for reference only - will be excluded)...")
aq_cols = [f'aq_{i}' for i in range(1, 11)]
aq_reverse_items = [2, 3, 4, 5, 6, 9]

for i in range(1, 11):
    col = f'aq_{i}'
    if col in df_card_aggregated.columns:
        df_card_aggregated[col] = pd.to_numeric(df_card_aggregated[col], errors='coerce')
        if i in aq_reverse_items:
            df_card_aggregated[col] = df_card_aggregated[col].apply(
                lambda x: 1 if pd.notna(x) and x in [1, 2] 
                else 0 if pd.notna(x) and x in [3, 4] 
                else np.nan
            )
        else:
            df_card_aggregated[col] = df_card_aggregated[col].apply(
                lambda x: 1 if pd.notna(x) and x in [3, 4] 
                else 0 if pd.notna(x) and x in [1, 2] 
                else np.nan
            )

df_card_aggregated['aq_total'] = df_card_aggregated[aq_cols].sum(axis=1)
print(f"  ✅ AQ-10 scored: total range {df_card_aggregated['aq_total'].min():.0f}-{df_card_aggregated['aq_total'].max():.0f}")
print(f"  ⚠️  NOTE: AQ features will be EXCLUDED from final feature set (data leakage prevention)")
```

### STEP 5: Create Target Variable

```python
print("\n" + "="*80)
print("CREATING TARGET VARIABLE")
print("="*80)

# Look for diagnosis columns
diagnosis_cols = [c for c in df_card_aggregated.columns if 'diagnos' in c.lower()]

if diagnosis_cols:
    # Create autism_target from diagnosis
    df_card_aggregated['autism_target'] = df_card_aggregated[diagnosis_cols[0]].astype(str).str.contains(
        'autism', case=False, na=False
    ).astype(int)
    
    print(f"\nAutism target distribution:")
    print(df_card_aggregated['autism_target'].value_counts().to_dict())
    print(f"Autism prevalence: {df_card_aggregated['autism_target'].mean()*100:.2f}%")
else:
    print("\n⚠️  WARNING: No diagnosis column found - cannot create autism_target")
    print("   Available columns:", [c for c in df_card_aggregated.columns if 'diagnos' in c.lower()])
    df_card_aggregated['autism_target'] = 0
```

### STEP 6: Demographic Feature Engineering

```python
print("\n" + "="*80)
print("DEMOGRAPHIC FEATURE ENGINEERING")
print("="*80)

# Age
age_cols = [c for c in df_card_aggregated.columns if 'age' in c.lower()]
if age_cols:
    age_col = age_cols[0]
    df_card_aggregated['age'] = pd.to_numeric(df_card_aggregated[age_col], errors='coerce')
    age_median = df_card_aggregated['age'].median()
    df_card_aggregated['age'] = df_card_aggregated['age'].fillna(age_median)
else:
    df_card_aggregated['age'] = 30  # Default
    print("⚠️  Age column not found - using default age=30")

print(f"Age: median={df_card_aggregated['age'].median():.1f}, range={df_card_aggregated['age'].min():.0f}-{df_card_aggregated['age'].max():.0f}")

# Sex
sex_mapping = {
    'male': 1, 'm': 1, '1': 1,
    'female': 2, 'f': 2, '2': 2,
    'other': 3, 'o': 3, '3': 3,
    'prefer not to say': 4, '4': 4
}

sex_cols = [c for c in df_card_aggregated.columns if 'sex' in c.lower() or 'gender' in c.lower()]
if sex_cols:
    sex_col = sex_cols[0]
    df_card_aggregated['sex'] = df_card_aggregated[sex_col].astype(str).str.strip().str.lower().map(sex_mapping).fillna(4)
else:
    df_card_aggregated['sex'] = 4  # Unknown
    print("⚠️  Sex column not found - set to unknown (4)")

df_card_aggregated['sex_num'] = df_card_aggregated['sex'].map({1: 0, 2: 1, 3: 2, 4: 3}).fillna(0).astype(int)
print(f"Sex distribution: {df_card_aggregated['sex'].value_counts().to_dict()}")

# Age groups
df_card_aggregated['age_group_19-30'] = ((df_card_aggregated['age'] >= 19) & (df_card_aggregated['age'] <= 30)).astype(int)
df_card_aggregated['age_group_31-45'] = ((df_card_aggregated['age'] >= 31) & (df_card_aggregated['age'] <= 45)).astype(int)
df_card_aggregated['age_group_46-60'] = ((df_card_aggregated['age'] >= 46) & (df_card_aggregated['age'] <= 60)).astype(int)
df_card_aggregated['age_group_61+'] = (df_card_aggregated['age'] >= 61).astype(int)

# sqrt_age
df_card_aggregated['sqrt_age'] = np.sqrt(np.clip(df_card_aggregated['age'], a_min=0, a_max=None))
print(f"✅ sqrt_age created: range {df_card_aggregated['sqrt_age'].min():.2f}-{df_card_aggregated['sqrt_age'].max():.2f}")
```

### STEP 7: Questionnaire Totals and Interactions

```python
print("\n" + "="*80)
print("QUESTIONNAIRE TOTALS AND INTERACTIONS")
print("="*80)

# d_score (SQR - EQ)
df_card_aggregated['d_score'] = df_card_aggregated['sqr_total'] - df_card_aggregated['eq_total']

# age_x_eq interaction
df_card_aggregated['age_x_eq'] = df_card_aggregated['age'] * df_card_aggregated['eq_total']

# eq_sqr_ratio
df_card_aggregated['eq_sqr_ratio'] = df_card_aggregated['eq_total'] / (df_card_aggregated['sqr_total'].replace(0, np.nan) + 1e-8)
df_card_aggregated['eq_sqr_ratio'] = df_card_aggregated['eq_sqr_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0.0)

print(f"✅ All questionnaire interactions created")
```

### STEP 8: Occupation Feature

```python
print("\n" + "="*80)
print("OCCUPATION FEATURE")
print("="*80)

occupation_cols = [c for c in df_card_aggregated.columns if 'occupation' in c.lower() or 'job' in c.lower()]
if occupation_cols:
    occupation_col = occupation_cols[0]
    df_card_aggregated['is_stem_occupation'] = df_card_aggregated[occupation_col].astype(str).str.contains(
        'science|technology|engineering|math|computer|software|data|research', 
        case=False, na=False
    ).astype(int)
    print(f"✅ is_stem_occupation created: {df_card_aggregated['is_stem_occupation'].sum()} STEM occupations")
else:
    df_card_aggregated['is_stem_occupation'] = 0
    print("⚠️  Occupation column not found - set to 0")
```

### STEP 9: Feature Alignment to C4 Schema

```python
print("\n" + "="*80)
print("FEATURE ALIGNMENT TO C4 SCHEMA")
print("="*80)

# Load C4 feature schema
with open(FEATURE_INFO_PATH, 'r') as f:
    feature_info = json.load(f)

c4_feature_names = feature_info['feature_names']  # 45 features
excluded_features = set(feature_info.get('excluded_features', []))

print(f"\nC4 expects {len(c4_feature_names)} features")
print(f"Excluded features (AQ-related): {len(excluded_features)}")

# Build aligned feature matrix
X_card = pd.DataFrame(index=df_card_aggregated.index)
missing_features = []
available_features = []

for feat_name in c4_feature_names:
    if feat_name in df_card_aggregated.columns:
        X_card[feat_name] = df_card_aggregated[feat_name]
        available_features.append(feat_name)
    else:
        # Missing feature - fill with 0
        X_card[feat_name] = 0
        missing_features.append(feat_name)

# Ensure correct order (CRITICAL)
X_card = X_card[c4_feature_names]

print(f"\n✅ Aligned feature matrix shape: {X_card.shape}")
print(f"✅ Available features: {len(available_features)}/{len(c4_feature_names)}")
print(f"⚠️  Missing features filled with 0: {len(missing_features)}")

if missing_features:
    print(f"\nMissing features (filled with 0):")
    for feat in missing_features[:10]:
        print(f"  - {feat}")
    if len(missing_features) > 10:
        print(f"  ... and {len(missing_features) - 10} more")

# Handle missing values and ensure numeric
X_card = X_card.fillna(0)
for col in X_card.columns:
    if X_card[col].dtype == 'object':
        X_card[col] = pd.to_numeric(X_card[col], errors='coerce').fillna(0)
    else:
        X_card[col] = pd.to_numeric(X_card[col], errors='coerce').fillna(0)

# Check for infinite values
inf_cols = []
for col in X_card.columns:
    if np.isinf(X_card[col]).any():
        inf_cols.append(col)
        X_card[col] = X_card[col].replace([np.inf, -np.inf], 0)

if inf_cols:
    print(f"\n⚠️  Found infinite values in: {inf_cols} (replaced with 0)")
else:
    print(f"\n✅ No infinite values found")

print(f"\n✅ Feature alignment complete - ready for scaling")
```

### STEP 10: Apply C4 Scaler (NO REFITTING)

```python
print("\n" + "="*80)
print("APPLYING C4 SCALER (NO REFITTING)")
print("="*80)

# Load scaler
scaler = joblib.load(SCALER_PATH)

# Apply scaler (fitted on C4, NOT refit on CARD)
X_card_scaled = scaler.transform(X_card.values)

print(f"\n✅ Scaled feature matrix shape: {X_card_scaled.shape}")
print(f"✅ Applied C4 scaler (fitted on C4 training data, NOT refit on CARD)")
print(f"✅ Feature order matches C4 exactly: {list(X_card.columns[:5])}...")

# Verify scaling worked
print(f"\nScaled feature statistics:")
print(f"  Mean: {X_card_scaled.mean():.4f}")
print(f"  Std: {X_card_scaled.std():.4f}")
print(f"  Min: {X_card_scaled.min():.4f}")
print(f"  Max: {X_card_scaled.max():.4f}")
```

### STEP 11: Generate Predictions from All Models

```python
print("\n" + "="*80)
print("GENERATING PREDICTIONS FROM ALL MODELS")
print("="*80)

# Load models
loaded_models = {}
for name, path in MODELS.items():
    if os.path.exists(path):
        loaded_models[name] = joblib.load(path)

# Generate predictions
predictions = {}
probabilities = {}

for name, model in loaded_models.items():
    y_proba = model.predict_proba(X_card_scaled)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    predictions[name] = y_pred
    probabilities[name] = y_proba
    print(f"\n{name}:")
    print(f"  Generated predictions for {len(y_pred)} samples")
    print(f"  Probability range: {y_proba.min():.4f} - {y_proba.max():.4f}")
    print(f"  Predicted positives: {y_pred.sum()} ({y_pred.mean()*100:.1f}%)")

print("\n✅ Predictions generated from all models")
```

### STEP 12: Evaluate Performance (if ground truth available)

```python
print("\n" + "="*80)
print("EVALUATING PERFORMANCE")
print("="*80)

if 'autism_target' in df_card_aggregated.columns:
    y_true = df_card_aggregated['autism_target'].values
    
    print(f"\nGround truth available:")
    print(f"  Total samples: {len(y_true)}")
    print(f"  Autism cases: {y_true.sum()} ({y_true.mean()*100:.2f}%)")
    print(f"  Non-autism cases: {(y_true == 0).sum()} ({(y_true == 0).mean()*100:.2f}%)")
    
    results = {}
    for name in loaded_models.keys():
        y_pred = predictions[name]
        y_proba = probabilities[name]
        
        results[name] = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_proba)
        }
    
    metrics_df = pd.DataFrame(results).T
    
    print("\n" + "="*80)
    print("EXTERNAL VALIDATION RESULTS ON CARD")
    print("="*80)
    print(metrics_df.round(4).sort_values('auc', ascending=False))
    
    # Compare with C4 performance
    print("\n" + "="*80)
    print("COMPARISON WITH C4 PERFORMANCE")
    print("="*80)
    
    c4_results_path = os.path.join(ARTIFACT_DIR, 'original_dataset_results.json')
    if os.path.exists(c4_results_path):
        with open(c4_results_path, 'r') as f:
            c4_results = json.load(f)
        
        comparison_data = []
        for model_name in metrics_df.index:
            if model_name in c4_results:
                comparison_data.append({
                    'Model': model_name,
                    'C4_F1': c4_results[model_name]['f1'],
                    'CARD_F1': metrics_df.loc[model_name, 'f1'],
                    'C4_AUC': c4_results[model_name]['auc'],
                    'CARD_AUC': metrics_df.loc[model_name, 'auc'],
                    'F1_Drop': c4_results[model_name]['f1'] - metrics_df.loc[model_name, 'f1'],
                    'AUC_Drop': c4_results[model_name]['auc'] - metrics_df.loc[model_name, 'auc']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.round(4))
    
    # Save results
    output_dir = '/Users/eb2007/playground/bullpy/c4_play2/data/processed'
    os.makedirs(output_dir, exist_ok=True)
    metrics_df.to_csv(os.path.join(output_dir, 'card_external_validation_results.csv'))
    print(f"\n✅ Results saved to: {output_dir}/card_external_validation_results.csv")
    
else:
    print("\n⚠️  No ground truth labels available - skipping evaluation")
    print("   Predictions will be saved but metrics cannot be calculated")
```

### STEP 13: Save Predictions and Metadata

```python
print("\n" + "="*80)
print("SAVING PREDICTIONS AND METADATA")
print("="*80)

output_dir = '/Users/eb2007/playground/bullpy/c4_play2/data/processed'
os.makedirs(output_dir, exist_ok=True)

# Save predictions
pred_df = pd.DataFrame({
    'userid': df_card_aggregated['userid'].values if 'userid' in df_card_aggregated.columns else np.arange(len(df_card_aggregated))
})

for name in loaded_models.keys():
    pred_df[f'proba_{name.replace(" ", "_").lower()}'] = probabilities[name]
    pred_df[f'pred_{name.replace(" ", "_").lower()}'] = predictions[name]

# Add ground truth if available
if 'autism_target' in df_card_aggregated.columns:
    pred_df['autism_target'] = df_card_aggregated['autism_target'].values

pred_path = os.path.join(output_dir, 'card_external_predictions.csv')
pred_df.to_csv(pred_path, index=False)
print(f"\n✅ Predictions saved to: {pred_path}")

# Save feature alignment info
alignment_info = {
    'c4_feature_count': len(c4_feature_names),
    'card_available_features': len(available_features),
    'card_missing_features': len(missing_features),
    'missing_features': missing_features,
    'available_features': available_features,
    'excluded_features': list(excluded_features),
    'note': 'CARD dataset has SPQ data (unlike YBT). AQ features excluded to prevent data leakage.'
}

alignment_path = os.path.join(output_dir, 'card_feature_alignment.json')
with open(alignment_path, 'w') as f:
    json.dump(alignment_info, f, indent=2)
print(f"✅ Feature alignment info saved to: {alignment_path}")

print("\n" + "="*80)
print("EXTERNAL VALIDATION COMPLETE")
print("="*80)
print(f"\nSummary:")
print(f"  Dataset: CARD")
print(f"  Participants: {len(df_card_aggregated)}")
print(f"  Features: {X_card_scaled.shape[1]} (aligned to C4)")
print(f"  Models tested: {len(loaded_models)}")
if 'autism_target' in df_card_aggregated.columns:
    best_model = metrics_df['auc'].idxmax()
    print(f"  Best model: {best_model} (AUC: {metrics_df.loc[best_model, 'auc']:.4f})")
print(f"\n✅ All outputs saved to: {output_dir}")
```

## Key Differences from YBT Preprocessing

1. **CARD has SPQ data** - Unlike YBT, CARD includes SPQ questionnaires
2. **Multiple questionnaires per participant** - Must aggregate before preprocessing
3. **Itemized scores in CSV format** - Must parse CSV strings to extract individual items
4. **Test name column** - Need to identify which questionnaire each row represents
5. **Priority system** - Prefer adult versions over child/adolescent versions

## Expected Outputs

1. **Preprocessed CARD dataset** with 45 features matching C4
2. **External validation metrics** (if ground truth available)
3. **Predictions** from all 5 original C4 models (with SPQ)
4. **Feature alignment report** showing which features were available/missing

## Notes

- **SPQ is available in CARD** - This is a key advantage over YBT
- **Original C4 models (with SPQ) are used** - These are preserved for CARD validation
- **AQ features excluded** - To prevent data leakage
- **File may be password-protected** - Use provided password if needed
