import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

print("=== ENHANCED FEATURE ENGINEERING ===")
print("Loading original dataset...")

# Load matched data 
df = pd.read_csv('data/processed/data_c4_matched_balanced.csv')
print(f"Original dataset shape: {df.shape}")
print(f"Original features: {len(df.columns)}")

# 1. BASIC FEATURE ENGINEERING (existing)
print("\n1. Creating basic features...")

# age group bins
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 30, 45, 60, 100], labels=['0-18', '19-30', '31-45', '46-60', '61+'])

# non linear transformation 
df['log_aq_total'] = np.log1p(df['aq_total'])
df['sqrt_age'] = np.sqrt(df['age'])

# interaction terms
df['aq_eq_interaction'] = df['aq_total'] * df['eq_total']
df['sqp_aq_interaction'] = df['spq_total'] * df['aq_total']
df['age_x_eq'] = df['age'] * df['eq_total']

# questionnaire score ratios 
df['aq_spq_ratio'] = df['aq_total'] / (df['spq_total'] + 1e-8)
df['eq_sqr_ratio'] = df['eq_total'] / (df['sqr_total'] + 1e-8)

# boolean: high aq (above 1 std)
df['high_aq'] = (df['aq_total'] > df['aq_total'].mean() + df['aq_total'].std()).astype(int)

print(f"After basic features: {len(df.columns)} features")

# 2. SCIENTIFIC LITERATURE FEATURES (new)
print("\n2. Adding scientific literature features...")

# AQ subdomains (Baron-Cohen et al., 2001)
df['aq_social_skills'] = df[['aq_1', 'aq_7', 'aq_8', 'aq_9', 'aq_10']].sum(axis=1)
df['aq_attention_switching'] = df[['aq_2', 'aq_4', 'aq_6']].sum(axis=1)
df['aq_attention_detail'] = df[['aq_3', 'aq_5']].sum(axis=1)

# EQ subdomains (Baron-Cohen & Wheelwright, 2004)
df['eq_cognitive'] = df[['eq_1', 'eq_3', 'eq_5', 'eq_7', 'eq_9']].sum(axis=1)
df['eq_affective'] = df[['eq_2', 'eq_4', 'eq_6', 'eq_8', 'eq_10']].sum(axis=1)

# SQR subdomains (Constantino & Gruber, 2005)
df['sqr_social_awareness'] = df[['sqr_1', 'sqr_2', 'sqr_3']].sum(axis=1)
df['sqr_social_cognition'] = df[['sqr_4', 'sqr_5', 'sqr_6']].sum(axis=1)
df['sqr_social_communication'] = df[['sqr_7', 'sqr_8', 'sqr_9']].sum(axis=1)
df['sqr_social_motivation'] = df[['sqr_10']].sum(axis=1)

# SPQ subdomains (Raine, 1991)
df['spq_cognitive_perceptual'] = df[['spq_1', 'spq_2', 'spq_3', 'spq_4']].sum(axis=1)
df['spq_interpersonal'] = df[['spq_5', 'spq_6', 'spq_7', 'spq_8']].sum(axis=1)
df['spq_disorganized'] = df[['spq_9', 'spq_10']].sum(axis=1)

print(f"After scientific features: {len(df.columns)} features")

# 3. CLINICAL THRESHOLDS (new)
print("\n3. Adding clinical thresholds...")

# Autism screening thresholds (based on literature)
df['aq_above_threshold'] = (df['aq_total'] > 26).astype(int)  # Baron-Cohen et al., 2001
df['aq_high_threshold'] = (df['aq_total'] > 32).astype(int)   # Clinical threshold

# Empathy deficits
df['eq_below_threshold'] = (df['eq_total'] < 30).astype(int)  # Baron-Cohen & Wheelwright, 2004

# Social responsiveness deficits
df['sqr_above_threshold'] = (df['sqr_total'] > 60).astype(int)  # Constantino & Gruber, 2005

print(f"After clinical thresholds: {len(df.columns)} features")

# 4. STATISTICAL FEATURES (new)
print("\n4. Adding statistical features...")

# Z-scores for questionnaire totals
for col in ['aq_total', 'eq_total', 'sqr_total', 'spq_total']:
    if col in df.columns:
        df[f'{col}_zscore'] = (df[col] - df[col].mean()) / df[col].std()

# Percentile ranks
for col in ['aq_total', 'eq_total', 'sqr_total', 'spq_total']:
    if col in df.columns:
        df[f'{col}_percentile'] = df[col].rank(pct=True)

# Extreme value indicators
for col in ['aq_total', 'eq_total', 'sqr_total', 'spq_total']:
    if col in df.columns:
        df[f'{col}_extreme_high'] = (df[col] > df[col].quantile(0.95)).astype(int)
        df[f'{col}_extreme_low'] = (df[col] < df[col].quantile(0.05)).astype(int)

# Quadratic terms
for col in ['aq_total', 'eq_total', 'sqr_total']:
    if col in df.columns:
        df[f'{col}_squared'] = df[col] ** 2

print(f"After statistical features: {len(df.columns)} features")

# 5. ADDITIONAL INTERACTIONS (new)
print("\n5. Adding additional interactions...")

# Three-way interactions
if all(col in df.columns for col in ['aq_total', 'eq_total', 'sqr_total']):
    df['aq_eq_sqr_interaction'] = df['aq_total'] * df['eq_total'] * df['sqr_total']

# Cross-ratio features
if all(col in df.columns for col in ['aq_total', 'eq_total', 'sqr_total']):
    df['aq_eq_cross_ratio'] = df['aq_total'] / (df['eq_total'] + 1e-8)
    df['aq_sqr_cross_ratio'] = df['aq_total'] / (df['sqr_total'] + 1e-8)
    df['eq_sqr_cross_ratio'] = df['eq_total'] / (df['sqr_total'] + 1e-8)

# Sex interactions (if available)
if 'sex_num' in df.columns:
    df['sex_x_aq'] = df['sex_num'] * df['aq_total']
    df['sex_x_eq'] = df['sex_num'] * df['eq_total']
    df['sex_x_sqr'] = df['sex_num'] * df['sqr_total']

print(f"After additional interactions: {len(df.columns)} features")

# 6. FEATURE SELECTION (existing logic)
print("\n6. Applying feature selection...")

# remove highly correlated features 
numeric_cols = df.drop(columns=['autism_target']).select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
df = df.drop(columns=to_drop)
print(f"Dropped {len(to_drop)} highly correlated features")

# drop low variance features 
feature_cols = df.drop(columns=['autism_target']).select_dtypes(include=[np.number]).columns
selector = VarianceThreshold(threshold=0.1)
selector.fit(df[feature_cols])
low_variance_cols = feature_cols[~selector.get_support()]
df = df.drop(columns=low_variance_cols)
print(f"Dropped {len(low_variance_cols)} low variance features")

# 7. ONE-HOT ENCODING
print("\n7. One-hot encoding categorical features...")
df = pd.get_dummies(df, columns=['age_group'], drop_first=True)

# 8. SAVE ENHANCED DATASET
print("\n8. Saving enhanced dataset...")
df.to_csv('data/processed/data_c4_enhanced_fe.csv', index=False)

print(f"\n=== FEATURE ENGINEERING COMPLETE ===")
print(f"Final dataset shape: {df.shape}")
print(f"Total features: {len(df.columns)}")
print(f"Features created: {len(df.columns) - 55}")  # 55 was original count
print(f"Target distribution: {df['autism_target'].value_counts()}")

# Show new features
new_features = [col for col in df.columns if col not in ['autism_target'] and any(term in col for term in ['social', 'cognitive', 'affective', 'threshold', 'zscore', 'percentile', 'extreme', 'squared', 'cross'])]
print(f"\nNew features created: {len(new_features)}")
print("Sample new features:", new_features[:10]) 