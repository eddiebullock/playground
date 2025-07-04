import pandas as pd
import numpy as np

# Path to the dataset
DATA_PATH = '/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/data_c4_clean.csv'

# Data key mappings
sex_map = {1: 'Male', 2: 'Female', 3: 'Transgender/Other', 4: 'Prefer not to say'}
handedness_map = {1: 'Right-handed', 2: 'Left-handed', 3: 'Ambidextrous', 4: 'Prefer not to say'}
education_map = {
    1: 'Did not complete High School (or A-levels)',
    2: 'High School (or A-levels) Diploma',
    3: 'Undergraduate degree',
    4: 'Postgraduate degree',
    5: 'Prefer not to say'
}
occupation_map = {
    1: 'Artist', 2: 'Civil Engineering', 3: 'Computers & I.T.', 4: 'Director', 5: 'Engineering',
    6: 'Entrepreneur', 7: 'Financial Banking', 8: 'Food & Drinks', 9: 'Healthcare', 10: 'Hospitality',
    11: 'Legal', 12: 'Leisure', 13: 'Musician', 14: 'Office Administration', 15: 'Other',
    16: 'Public Sector', 17: 'Services', 18: 'Publishing & Media', 19: 'Retail', 20: 'Sales',
    21: 'Scientific & Technical', 22: 'Supply chain', 23: 'Teaching & Interpretation', 24: 'Transport',
    25: 'Other', 26: 'Prefer not to say'
}
region_map = {
    1: 'Wales', 2: 'Scotland', 3: 'Northern Ireland', 4: 'London (England)', 5: 'North East (England)',
    6: 'North West (England)', 7: 'Yorkshire and Humber (England)', 8: 'West Midlands (England)',
    9: 'East Midlands (England)', 10: 'South East (England)', 11: 'South West (England)',
    12: 'Other (outside of the United Kingdom)', 13: 'Other (in the United Kingdom)', 14: 'Prefer not to say'
}
diagnosis_map = {
    1: 'Attention Deficit / Hyperactivity Disorder',
    2: 'Autism Spectrum Disorder',
    3: 'Bipolar Disorder',
    4: 'Depression',
    5: 'Learning disability',
    6: 'Obsessive-Compulsive Disorder',
    7: 'Schizophrenia',
    8: 'I prefer not to say',
    9: 'I have not been diagnosed with any of these conditions'
}
asd_map = {1: 'Autism (classical autism)', 2: 'Asperger Syndrome (AS)', 3: 'Other'}

# Load data
df = pd.read_csv(DATA_PATH)
print('Initial shape:', df.shape)

# Convert relevant columns to numeric (if not already)
for col in ['sex', 'handedness', 'education', 'occupation', 'countryregion']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Merge occupation 15 and 25 as 'Other'
df['occupation'] = df['occupation'].replace({25: 15})

# Map codes to strings
df['sex_str'] = df['sex'].map(sex_map)
df['handedness_str'] = df['handedness'].map(handedness_map)
df['education_str'] = df['education'].map(education_map)
df['occupation_str'] = df['occupation'].map(occupation_map)
df['region_str'] = df['countryregion'].map(region_map)

# Diagnosis columns
diagnosis_cols = [f'diagnosis_{i}' for i in range(0, 9)]
# Clean diagnosis columns: replace '#NULL!' with np.nan and convert to numeric
for col in diagnosis_cols:
    df[col] = df[col].replace('#NULL!', np.nan)
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Step 5: Check for over-filtering (print shape after all cleaning)
print('Shape after cleaning:', df.shape)

# Step 6: Check for duplicates (by userid if present)
if 'userid' in df.columns:
    n_unique_users = df['userid'].nunique()
    print(f'Unique userids: {n_unique_users}, Total rows: {len(df)}')
    dupes = df['userid'].duplicated().sum()
    print(f'Number of duplicated userids: {dupes}')
else:
    print('No userid column found for duplicate check.')

# Step 7: Check for repeat testers (if repeat column exists)
if 'repeat' in df.columns:
    print('Repeat tester value counts:')
    print(df['repeat'].value_counts(dropna=False))
    # Optionally, filter out repeat testers and print new shape
    df_no_repeat = df[df['repeat'] != 1]
    print('Shape after removing repeat testers:', df_no_repeat.shape)
else:
    print('No repeat column found.')

# Step 8: Print a sample of rows where autism should be detected
print('\nSample of rows with autism diagnosis (any diagnosis column == 2):')
autism_rows = df[df[diagnosis_cols].eq(2).any(axis=1)]
print(autism_rows[diagnosis_cols + ['userid']].head(20))
print(f'Total rows with autism diagnosis: {len(autism_rows)}')

# Print the full list of userids for autistic individuals
if 'userid' in autism_rows.columns:
    print('\nFull list of userids for autistic individuals:')
    print(autism_rows['userid'].to_list())
    # Save to file
    with open('autistic_userids.txt', 'w') as f:
        for uid in autism_rows['userid']:
            f.write(f"{int(uid)}\n")

# Step 9: Check for data subsetting (print total rows and unique userids)
print(f'Total rows in dataset: {len(df)}')
if 'userid' in df.columns:
    print(f'Total unique userids: {df["userid"].nunique()}')

# Step 10: Summary table for debugging
print('\n--- Debug Summary ---')
print('Initial shape:', df.shape)
if 'repeat' in df.columns:
    print('Shape after removing repeat testers:', df_no_repeat.shape)
print('Autism count (any diagnosis column == 2):', df[diagnosis_cols].eq(2).any(axis=1).sum())

# Print updated diagnosis value counts for verification
for col in diagnosis_cols:
    print(f"\nValue counts for {col} after cleaning:")
    print(df[col].value_counts(dropna=False))

# Diagnosis summaries
print('\n--- Diagnosis Counts (total selections, not unique people) ---')
diagnosis_counts = pd.Series(dtype=int)
for code, label in diagnosis_map.items():
    count = (df[diagnosis_cols] == code).sum().sum()
    diagnosis_counts[label] = count
print(diagnosis_counts)

print('\n--- Diagnosis Counts (unique people) ---')
diagnosis_unique_counts = pd.Series(dtype=int)
for code, label in diagnosis_map.items():
    diagnosis_unique_counts[label] = df[diagnosis_cols].eq(code).any(axis=1).sum()
print(diagnosis_unique_counts)

# ASD subtypes (columns R to T, i.e., autism_diagnosis_0/1/2)
asd_cols = [f'autism_diagnosis_{i}' for i in range(0, 3)]
for col in asd_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

asd_counts = pd.Series(dtype=int)
for code, label in asd_map.items():
    asd_counts[label] = (df[asd_cols] == code).sum().sum()
print('\n--- ASD Subtype Counts (total selections) ---')
print(asd_counts)

# ASD subtypes (unique people)
asd_unique_counts = pd.Series(dtype=int)
for code, label in asd_map.items():
    asd_unique_counts[label] = df[asd_cols].eq(code).any(axis=1).sum()
print('\n--- ASD Subtype Counts (unique people) ---')
print(asd_unique_counts)

df['is_autistic'] = df[diagnosis_cols].eq(2).any(axis=1)
print('Number of autistic individuals:', df['is_autistic'].sum()) 