import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the data (adjust path as needed)
print("Loading data...")
try:
    df = pd.read_csv('/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/YBT.csv')
    print("Data loaded successfully!")
    print("Data shape:", df.shape)
except Exception as e:
    print("Error loading data:", str(e))
    exit()

# Create target column
print("\nCreating target column...")
if 'autism_diagnosis' not in df.columns:
    df['autism_diagnosis'] = (
        df['diagnosis']
        .fillna('')
        .str.lower()
        .str.contains('autism')
        .astype(int)
    )

# Define columns
aq_cols = [f'aq_{i}' for i in range(1, 11)]
sq_cols = [f'sq10_{i}' for i in range(1, 11)]
eq_cols = [f'eq10_{i}' for i in range(1,11)]
demo_cols = ['age', 'sex','gender', 'ethn', 'hand', 'country']
target_col = 'autism_diagnosis'
all_cols = aq_cols + sq_cols + eq_cols + demo_cols + [target_col]

print("Checking which columns exist...")
existing_cols = [col for col in all_cols if col in df.columns]
missing_cols = [col for col in all_cols if col not in df.columns]
print(f"Existing columns: {len(existing_cols)}")
print(f"Missing columns: {len(missing_cols)}")
if missing_cols:
    print("Missing:", missing_cols[:5], "...")

# Clean data
print("\nCleaning data...")
df_clean = df.dropna(subset=existing_cols)
print("After cleaning, shape:", df_clean.shape)

# Encode categorical variables
print("\nEncoding categorical variables...")
categorical_cols = ['sex', 'gender', 'ethn', 'hand', 'country']
existing_cat_cols = [col for col in categorical_cols if col in df_clean.columns]
print("Categorical columns to encode:", existing_cat_cols)

try:
    df_clean = pd.get_dummies(df_clean, columns=existing_cat_cols, drop_first=True)
    print("Encoding successful!")
except Exception as e:
    print("Error during encoding:", str(e))

# Prepare X and y
print("\nPreparing X and y...")
feature_cols = [col for col in df_clean.columns if col != 'autism_diagnosis']
X = df_clean[feature_cols]
y = df_clean['autism_diagnosis']
print("X shape:", X.shape)
print("y shape:", y.shape)

# Check data types
print("\nX data types:")
print(X.dtypes.value_counts())

# Test numeric column selection
print("\nTesting numeric column selection...")
try:
    numeric_columns = X.select_dtypes(include=['number']).columns
    print("Numeric columns found:", len(numeric_columns))
    print("First 5 numeric columns:", list(numeric_columns[:5]))
except Exception as e:
    print("Error selecting numeric columns:", str(e))

# Test scaling
print("\nTesting scaling...")
try:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[numeric_columns])
    print("Scaling successful!")
    print("Scaled data shape:", X_scaled.shape)
except Exception as e:
    print("Error during scaling:", str(e))
    print("Error type:", type(e).__name__) 