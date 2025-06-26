import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Create a simple test DataFrame to debug the scaling
print("Creating test data...")
test_data = pd.DataFrame({
    'age': [25, 30, 35, 40],
    'aq_1': [2, 3, 1, 4],
    'sex_Male': [0, 1, 0, 1],
    'country_USA': [1, 0, 1, 0]
})

print("Test data:")
print(test_data)
print("\nData types:")
print(test_data.dtypes)

print("\nTesting numeric column selection...")
numeric_columns = test_data.select_dtypes(include=['number']).columns
print("Numeric columns:", list(numeric_columns))

print("\nTesting scaling...")
try:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(test_data[numeric_columns])
    print("Scaling successful!")
    print("Scaled data shape:", scaled_data.shape)
except Exception as e:
    print("Error during scaling:", str(e))
    print("Error type:", type(e).__name__) 