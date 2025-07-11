"""
Data Analysis Practice Project
- Implements data cleaning, normalization, statistical testing, and basic visualization
- All from scratch without relying on specialized libraries
"""

import random
import math
import csv
from collections import defaultdict  # Fixed import

def generate_messy_dataset(size=1000):
    """
    Generate a messy dataset with missing values, outliers, and varying data types
    """
    data = []
    for i in range(size):
        if random.random() < 0.1:  # 10% missing values
            age = None
        else:
            age = random.randint(18, 80)

        # Add some outliers
        if random.random() < 0.05:  # 5% outliers
            age = random.randint(100, 120)

        # generate income with some correlation to age
        if age is None:
            income = None
        else:
            base_income = 30000 + (age * 2000)
            noise = random.uniform(0.7, 1.3)
            income = base_income * noise
            # some missing values
            if random.random() < 0.15:
                income = None

        # generate categorical data
        categories = ["cat_A", "cat_B", "cat_C", "cat_D"]
        if random.random() < 0.1:
            category = None
        else:
            category = random.choice(categories)

        # add some formatting inconsistencies
        if category is not None:  # Added null check
            if random.random() < 0.2:
                category = category.upper()
            elif random.random() < 0.2:
                category = category.lower()

        data.append({"age": age, "income": income, "category": category})

    return data

def impute_missing_values(data, strategy="mean"):
    """
    Impute missing values based on a specified strategy
    """
    # first calculate statistics
    stats = {}
    for col in data[0].keys():
        values = [row[col] for row in data if row[col] is not None]

        if all(isinstance(v, (int, float)) for v in values):
            stats[col] = {
                "mean": sum(values) / len(values),
                "median": sorted(values)[len(values) // 2],
                "min": min(values),
                "max": max(values),
            }
        elif all(isinstance(v, str) for v in values):
            # for categorical data, get the most frequent value
            freq = defaultdict(int)
            for v in values:
                freq[v] += 1
            most_common = max(freq.items(), key=lambda x: x[1])[0]
            stats[col] = {"most_frequent": most_common}

    # Fixed indentation: This section should be outside the column loop
    # impute data
    cleaned_data = []
    for row in data:
        new_row = {}
        for col, val in row.items():
            if val is None:
                if col in stats:
                    if "most_frequent" in stats[col]:
                        new_row[col] = stats[col]["most_frequent"]
                    elif strategy == "mean":
                        new_row[col] = stats[col]["mean"]
                    elif strategy == "median":
                        new_row[col] = stats[col]["median"]
                    else:
                        new_row[col] = None
                else:
                    new_row[col] = None
            else:
                new_row[col] = val
        cleaned_data.append(new_row)  # Fixed: append outside inner loop

    return cleaned_data  # Added return statement

def detect_outliers_iqr(data, column):
    """
    Detect outliers in a dataset based on the IQR method
    """
    # Get values and sort them for quartile calculation
    values = sorted([row[column] for row in data if row[column] is not None])
    q1_pos = int(0.25 * len(values))
    q3_pos = int(0.75 * len(values))

    q1 = values[q1_pos]
    q3 = values[q3_pos]
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = []
    for i, row in enumerate(data):
        val = row[column]
        if val is not None and (val < lower_bound or val > upper_bound):
            outliers.append((i, val))  # Store both index and value for better debugging

    return outliers, (lower_bound, upper_bound)

def normalize_min_max(data, column):
    """Apply min-max normalization to a column"""
    values = [row[column] for row in data if row[column] is not None]
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val

    normalized_data = []
    for row in data:
        new_row = row.copy()
        if row[column] is not None:
            new_row[column] = (row[column] - min_val) / range_val
        normalized_data.append(new_row)

    return normalized_data

def standardize_data(data, column):
    """Apply z-score standardization to a column"""
    values = [row[column] for row in data if row[column] is not None]
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std_dev = math.sqrt(variance)

    standardized_data = []
    for row in data:
        new_row = row.copy()
        if row[column] is not None:
            new_row[column] = (row[column] - mean) / std_dev
        standardized_data.append(new_row)

    return standardized_data

def standardize_columns(data, columns):
    """Apply z-score standardization to multiple columns"""
    result = data.copy()
    for col in columns:
        result = standardize_data(result, col)
    return result

# Additional useful functions

def detect_missing_values(data, columns):
    """Count missing values in each column"""
    missing_counts = {col: 0 for col in columns}
    for row in data:
        for col in columns:
            if row[col] is None:
                missing_counts[col] += 1
    
    return missing_counts

def save_to_csv(data, filename):
    """Save data to a CSV file"""
    if not data:
        return "No data to save"
        
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    
    return f"Data saved to {filename}"

def load_from_csv(filename):
    """Load data from a CSV file"""
    data = []
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Convert string values to appropriate types
            processed_row = {}
            for key, value in row.items():
                if value == '':
                    processed_row[key] = None
                elif value.lower() == 'none':
                    processed_row[key] = None
                elif value.replace('.', '', 1).isdigit():
                    # Convert to float if it's a number
                    processed_row[key] = float(value)
                    # Convert to int if it's a whole number
                    if processed_row[key].is_integer():
                        processed_row[key] = int(processed_row[key])
                else:
                    processed_row[key] = value
            data.append(processed_row)
    
    return data