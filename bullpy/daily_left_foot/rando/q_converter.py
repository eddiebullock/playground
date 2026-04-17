#converting full AQ, EQ, SQ into short 10-item versions 

# import packages
import pandas as pd
import numpy as np

# load dataset
df_card = pd.read_csv("/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/CARD_Nov(Sheet1).csv") # replace with your path

# detect itemised score column 
itemised_col = find_col(df_card, ['Itemised Score'])
if itemised_col is None:
    raise ValueError("Could not find itemised score column in dataset")

print(f"Found itemised score column: {itemised_col}")

# detect questionnaire columns
questionnaire_cols = find_col(df_card, ['Test Name']),
if not questionnaire_cols:
    raise ValueError("Could not find questionnaire columns in dataset")

# drop adolescent rows 


# define question mappings 
# AQ-50 items 5,20,27,28,31,32,36,37,41,45 (1-indexed)
# Converted to 0-indexed positions below.
ITEM_MAPPINGS = {
    'aq':  {'full_length': 50,  'short_items': [4, 19, 26, 27, 30, 31, 35, 36, 40, 44]},
    'eq':  {'full_length': 60,  'short_items': [13, 3, 8, 30, 27, 34, 11, 21, 17, 33]},
    'sqr': {'full_length': 75,  'short_items': [31, 15, 26, 8, 29, 32, 11, 24, 7, 6]},
    'spq': {'full_length': 92,  'short_items': [1, 20, 31, 34, 37, 57, 61, 72, 73, 87]},
}


