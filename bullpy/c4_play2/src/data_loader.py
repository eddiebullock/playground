#src/data_loader.py

import pandas as pd

def load_data(file_path):
    """load csv file into pandas dataframe"""
    return pd.read_csv(filepath)

if __name__ == "__main__":
    df = load_data("/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/data_c4_raw.csv")
    print(df.info())
    print(df.head())

    ### small test
    