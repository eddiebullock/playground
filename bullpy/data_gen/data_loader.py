import pandas as pd
import numpy as np

def load_data(file_path):
    """load data from a csv file"""
    df = pd.read_csv(file_path)

    print(f"dataset shape: {df.shape}")
    print(f"number of columns: {df.shape[1]}")
    print(f"column names: {df.columns.tolist()}")
    print(df.nunique())

    return df

def calculate_questionnaire_totals(df):
    """
    calculate total scores for columns
    """
    aq_cols = [col for col in df.columns if col.startswith("aq_")]
    sq_cols = [col for col in df.columns if col.startswith("sqr_")]
    eq_cols = [col for col in df.columns if col.startswith("eq_")]
    spq_cols = [col for col in df.columns if col.startswith("spq_")]

    #calculate totals 
    df["aq_total"] = df[aq_cols].sum(axis=1)
    df["sq_total"] = df[sq_cols].sum(axis=1)
    df["eq_total"] = df[eq_cols].sum(axis=1)
    df["spq_total"] = df[spq_cols].sum(axis=1)

    return df

def create_autism_target(df):
    """
    create binary autism diagnosis target from diagnosis column 
    """
    #find autism diagnosis columns 
    autism_cols = [col for col in df.columns if col.startswith("autism_diagnosis_")]

    #find general diagnosis column 
    diagnosis_cols = [col for col in df.columns if col.startswith("diagnosis_")]

    #create autism target from autism_diagnosis columns 
    autism_from_autism_cols = df[autism_cols].sum(axis=1)
    autism_from_autism_cols = (autism_from_autism_cols > 0).astype(int)

    #create autism target from diagnosis columns (where value = 2)
    autism_from_diagnosis_cols = (df[diagnosis_cols] == 2).any(axis=1).astype(int)

    #combine both conditions with OR logic
    df['autism_diagnosis'] = ((autism_from_autism_cols == 1) | (autism_from_diagnosis_cols == 1)).astype(int)

    print(f"autism cases: {df['autism_diagnosis'].sum()}")
    print(f"non-autism cases: {len(df) - df['autism_diagnosis'].sum()}")
    print(f"autism prevalence: {df['autism_diagnosis'].mean():.3f}")

    #print breakdown for debugging 
    print(f"breakdown:")
    print(f" autism from autism_diagnosis columns: {autism_from_autism_cols.sum()}")
    print(f" autism from diagnosis columns (value = 2): {autism_from_diagnosis_cols.sum()}")
    print(f" total autism cases: {df['autism_diagnosis'].sum()}")

    #some debuggin code to see what actual value is 
    print("\nSample of diagnosis values")
    print(df[diagnosis_cols].head())

    print("\nsample of autism diagnosis values")
    print(df[autism_cols].head())