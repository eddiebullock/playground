#!/usr/bin/env python3
"""
Compute demographic statistics from balanced datasets used for modeling.

This script loads the balanced datasets (50/50) as used in Study 1 and extracts:
- Age statistics (mean, SD, median, range)
- Sex distributions
- Questionnaire score means (AQ-10, EQ-10, SQ-R-10, SPQ-10)
- Comorbidity counts (ADHD, anxiety, depression)
"""

import os
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from study_utils import (
    load_cohort_c4,
    load_cohort_card,
    load_cohort_ybt,
    REPO_ROOT,
    RANDOM_STATE,
    COMOBIDITY_COLUMNS,
)

def format_statistics(df: pd.DataFrame, cohort_name: str, target_col: str = "diagnosis") -> dict:
    """Compute and format demographic statistics for a cohort."""
    stats = {"Cohort": cohort_name}
    
    # Sample sizes
    n_total = len(df)
    n_autism = (df[target_col] == 1).sum()
    n_non_autism = (df[target_col] == 0).sum()
    stats["N_total"] = n_total
    stats["N_autism"] = n_autism
    stats["N_non_autism"] = n_non_autism
    
    # Age statistics
    if "age" in df.columns:
        age_all = df["age"]
        age_autism = df[df[target_col] == 1]["age"]
        age_non_autism = df[df[target_col] == 0]["age"]
        
        stats["Age_mean_all"] = f"{age_all.mean():.2f}"
        stats["Age_SD_all"] = f"{age_all.std():.2f}"
        stats["Age_median_all"] = f"{age_all.median():.1f}"
        stats["Age_range_all"] = f"{age_all.min():.0f}-{age_all.max():.0f}"
        
        stats["Age_mean_autism"] = f"{age_autism.mean():.2f}"
        stats["Age_SD_autism"] = f"{age_autism.std():.2f}"
        stats["Age_mean_non_autism"] = f"{age_non_autism.mean():.2f}"
        stats["Age_SD_non_autism"] = f"{age_non_autism.std():.2f}"
    
    # Sex distribution
    if "sex_num" in df.columns:
        sex_all = df["sex_num"]
        sex_autism = df[df[target_col] == 1]["sex_num"]
        sex_non_autism = df[df[target_col] == 0]["sex_num"]
        
        # sex_num: 0=male, 1=female
        male_all = (sex_all == 0).sum()
        female_all = (sex_all == 1).sum()
        male_autism = (sex_autism == 0).sum()
        female_autism = (sex_autism == 1).sum()
        male_non_autism = (sex_non_autism == 0).sum()
        female_non_autism = (sex_non_autism == 1).sum()
        
        stats["Sex_male_all"] = f"{male_all} ({male_all/n_total*100:.2f}%)"
        stats["Sex_female_all"] = f"{female_all} ({female_all/n_total*100:.2f}%)"
        stats["Sex_male_autism"] = f"{male_autism} ({male_autism/n_autism*100:.2f}%)" if n_autism > 0 else "0"
        stats["Sex_female_autism"] = f"{female_autism} ({female_autism/n_autism*100:.2f}%)" if n_autism > 0 else "0"
        stats["Sex_male_non_autism"] = f"{male_non_autism} ({male_non_autism/n_non_autism*100:.2f}%)" if n_non_autism > 0 else "0"
        stats["Sex_female_non_autism"] = f"{female_non_autism} ({female_non_autism/n_non_autism*100:.2f}%)" if n_non_autism > 0 else "0"
    
    # Questionnaire totals
    for q_name, q_col in [("AQ", "aq_total"), ("EQ", "eq_total"), ("SQ-R", "sqr_total"), ("SPQ", "spq_total")]:
        if q_col in df.columns:
            q_all = df[q_col]
            q_autism = df[df[target_col] == 1][q_col]
            q_non_autism = df[df[target_col] == 0][q_col]
            
            stats[f"{q_name}_mean_all"] = f"{q_all.mean():.2f}"
            stats[f"{q_name}_SD_all"] = f"{q_all.std():.2f}"
            stats[f"{q_name}_mean_autism"] = f"{q_autism.mean():.2f}"
            stats[f"{q_name}_SD_autism"] = f"{q_autism.std():.2f}"
            stats[f"{q_name}_mean_non_autism"] = f"{q_non_autism.mean():.2f}"
            stats[f"{q_name}_SD_non_autism"] = f"{q_non_autism.std():.2f}"
    
    # Comorbidities
    for comorb in ["has_adhd", "has_anxiety", "has_depression"]:
        if comorb in df.columns:
            comorb_all = df[comorb].sum()
            comorb_autism = df[df[target_col] == 1][comorb].sum()
            comorb_non_autism = df[df[target_col] == 0][comorb].sum()
            
            comorb_name = comorb.replace("has_", "").title()
            stats[f"{comorb_name}_all"] = f"{comorb_all} ({comorb_all/n_total*100:.2f}%)"
            stats[f"{comorb_name}_autism"] = f"{comorb_autism} ({comorb_autism/n_autism*100:.2f}%)" if n_autism > 0 else "0"
            stats[f"{comorb_name}_non_autism"] = f"{comorb_non_autism} ({comorb_non_autism/n_non_autism*100:.2f}%)" if n_non_autism > 0 else "0"
    
    return stats


def load_ybt_with_aq(data_path: str, age_min: int = 18, age_max: int = 55) -> pd.DataFrame:
    """
    Load raw YBT, apply same preprocessing as study_utils (age, AQ/EQ/SQ-R scoring,
    AQ>=6 filter, 50/50 balance) but retain aq_total and columns needed for demographics.
    """
    df = pd.read_csv(data_path, low_memory=False)
    if "diagnosis" in df.columns and df["diagnosis"].dtype == object:
        diag_str = df["diagnosis"].astype(str)
        df["has_adhd"] = diag_str.str.contains("ADHD", case=False, na=False).astype(int)
        df["has_anxiety"] = diag_str.str.contains("Anxiety", case=False, na=False).astype(int)
        df["has_depression"] = diag_str.str.contains("Depression", case=False, na=False).astype(int)
    if "autism_target" in df.columns:
        df = df.rename(columns={"autism_target": "diagnosis"})
    elif "diagnosis_yes_no" in df.columns and "diagnosis" in df.columns:
        autism_keywords = ["autism", "asd", "asperger", "autistic"]
        df["diagnosis"] = df["diagnosis"].astype(str).str.contains(
            "|".join(autism_keywords), case=False, na=False
        ).astype(int)
    eq_map = {f"eq10_{i}": f"eq_{i}" for i in range(1, 11)}
    sq_map = {f"sq10_{i}": f"sqr_{i}" for i in range(1, 11)}
    df = df.rename(columns={**eq_map, **sq_map})
    for i in range(1, 11):
        if f"eq_{i}" not in df.columns and f"eq10_{i}" in df.columns:
            df[f"eq_{i}"] = df[f"eq10_{i}"]
        if f"sqr_{i}" not in df.columns and f"sq10_{i}" in df.columns:
            df[f"sqr_{i}"] = df[f"sq10_{i}"]
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df[df["age"].notna() & (df["age"] >= age_min) & (df["age"] <= age_max)]
    _ybt_map = {"strongly agree": 4, "slightly agree": 3, "slightly disagree": 2, "strongly disagree": 1}

    def _ybt_to_14(ser):
        out = ser.astype(str).str.strip().str.lower().replace(_ybt_map)
        return pd.to_numeric(out, errors="coerce")

    for i in range(1, 11):
        for col, reverse in [
            (f"eq_{i}", i == 3),
            (f"sqr_{i}", i in (2, 4, 6, 8, 10)),
            (f"aq_{i}", i in (2, 3, 4, 5, 6, 9)),
        ]:
            if col not in df.columns:
                continue
            if df[col].dtype == object:
                raw = _ybt_to_14(df[col])
                if reverse:
                    df[col] = raw.apply(
                        lambda x: 1 if pd.notna(x) and 1 <= x <= 2 else (0 if pd.notna(x) and 3 <= x <= 4 else np.nan)
                    )
                else:
                    df[col] = raw.apply(
                        lambda x: 1 if pd.notna(x) and 3 <= x <= 4 else (0 if pd.notna(x) and 1 <= x <= 2 else np.nan)
                    )
                df[col] = df[col].fillna(0)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["eq_total"] = df[[f"eq_{i}" for i in range(1, 11) if f"eq_{i}" in df.columns]].sum(axis=1)
    df["sqr_total"] = df[[f"sqr_{i}" for i in range(1, 11) if f"sqr_{i}" in df.columns]].sum(axis=1)
    df["aq_total"] = df[[f"aq_{i}" for i in range(1, 11) if f"aq_{i}" in df.columns]].sum(axis=1)
    sex_map = {"male": 0, "female": 1, "m": 0, "f": 1, "Male": 0, "Female": 1}
    df["sex_num"] = df["sex"].astype(str).str.strip().str.lower().map(sex_map).fillna(0)
    autism = df[df["diagnosis"] == 1]
    non_autism = df[df["diagnosis"] == 0]
    autism = autism[autism["aq_total"] >= 6]
    df = pd.concat([autism, non_autism], ignore_index=True)
    pos = df[df["diagnosis"] == 1]
    neg = df[df["diagnosis"] == 0]
    n = min(len(pos), len(neg))
    if n == 0:
        return df
    pos = pos.sample(n=n, random_state=RANDOM_STATE)
    neg = neg.sample(n=n, random_state=RANDOM_STATE)
    df = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=RANDOM_STATE)
    keep = ["age", "sex_num", "diagnosis", "aq_total", "eq_total", "sqr_total"] + [
        c for c in COMOBIDITY_COLUMNS if c in df.columns
    ]
    return df[[c for c in keep if c in df.columns]].copy()


def main():
    """Load balanced datasets and compute demographics."""
    print("=" * 80)
    print("DEMOGRAPHIC STATISTICS FROM BALANCED DATASETS (50/50)")
    print("=" * 80)
    print("\nThese statistics are computed from the balanced datasets used for modeling")
    print("(Study 1: balance_50_50=True, age 18-55, AQ>=6 for autism cases).\n")
    
    all_stats = []
    
    # C4 - load and balance
    print("Loading C4 cohort...")
    c4_path = os.path.join(REPO_ROOT, "data", "processed", "data_c4_final_recreated_cleaned.csv")
    if os.path.isfile(c4_path):
        df_c4, _, _ = load_cohort_c4(
            c4_path,
            age_min=18,
            age_max=55,
            balance_50_50=True,
            apply_aq_filter=True,
            keep_all_columns=True  # Keep all columns for demographics
        )
        stats_c4 = format_statistics(df_c4, "C4")
        all_stats.append(stats_c4)
        print(f"  Loaded: {len(df_c4)} rows (balanced 50/50)")
    else:
        print(f"  ⚠️  File not found: {c4_path}")
    
    # CARD - card_aligned.csv is already balanced, load directly
    print("\nLoading CARD cohort...")
    card_path = os.path.join(REPO_ROOT, "data", "processed", "card_aligned.csv")
    if os.path.isfile(card_path):
        df_card = pd.read_csv(card_path)
        if "autism_target" in df_card.columns:
            df_card = df_card.rename(columns={"autism_target": "diagnosis"})
        # card_aligned.csv is already balanced and filtered, just verify age range
        if "age" in df_card.columns:
            df_card = df_card[(df_card["age"] >= 18) & (df_card["age"] <= 55)]
        
        stats_card = format_statistics(df_card, "CARD")
        all_stats.append(stats_card)
        print(f"  Loaded: {len(df_card)} rows (already balanced 50/50)")
    else:
        print(f"  ⚠️  File not found: {card_path}")
    
    # YBT - check if ybt_aligned.csv exists (already balanced), otherwise load and balance
    print("\nLoading YBT cohort...")
    ybt_aligned_path = os.path.join(REPO_ROOT, "data", "processed", "ybt_aligned.csv")
    _default_ybt = os.path.expanduser("~/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/YBT.csv")
    _repo_ybt = os.path.join(REPO_ROOT, "data", "raw", "YBT.csv")
    
    # Prefer aligned file if it exists (already balanced); otherwise load raw with aq_total retained
    if os.path.isfile(ybt_aligned_path):
        df_ybt = pd.read_csv(ybt_aligned_path)
        if "autism_target" in df_ybt.columns:
            df_ybt = df_ybt.rename(columns={"autism_target": "diagnosis"})
        if "age" in df_ybt.columns:
            df_ybt = df_ybt[(df_ybt["age"] >= 18) & (df_ybt["age"] <= 55)]
        if "aq_total" not in df_ybt.columns and all(f"aq_{i}" in df_ybt.columns for i in range(1, 11)):
            df_ybt["aq_total"] = df_ybt[[f"aq_{i}" for i in range(1, 11)]].sum(axis=1)
        print(f"  Loaded aligned file: {len(df_ybt)} rows (already balanced)")
    else:
        ybt_path = _default_ybt if os.path.isfile(_default_ybt) else _repo_ybt
        if os.path.isfile(ybt_path):
            df_ybt = load_ybt_with_aq(ybt_path, age_min=18, age_max=55)
            print(f"  Loaded and balanced (with AQ): {len(df_ybt)} rows")
        else:
            df_ybt = pd.DataFrame()
            print(f"  ⚠️  File not found: {ybt_path}")
    
    if not df_ybt.empty:
        stats_ybt = format_statistics(df_ybt, "YBT")
        all_stats.append(stats_ybt)
    
    # Print results
    print("\n" + "=" * 80)
    print("DEMOGRAPHIC STATISTICS SUMMARY")
    print("=" * 80)
    
    if all_stats:
        df_stats = pd.DataFrame(all_stats)
        
        # Print formatted table
        print("\n### Sample Sizes")
        print(df_stats[["Cohort", "N_total", "N_autism", "N_non_autism"]].to_string(index=False))
        
        if "Age_mean_all" in df_stats.columns:
            print("\n### Age Statistics")
            age_cols = [c for c in df_stats.columns if c.startswith("Age_")]
            print(df_stats[["Cohort"] + age_cols].to_string(index=False))
        
        if "Sex_male_all" in df_stats.columns:
            print("\n### Sex Distribution")
            sex_cols = [c for c in df_stats.columns if c.startswith("Sex_")]
            print(df_stats[["Cohort"] + sex_cols].to_string(index=False))
        
        for q_name in ["AQ", "EQ", "SQ-R", "SPQ"]:
            q_cols = [c for c in df_stats.columns if c.startswith(f"{q_name}_")]
            if q_cols:
                print(f"\n### {q_name}-10 Statistics")
                print(df_stats[["Cohort"] + q_cols].to_string(index=False))
        
        comorb_cols = [c for c in df_stats.columns if any(c.startswith(com) for com in ["Adhd_", "Anxiety_", "Depression_"])]
        if comorb_cols:
            print("\n### Comorbidity Counts")
            print(df_stats[["Cohort"] + comorb_cols].to_string(index=False))
        
        # Save to CSV
        results_dir = os.path.join(REPO_ROOT, "results")
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, "balanced_demographics.csv")
        df_stats.to_csv(out_path, index=False)
        print(f"\n✅ Saved full statistics to: {out_path}")
    else:
        print("\n⚠️  No statistics computed (no datasets loaded)")


if __name__ == "__main__":
    main()
