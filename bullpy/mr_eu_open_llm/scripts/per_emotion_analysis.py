from __future__ import annotations

"""Per-mental-state accuracy; excludes Neutral; high vs low intensity summary."""

import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

LOW_INTENSITY_TOKEN = "low intensity"


def compute_per_emotion_accuracy(results_df: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "mental_state", "is_correct"}
    missing = required - set(results_df.columns)
    if missing:
        raise ValueError(f"results_df missing columns: {sorted(missing)}")
    df = results_df[results_df["is_correct"].notna()].copy()
    df["is_correct"] = df["is_correct"].astype(bool)
    out = (
        df.groupby(["mental_state", "model"], as_index=False)
        .agg(n_trials=("is_correct", "size"), n_correct=("is_correct", "sum"))
    )
    out["accuracy"] = out["n_correct"] / out["n_trials"]
    return out[["mental_state", "model", "n_trials", "n_correct", "accuracy"]]


def remove_neutral(results_df: pd.DataFrame) -> pd.DataFrame:
    col = "mental_state" if "mental_state" in results_df.columns else "correct_label"
    if col not in results_df.columns:
        return results_df
    return results_df[results_df[col].astype(str).str.casefold() != "neutral"].copy()


def intensity_summary(per_emotion_df: pd.DataFrame) -> pd.DataFrame:
    df = per_emotion_df.copy()
    df["ms_cf"] = df["mental_state"].astype(str).str.casefold()
    df["is_low"] = df["ms_cf"].str.contains(LOW_INTENSITY_TOKEN, regex=False)
    df["base"] = df["ms_cf"].str.replace(f" {LOW_INTENSITY_TOKEN}", "", regex=False).str.strip()
    high = df[~df["is_low"]][["model", "base", "accuracy"]].rename(columns={"accuracy": "high_accuracy"})
    low = df[df["is_low"]][["model", "base", "accuracy"]].rename(columns={"accuracy": "low_accuracy"})
    merged = high.merge(low, on=["model", "base"], how="inner")
    if merged.empty:
        return pd.DataFrame(
            columns=["model", "n_pairs", "mean_high_accuracy", "mean_low_accuracy", "mean_difference"]
        )
    merged["difference"] = merged["high_accuracy"] - merged["low_accuracy"]
    return (
        merged.groupby("model", as_index=False)
        .agg(
            n_pairs=("base", "size"),
            mean_high_accuracy=("high_accuracy", "mean"),
            mean_low_accuracy=("low_accuracy", "mean"),
            mean_difference=("difference", "mean"),
        )
        .sort_values("mean_difference", ascending=False)
    )


def _load_results_csvs(results_dir: str) -> pd.DataFrame:
    results_path = Path(results_dir)
    csvs = sorted(results_path.glob("*_results.csv"))
    if not csvs:
        raise FileNotFoundError(f"No '*_results.csv' in {results_path}")
    return pd.concat([pd.read_csv(p).assign(source_file=p.name) for p in csvs], ignore_index=True)


def generate_report(results_dir: str, output_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    results_df = _load_results_csvs(results_dir)
    if "mental_state" not in results_df.columns and "correct_label" in results_df.columns:
        results_df = results_df.rename(columns={"correct_label": "mental_state"})
    results_df = remove_neutral(results_df)
    per_emotion = compute_per_emotion_accuracy(results_df)
    intens = intensity_summary(per_emotion)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    per_emotion.to_csv(output_path, index=False)
    return per_emotion, intens
