#!/usr/bin/env python3
"""
02_clean_and_engineer.py

Cleans raw AMR data and engineers features for machine learning.

This script performs the following steps:
    1. Subsets data to three clinically relevant aminoglycosides
    2. Drops uninformative columns (taxon_id, evidence, genome_id, genome_name)
    3. Drops target-leaking columns (measurement_value, measurement_sign)
    4. One-hot encodes categorical features (antibiotic, laboratory_typing_method)
    5. Encodes the target label (Resistant=1, Susceptible=0)
    6. Saves the clean feature set as a TSV file

Inputs:
    data/raw/acinetobacter_baumannii_amr_raw.tsv
        - Raw AMR phenotype data from 01_download_data.py

Outputs:
    data/processed/amr_features.tsv
        - Engineered features ready for modelling
        - Columns: mic_above_range, antibiotic_*, laboratory_typing_method_*, label
        - Target leakage removed (no MIC values)

Usage:
    python src/02_clean_and_engineer.py

Author: Menzi Sikakane (menzisk)
Date:   2026-06-17
License: MIT
"""

import pandas as pd
import numpy as np
import os
import yaml

# ── Load Configuration ─────────────────────────────────────────────────────
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# Extract settings
RAW_DIR = CONFIG['data']['raw_dir']
PROCESSED_DIR = CONFIG['data']['processed_dir']
MODEL_ANTIBIOTICS = CONFIG['biology']['model_antibiotics']

# ── Constants ──────────────────────────────────────────────────────────────

INPUT_FILE = os.path.join(RAW_DIR, "acinetobacter_baumannii_amr_raw.tsv")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "amr_features.tsv")

# Columns to drop - reasons documented explicitly
DROP_COLUMNS = {
    "taxon_id": "constant (always 470) — zero information",
    "evidence": "constant after lab filter — zero information",
    "genome_id": "unique identifier — not a biological feature",
    "genome_name": "free-text strain name — too many unique values",
    "measurement_unit": "redundant with laboratory_typing_method",
    "measurement_value": "TARGET LEAKAGE — MIC used to derive resistance label",
    "measurement_sign": "TARGET LEAKAGE — sign part of MIC reporting",
}


# ── Functions ──────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load raw TSV and report shape."""
    df = pd.read_csv(filepath, sep="\t")
    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def subset_antibiotics(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the three aminoglycosides we are modelling."""
    df_sub = df[df["antibiotic"].isin(MODEL_ANTIBIOTICS)].copy()
    print(f"\nAfter antibiotic subset: {df_sub.shape[0]:,} rows")
    return df_sub


def drop_uninformative_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that carry no predictive information."""
    print("\nDropping uninformative columns:")
    for col, reason in DROP_COLUMNS.items():
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"  Dropped '{col}': {reason}")
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical features."""
    categorical_cols = ["antibiotic", "laboratory_typing_method"]

    print("\nOne-hot encoding categorical columns:")
    for col in categorical_cols:
        if col in df.columns:
            n_unique = df[col].nunique()
            print(f"  {col}: {n_unique} categories → {n_unique} new columns")

    # Fill missing laboratory_typing_method before encoding
    df["laboratory_typing_method"] = df["laboratory_typing_method"].fillna("Unknown")

    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=False,
        dtype=int,
    )

    return df


def encode_label(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the target variable as a binary integer."""
    label_map = {"Resistant": 1, "Susceptible": 0}
    df["label"] = df["resistant_phenotype"].map(label_map)

    print(f"\nLabel encoding:")
    print(f"  Resistant   → 1  ({(df['label']==1).sum():,} records)")
    print(f"  Susceptible → 0  ({(df['label']==0).sum():,} records)")

    return df


def drop_raw_columns_post_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that have been superseded."""
    cols_to_drop = ["resistant_phenotype"]
    existing = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing)
    print(f"\nDropped post-encoding columns: {existing}")
    return df


def save_processed_data(df: pd.DataFrame) -> None:
    """Save the engineered feature set."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, sep="\t", index=False)
    print(f"\nProcessed data saved → {OUTPUT_FILE}")


def print_feature_summary(df: pd.DataFrame) -> None:
    """Print a summary of the final feature set."""
    print("\n" + "=" * 60)
    print("FINAL FEATURE SET SUMMARY")
    print("=" * 60)
    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("\nAll columns and dtypes:")
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_missing = df[col].isnull().sum()
        flag = " ← TARGET" if col == "label" else ""
        missing_str = f"  [{n_missing} missing]" if n_missing > 0 else ""
        print(f"  {col:<45} {dtype:<10}{missing_str}{flag}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("AMR Feature Engineering Pipeline")
    print("=" * 60)

    df = load_data(INPUT_FILE)
    df = subset_antibiotics(df)
    df = drop_uninformative_columns(df)
    df = encode_categorical_features(df)
    df = encode_label(df)
    df = drop_raw_columns_post_encoding(df)
    save_processed_data(df)
    print_feature_summary(df)

    print("\n Feature engineering complete.")