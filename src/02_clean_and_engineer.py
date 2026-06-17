"""
02_clean_and_engineer.py

Cleans the raw AMR dataset and engineers features for modelling.

Input:  data/raw/acinetobacter_baumannii_amr_raw.tsv
Output: data/processed/amr_features.tsv

Cleaning steps:
   1. Subset to aminoglycoside of interest
   2. Drop uniformative columns
   3. Parse measurement_value to numeric
   4. Encode categorical features
   5. Encode labels as binary interger
   6. Handle missing values

Aurthor: Menzi Sikakane
Date:    2026-06-17
"""

import pandas as pd
import numpy as np
import os

#constants/inputs 
INPUT_FILE  = "data/raw/acinetobacter_baumannii_amr_raw.tsv"
OUTPUT_DIR  = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "amr_features.tsv") 

#Aminoglycosides we are modelling
MODEL_ANTIBIOTICS = ["amikacin", "gentamicin", "tobramycin"]

#coolumns to drop
DROP_COLUMNS = {
    "taxon_id"        : "constant (always 470) — zero information",
    "evidence"        : "constant after lab filter — zero information",
    "genome_id"       : "unique identifier — not a biological feature",
    "genome_name"     : "free-text strain name — too many unique values",
    "measurement_unit": "redundant with laboratory_typing_method",
    "measurement_value": "TARGET LEAKAGE — MIC is used to derive the resistance label",
    "measurement_sign" : "TARGET LEAKAGE — sign is part of MIC reporting",
}

#funtions
def load_data(filepath: str) -> pd.DataFrame:
    """Load raw TSV and report shape"""
    df = pd.read_csv(filepath, sep="\t")
    print(f"Loaded : {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df

def subset_antibiotics(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Keep only 3 aminoglycosides we are modelling.

    This focuses our model on biologically coherent drug class relevant to ANT(3")-la enzyme inhibitor research.
    """
    df_sub = df[df["antibiotic"].isin(MODEL_ANTIBIOTICS)].copy()
    print(f"\nAfter antibiotic subset: {df_sub.shape[0]:,} rows")
    return df_sub

def drop_uniformative_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that carry no predictive information.
    Each dropped column is documented with a reason.
    """
    print("\nDropping uninformative columns:")
    for col, reason in DROP_COLUMNS.items():
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f" Dropped '{col}': {reason}")
    return df

def parse_measurement_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert measurement_value from text to a numeric MIC value.

    The raw column contains mixed formats:
        "16"     → clean numeric, MIC = 16
        ">32"    → above range, we extract 32
        ">=16"   → at or above, we extract 16
        "2/38"   → combination drug ratio, set to NaN (not interpretable)
        ""       → empty string, set to NaN

    Why does this matter?
    MIC (Minimum Inhibitory Concentration) is the lowest concentration
    of an antibiotic that prevents visible bacterial growth.
    Higher MIC → more resistant.
    This is one of our most biologically informative numeric features.

    Mathematical note:
    MIC values are typically reported on a doubling dilution scale:
    0.5, 1, 2, 4, 8, 16, 32, 64, 128 mg/L
    This is a geometric (log2) scale, not linear.
    We will log2-transform MIC values so that equal steps on our
    feature scale correspond to equal biological differences.
    """

    # Step 1: Replace empty strings with NaN
    df["measurement_value"] = df["measurement_value"].replace("", np.nan)

    # Step 2: Remove sign characters (>, <, =, >=, <=) to extract number
    df["mic_numeric"] = (
        df["measurement_value"]
        .astype(str)
        .str.replace(r"[><=]+", "", regex=True)  
        .str.strip()                               
    )

    # Step 3: Handle combination drug values like "2/38"
    df["mic_numeric"] = df["mic_numeric"].where(
        ~df["mic_numeric"].str.contains("/", na=False),
        other=np.nan
    )

    # Step 4: Convert to float (non-numeric strings become NaN)
    df["mic_numeric"] = pd.to_numeric(df["mic_numeric"], errors="coerce")

    # Step 5: Log2 transform
    df["mic_log2"] = np.log2(df["mic_numeric"] + 0.001)

    # Step 6: Extract measurement sign as a separate feature
    df["mic_above_range"] = (
        df["measurement_sign"]
        .fillna("")
        .str.contains(">")
        .astype(int)  
    )

    n_mic = df["mic_numeric"].notna().sum()
    print(f"\nMIC values successfully parsed: {n_mic:,} of {len(df):,}")
    print(f"MIC range: {df['mic_numeric'].min()} – {df['mic_numeric'].max()} mg/L")

    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical features.

    One-hot encoding converts a categorical column with k categories
    into k binary (0/1) columns. This avoids implying a false numerical
    ordering between categories.

    Example:
        antibiotic         →   antibiotic_amikacin  antibiotic_gentamicin  ...
        amikacin           →         1                      0
        gentamicin         →         0                      1

    drop_first=False: we keep all k columns (not k-1) because XGBoost
    handles multicollinearity and we want explicit biological interpretability
    for each drug in our SHAP plots.
    """
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
    """
    Encode the target variable (label) as a binary integer.

    ML classifiers require numeric labels:
        Resistant   → 1
        Susceptible → 0

    This choice (Resistant=1) means our model's positive class
    is resistance — which is the clinically important outcome to detect.
    """
    label_map = {"Resistant": 1, "Susceptible": 0}
    df["label"] = df["resistant_phenotype"].map(label_map)

    print(f"\nLabel encoding:")
    print(f"  Resistant   → 1  ({(df['label']==1).sum():,} records)")
    print(f"  Susceptible → 0  ({(df['label']==0).sum():,} records)")

    return df


def drop_raw_columns_post_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that have been superseded by engineered features.
    """
    cols_to_drop = [
        "resistant_phenotype",  # superseded by label
    ]
    existing = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing)
    print(f"\nDropped post-encoding columns: {existing}")
    return df


def save_processed_data(df: pd.DataFrame) -> None:
    """Save the engineered feature set to data/processed/."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
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
    df = drop_uniformative_columns(df)
    df = encode_categorical_features(df)
    df = encode_label(df)
    df = drop_raw_columns_post_encoding(df)
    save_processed_data(df)
    print_feature_summary(df)

    print("\n Feature engineering complete.")

