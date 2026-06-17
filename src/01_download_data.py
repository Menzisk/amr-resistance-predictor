"""
01_download_data.py

Downloads Acinetobacter baumannii AMR phenotype data from BV-BRC public API.
Saves raw data to data/raw/ as a clean TSV.

Data source: BV-BRC (https://www.bv-brc.org)
Organism:    Acinetobacter baumannii (NCBI Taxon ID: 470)

Strategy: Download Resistant and Susceptible records separately to ensure
both classes are captured regardless of API result ordering.

Field names verified by direct schema inspection on 2026-06-16.
"""

import requests
import pandas as pd
import os
import json
import time

# ── Constants ────────────────────────────────────────────────────────────────

BASE_URL = "https://www.bv-brc.org/api/genome_amr/"

FIELDS = [
    "genome_id",
    "genome_name",
    "antibiotic",
    "resistant_phenotype",
    "measurement_value",
    "measurement_unit",
    "measurement_sign",
    "laboratory_typing_method",
    "evidence",
    "taxon_id",
]

# NCBI Taxonomy ID for Acinetobacter baumannii
TAXON_ID = 470

# We download each phenotype class separately to guarantee both are captured
PHENOTYPE_CLASSES = ["Resistant", "Susceptible"]

# How many records to request per class
# BV-BRC allows up to 25000 per request
LIMIT_PER_CLASS = 25000

OUTPUT_DIR  = "data/raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "acinetobacter_baumannii_amr_raw.tsv")


# ── Functions ────────────────────────────────────────────────────────────────

def build_query_url(phenotype: str) -> str:
    """
    Construct a BV-BRC RQL query URL for a single phenotype class.

    Args:
        phenotype: either "Resistant" or "Susceptible"

    Returns:
        str: fully constructed query URL
    """
    field_str = ",".join(FIELDS)

    url = (
        BASE_URL
        + f"?eq(taxon_id,{TAXON_ID})"
        + f"&eq(resistant_phenotype,{phenotype})"
        + f"&select({field_str})"
        + f"&limit({LIMIT_PER_CLASS})"
    )
    return url


def download_one_class(phenotype: str) -> pd.DataFrame:
    """
    Download AMR records for a single phenotype class.

    Args:
        phenotype: "Resistant" or "Susceptible"

    Returns:
        pd.DataFrame: records for that phenotype class

    Raises:
        ConnectionError: if the HTTP request fails
    """
    url = build_query_url(phenotype)
    print(f"\nDownloading: {phenotype}")
    print(f"URL: {url}")

    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers, timeout=60)

    print(f"HTTP status: {response.status_code}")

    if response.status_code != 200:
        raise ConnectionError(
            f"API request failed for {phenotype}.\n"
            f"Status : {response.status_code}\n"
            f"Details: {response.text[:500]}"
        )

    records = json.loads(response.text)
    df = pd.DataFrame(records)

    print(f"Records received: {len(df):,}")
    return df


def download_amr_data() -> pd.DataFrame:
    """
    Download both Resistant and Susceptible records and combine them.

    Why separate downloads?
    The API returns records in an unspecified order and caps results
    at our limit. Downloading each class separately guarantees we get
    both, regardless of how the database orders its output.

    Returns:
        pd.DataFrame: combined AMR records for both phenotype classes
    """
    print("=" * 60)
    print("BV-BRC AMR Data Download")
    print("=" * 60)
    print(f"Organism : Acinetobacter baumannii (taxon_id={TAXON_ID})")
    print(f"Classes  : {PHENOTYPE_CLASSES}")
    print(f"Limit    : {LIMIT_PER_CLASS:,} records per class")

    frames = []

    for phenotype in PHENOTYPE_CLASSES:
        df_class = download_one_class(phenotype)
        frames.append(df_class)
        # Wait 1 second between requests
        # This is called rate limiting and is good API etiquette
        time.sleep(1)

    # pd.concat() stacks DataFrames vertically (row-wise)
    # ignore_index=True resets the row numbers so they run 0,1,2,...
    # without it you would have duplicate index numbers from each chunk
    df_combined = pd.concat(frames, ignore_index=True)

    print(f"\nTotal records combined: {len(df_combined):,}")
    return df_combined


def filter_laboratory_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only laboratory-confirmed phenotypes.

    Computational predictions are outputs of other models.
    Training our model on another model's predictions would be
    circular — we would be learning from model noise, not biology.

    This is a fundamental principle in machine learning:
    your labels must come from ground truth, not other models.

    Args:
        df: raw combined DataFrame

    Returns:
        pd.DataFrame: filtered to laboratory evidence only
    """
    before = len(df)
    df_lab = df[df["evidence"] == "Laboratory Method"].copy()
    after  = len(df_lab)

    print(f"\nFiltering to Laboratory Method only:")
    print(f"  Before : {before:,} records")
    print(f"  After  : {after:,} records")
    print(f"  Removed: {before - after:,} computational predictions")

    return df_lab


def save_raw_data(df: pd.DataFrame) -> None:
    """
    Save raw DataFrame to TSV without modification.
    Raw data is never modified after saving — only read.

    Args:
        df: DataFrame to save
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, sep="\t", index=False)
    print(f"\nRaw data saved → {OUTPUT_FILE}")


def print_data_summary(df: pd.DataFrame) -> None:
    """
    Print a plain-language summary of the downloaded dataset.
    First sanity check before any cleaning or analysis.

    Args:
        df: raw DataFrame
    """
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)

    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    print("\nColumn names:")
    for col in df.columns:
        print(f"  - {col}")

    print("\nResistant phenotype distribution:")
    print(df["resistant_phenotype"].value_counts().to_string())

    print("\nTop 15 antibiotics by record count:")
    print(df["antibiotic"].value_counts().head(15).to_string())

    print("\nLaboratory typing methods:")
    print(df["laboratory_typing_method"].value_counts().to_string())

    print("\nMissing values per column:")
    print(df.isnull().sum().to_string())


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_raw = download_amr_data()
    df_lab = filter_laboratory_only(df_raw)
    save_raw_data(df_lab)
    print_data_summary(df_lab)
    print("\n✓ Download complete.")