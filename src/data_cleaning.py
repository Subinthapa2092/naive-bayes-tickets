"""
data_cleaning.py
Reusable cleaning pipeline for naive-bayes-tickets project.

Author : Subin (Person 1)

Usage:
    from src.data_cleaning import load_raw_data, clean_tickets, save_clean_data

    df_raw   = load_raw_data("data/customer_support_tickets.csv")
    df_clean = clean_tickets(df_raw)
    save_clean_data(df_clean, "data/cleaned_tickets.csv")
"""

import os
import warnings
import logging
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Column constants
PII_COLS          = ["Customer Name", "Customer Email"]
RAW_DATETIME_COLS = ["Date of Purchase", "First Response Time", "Time to Resolution"]
INT_COLS          = ["Customer Age", "Ticket ID", "purchase_year", "purchase_month", "purchase_day", "purchase_dow"]


# Load
def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV from filepath."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")
    df = pd.read_csv(path)
    log.info("Loaded %s | shape %s", path.name, df.shape)
    return df


# Audit
def null_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return null count and percentage for columns that have missing values."""
    counts = df.isnull().sum()
    pct    = (counts / len(df) * 100).round(2)
    report = pd.DataFrame({"null_count": counts, "null_%": pct})
    return report[report["null_count"] > 0]


# Cleaning steps
def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log.info("Duplicates removed: %d | rows remaining: %d", before - len(df), len(df))
    return df


def drop_pii(df: pd.DataFrame) -> pd.DataFrame:
    """Drop PII columns (Customer Name, Customer Email)."""
    cols = [c for c in PII_COLS if c in df.columns]
    df = df.drop(columns=cols)
    log.info("Dropped PII: %s", cols)
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse datetime columns and extract numeric features."""
    # Date of Purchase → year, month, day, day-of-week
    df["Date of Purchase"] = pd.to_datetime(df["Date of Purchase"], errors="coerce")
    df["purchase_year"]    = df["Date of Purchase"].dt.year
    df["purchase_month"]   = df["Date of Purchase"].dt.month
    df["purchase_day"]     = df["Date of Purchase"].dt.day
    df["purchase_dow"]     = df["Date of Purchase"].dt.dayofweek

    # First Response Time → hour + binary flag
    df["First Response Time"]  = pd.to_datetime(df["First Response Time"], errors="coerce")
    df["first_response_hour"]  = df["First Response Time"].dt.hour
    df["has_first_response"]   = df["First Response Time"].notna().astype(int)

    # Time to Resolution → hour + binary flag
    df["Time to Resolution"] = pd.to_datetime(df["Time to Resolution"], errors="coerce")
    df["resolution_hour"]    = df["Time to Resolution"].dt.hour
    df["is_resolved"]        = df["Time to Resolution"].notna().astype(int)

    log.info("Date features extracted.")
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with domain-appropriate strategies."""
    # Resolution: null only for open tickets → fill with 'Pending'
    df["Resolution"] = df["Resolution"].fillna("Pending")

    # Satisfaction Rating: null for non-closed tickets → fill 0, add flag
    df["satisfaction_rated_flag"]        = df["Customer Satisfaction Rating"].notna().astype(int)
    df["Customer Satisfaction Rating"]   = df["Customer Satisfaction Rating"].fillna(0)

    # Hour columns: -1 sentinel (valid hours are 0–23)
    df["first_response_hour"] = df["first_response_hour"].fillna(-1).astype(int)
    df["resolution_hour"]     = df["resolution_hour"].fillna(-1).astype(int)

    log.info("Missing values filled.")
    return df


def fix_placeholder(df: pd.DataFrame) -> pd.DataFrame:
    """Replace {product_purchased} placeholder in Ticket Description with actual product name."""
    df["Ticket Description"] = df.apply(
        lambda row: row["Ticket Description"].replace(
            "{product_purchased}", str(row["Product Purchased"])
        ) if isinstance(row["Ticket Description"], str) else row["Ticket Description"],
        axis=1,
    )
    log.info("Placeholders replaced.")
    return df


def cast_and_drop_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to int and drop raw datetime columns."""
    for col in INT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(int)

    present = [c for c in RAW_DATETIME_COLS if c in df.columns]
    df = df.drop(columns=present)
    log.info("Dtypes cast | raw datetime cols dropped.")
    return df


# Full pipeline
def clean_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline and return a clean dataframe."""
    log.info("Starting cleaning pipeline...")
    df = (
        df
        .pipe(drop_duplicates)
        .pipe(drop_pii)
        .pipe(parse_dates)
        .pipe(fill_missing_values)
        .pipe(fix_placeholder)
        .pipe(cast_and_drop_raw)
    )
    log.info("Pipeline complete | final shape: %s", df.shape)
    return df


# Save
def save_clean_data(df: pd.DataFrame, output_path: str = "data/cleaned_tickets.csv") -> None:
    """Save cleaned dataframe to CSV and verify the write."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    # Verify
    saved = pd.read_csv(path)
    assert saved.shape == df.shape, "Shape mismatch after save!"
    log.info("Saved %s | shape %s | nulls %d", path, saved.shape, saved.isnull().sum().sum())