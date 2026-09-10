import os
import re
import pandas as pd

"""
=============================================================================
STAGE 1: DATA CLEANING & PREPROCESSING PIPELINE
Project: Judicia — Legal Domain Classification System
=============================================================================
This script performs rigorous data hygiene on the raw dataset:
1. Missing Value Detection & Removal (NaN text or label, empty rows)
2. Text Whitespace & Formatting Normalization
3. Exact Duplicate & Duplicate Statement Elimination
4. Class Balance Verification & Clean Export
"""

# Determine absolute path to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "dataset", "raw", "legal_domain_dataset.xlsx")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "dataset", "processed")
CLEANED_DATA_PATH = os.path.join(PROCESSED_DIR, "legal_domain_dataset_cleaned.xlsx")

def clean_data():
    print("=" * 75)
    print("STAGE 1: DATASET CLEANING & PREPROCESSING")
    print("=" * 75)

    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw dataset file not found at: {RAW_DATA_PATH}")

    print(f"Loading raw dataset from: dataset/raw/legal_domain_dataset.xlsx")
    df = pd.read_excel(RAW_DATA_PATH)

    # 1. Initial Statistics Before Cleaning
    initial_rows = len(df)
    missing_text_count = df["text"].isnull().sum() if "text" in df.columns else 0
    missing_label_count = df["label"].isnull().sum() if "label" in df.columns else 0
    duplicate_rows = df.duplicated().sum()
    duplicate_texts = df["text"].duplicated().sum() if "text" in df.columns else 0

    print("\n--- STATS BEFORE CLEANING ---")
    print(f"  Total Raw Rows             : {initial_rows}")
    print(f"  Missing Text Statements    : {missing_text_count}")
    print(f"  Missing Label Entries      : {missing_label_count}")
    print(f"  Duplicate Full Rows        : {duplicate_rows}")
    print(f"  Duplicate Text Statements  : {duplicate_texts}")

    # 2. Remove completely empty rows
    df = df.dropna(how="all")

    # 3. Remove rows missing text or label
    df = df.dropna(subset=["text", "label"])

    # 4. Clean & normalize text formatting
    df["text"] = df["text"].astype(str)
    df["text"] = df["text"].str.strip()  # Remove leading & trailing whitespace
    df["text"] = df["text"].apply(lambda x: re.sub(r"\s+", " ", x))  # Collapse multiple spaces, tabs & newlines

    # 5. Remove empty strings
    df = df[df["text"] != ""]

    # 6. Deduplication
    df = df.drop_duplicates()  # Drop exact duplicate rows
    df = df.drop_duplicates(subset=["text"])  # Drop duplicate legal queries

    # 7. Reset index
    df = df.reset_index(drop=True)

    # 8. Export to dataset/processed/
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_excel(CLEANED_DATA_PATH, index=False)

    # 9. Final Statistics Verification
    final_rows = len(df)
    removed_rows = initial_rows - final_rows

    print("\n--- STATS AFTER CLEANING ---")
    print(f"  Cleaned Total Rows         : {final_rows}")
    print(f"  Removed Noisy/Duplicate Rows: {removed_rows} rows removed")
    print(f"  Remaining Duplicates       : {df['text'].duplicated().sum()}")
    print("\nClass Balance Distribution:")
    print(df["label"].value_counts().to_string())
    print("-" * 75)
    print(f"Cleaned dataset saved to: dataset/processed/legal_domain_dataset_cleaned.xlsx\n")

    return df

if __name__ == "__main__":
    clean_data()
