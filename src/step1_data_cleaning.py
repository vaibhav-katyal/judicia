import os
import re
import pandas as pd

"""
=============================================================================
STAGE 1: DATA CLEANING & PREPROCESSING PIPELINE
Project: Judicia — Legal Domain Classification System
=============================================================================
This script performs rigorous data hygiene on the raw dataset:
1. Missing Value Detection & Removal
2. Text Whitespace & Formatting Normalization
3. Duplicate Row & Duplicate Text Elimination
4. Class Balance Verification
"""

def clean_data():
    print("=" * 70)
    print("STAGE 1: DATASET CLEANING & PREPROCESSING")
    print("=" * 70)

    # 1. Locate raw input file
    possible_raw_paths = [
        "dataset/raw/legal_domain_dataset.xlsx",
        "dataset/legal_domain_dataset.xlsx",
        "../dataset/raw/legal_domain_dataset.xlsx",
        "../dataset/legal_domain_dataset.xlsx"
    ]

    input_file = None
    for path in possible_raw_paths:
        if os.path.exists(path):
            input_file = path
            break

    if not input_file:
        raise FileNotFoundError("Raw dataset legal_domain_dataset.xlsx not found!")

    print(f"Loading raw dataset from: {input_file}")
    df = pd.read_excel(input_file)

    # 2. Initial Statistics
    initial_rows = len(df)
    missing_count = df.isnull().sum().to_dict()
    duplicate_rows = df.duplicated().sum()
    duplicate_texts = df["text"].duplicated().sum() if "text" in df.columns else 0

    print("\n--- BEFORE CLEANING ---")
    print(f"Total Raw Rows       : {initial_rows}")
    print(f"Missing Values       : {missing_count}")
    print(f"Duplicate Full Rows  : {duplicate_rows}")
    print(f"Duplicate Text Statements: {duplicate_texts}")

    # 3. Remove completely empty rows
    df = df.dropna(how="all")

    # 4. Remove rows missing text or label
    df = df.dropna(subset=["text", "label"])

    # 5. Clean & normalize text formatting
    df["text"] = df["text"].astype(str)
    df["text"] = df["text"].str.strip()  # Remove leading & trailing whitespace
    df["text"] = df["text"].apply(lambda x: re.sub(r"\s+", " ", x))  # Collapse multiple spaces & newlines

    # 6. Remove empty strings
    df = df[df["text"] != ""]

    # 7. Deduplication
    df = df.drop_duplicates()  # Drop exact duplicate rows
    df = df.drop_duplicates(subset=["text"])  # Drop duplicate legal queries

    # 8. Reset index
    df = df.reset_index(drop=True)

    # 9. Save cleaned dataset
    output_dir_processed = "dataset/processed"
    output_dir_base = "dataset"
    os.makedirs(output_dir_processed, exist_ok=True)

    out_path_processed = os.path.join(output_dir_processed, "legal_domain_dataset_cleaned.xlsx")
    out_path_base = os.path.join(output_dir_base, "legal_domain_dataset_cleaned.xlsx")

    df.to_excel(out_path_processed, index=False)
    df.to_excel(out_path_base, index=False)

    # 10. Final Statistics Verification
    final_rows = len(df)
    removed_rows = initial_rows - final_rows

    print("\n--- AFTER CLEANING ---")
    print(f"Cleaned Total Rows   : {final_rows} (Removed {removed_rows} noisy/duplicate rows)")
    print(f"Remaining Duplicates : {df['text'].duplicated().sum()}")
    print("\nClass Distribution:")
    print(df["label"].value_counts().to_string())
    print("-" * 70)
    print(f"Cleaned dataset exported to: {out_path_processed}\n")

    return df

if __name__ == "__main__":
    clean_data()
