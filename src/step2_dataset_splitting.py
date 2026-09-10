import os
import pandas as pd
from sklearn.model_selection import train_test_split

"""
=============================================================================
STAGE 2: STRATIFIED DATASET SPLITTING PIPELINE
Project: Judicia — Legal Domain Classification System
=============================================================================
This script performs stratified 80/10/10 dataset splitting:
1. 80% Training Set (Used to fine-tune InLegalBERT weights)
2. 10% Validation Set (Used for hyperparameter tuning & early stopping)
3. 10% Test Set (Held out for unseen test evaluation)
Stratification ensures equal class representation across all 8 legal domains.
"""

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

CLEANED_DATA_PATH = os.path.join(PROJECT_ROOT, "dataset", "processed", "legal_domain_dataset_cleaned.xlsx")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "dataset", "processed")

def split_data():
    print("=" * 75)
    print("STAGE 2: STRATIFIED DATASET SPLITTING (80/10/10)")
    print("=" * 75)

    if not os.path.exists(CLEANED_DATA_PATH):
        raise FileNotFoundError(f"Cleaned dataset not found at: {CLEANED_DATA_PATH}! Run step1_data_cleaning.py first.")

    print(f"Loading cleaned dataset from: dataset/processed/legal_domain_dataset_cleaned.xlsx")
    df = pd.read_excel(CLEANED_DATA_PATH)

    # 1. Perform 80% Train, 20% Temporary Split (Stratified by label)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["label"],
        random_state=42
    )

    # 2. Perform 50/50 Split on Temporary Set -> 10% Validation, 10% Test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=42
    )

    # 3. Save splits strictly inside dataset/processed/
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, "validation.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)

    # 4. Output Summary Statistics
    total_samples = len(df)
    print(f"\nTotal Cleaned Samples : {total_samples}")
    print(f"  Training Set (80%)  : {len(train_df)} samples")
    print(f"  Validation Set (10%): {len(val_df)} samples")
    print(f"  Test Set (10%)      : {len(test_df)} samples")

    print("\nTraining Set Class Distribution:")
    print(train_df["label"].value_counts().to_string())
    print("-" * 75)
    print("Splitting complete! Saved train.csv, validation.csv, and test.csv in dataset/processed/\n")

    return train_df, val_df, test_df

if __name__ == "__main__":
    split_data()
