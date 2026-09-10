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

def split_data():
    print("=" * 70)
    print("STAGE 2: STRATIFIED DATASET SPLITTING")
    print("=" * 70)

    # 1. Locate cleaned dataset
    possible_paths = [
        "dataset/processed/legal_domain_dataset_cleaned.xlsx",
        "dataset/legal_domain_dataset_cleaned.xlsx",
        "dataset/raw/legal_domain_dataset.xlsx",
        "dataset/legal_domain_dataset.xlsx"
    ]

    input_file = None
    for path in possible_paths:
        if os.path.exists(path):
            input_file = path
            break

    if not input_file:
        raise FileNotFoundError("Cleaned dataset not found for splitting!")

    print(f"Loading cleaned dataset from: {input_file}")
    df = pd.read_excel(input_file)

    # 2. Perform 80% Train, 20% Temporary Split (Stratified by label)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["label"],
        random_state=42
    )

    # 3. Perform 50/50 Split on Temporary Set -> 10% Validation, 10% Test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=42
    )

    # 4. Save splits to dataset/processed and dataset/
    output_dirs = ["dataset/processed", "dataset"]
    for out_dir in output_dirs:
        os.makedirs(out_dir, exist_ok=True)
        train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(out_dir, "validation.csv"), index=False)
        test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)

    # 5. Output Summary Statistics
    total_samples = len(df)
    print(f"\nTotal Dataset Samples : {total_samples}")
    print(f"Training Set (80%)    : {len(train_df)} samples")
    print(f"Validation Set (10%)  : {len(val_df)} samples")
    print(f"Test Set (10%)        : {len(test_df)} samples")

    print("\n--- Training Set Class Distribution ---")
    print(train_df["label"].value_counts().to_string())
    print("-" * 70)
    print("Splitting complete! Exported train.csv, validation.csv, and test.csv\n")

    return train_df, val_df, test_df

if __name__ == "__main__":
    split_data()
