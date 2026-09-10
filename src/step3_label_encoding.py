import os
import json
import pandas as pd

"""
=============================================================================
STAGE 3: LABEL ENCODING & MAPPING PIPELINE
Project: Judicia — Legal Domain Classification System
=============================================================================
This script maps textual target labels ('Civil', 'Cyber', etc.) into integer IDs (0..7)
and generates:
1. label_mapping.json (ID-to-Label & Label-to-ID lookup dictionary)
2. train_encoded.csv, validation_encoded.csv, test_encoded.csv
"""

def encode_labels():
    print("=" * 70)
    print("STAGE 3: LABEL ENCODING & MAPPING")
    print("=" * 70)

    # 1. Locate train/val/test splits
    base_dir = "dataset/processed" if os.path.exists("dataset/processed/train.csv") else "dataset"

    train_path = os.path.join(base_dir, "train.csv")
    val_path = os.path.join(base_dir, "validation.csv")
    test_path = os.path.join(base_dir, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Split files not found in {base_dir}! Run 02_dataset_splitting.py first.")

    print(f"Loading split files from: {base_dir}/")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # 2. Extract unique labels and build mapping
    labels = sorted(train_df["label"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}

    print("\n--- GENERATED LABEL MAPPING ---")
    for k, v in label2id.items():
        print(f"  {k:<15} -> Class ID: {v}")

    # 3. Map textual labels to numeric IDs
    train_df["label"] = train_df["label"].map(label2id)
    val_df["label"] = val_df["label"].map(label2id)
    test_df["label"] = test_df["label"].map(label2id)

    # 4. Save mapping JSON and encoded datasets to processed/ and dataset/
    mapping_dict = {
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()}
    }

    output_dirs = ["dataset/processed", "dataset"]
    for out_dir in output_dirs:
        os.makedirs(out_dir, exist_ok=True)
        train_df.to_csv(os.path.join(out_dir, "train_encoded.csv"), index=False)
        val_df.to_csv(os.path.join(out_dir, "validation_encoded.csv"), index=False)
        test_df.to_csv(os.path.join(out_dir, "test_encoded.csv"), index=False)

        with open(os.path.join(out_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
            json.dump(mapping_dict, f, indent=4)

    print("-" * 70)
    print("Label encoding complete! Saved label_mapping.json & *_encoded.csv files.\n")
    return label2id, id2label

if __name__ == "__main__":
    encode_labels()
