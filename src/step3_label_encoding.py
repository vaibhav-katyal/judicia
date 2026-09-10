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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "dataset", "processed")

def encode_labels():
    print("=" * 75)
    print("STAGE 3: LABEL ENCODING & MAPPING GENERATION")
    print("=" * 75)

    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    val_path = os.path.join(PROCESSED_DIR, "validation.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Split files not found in dataset/processed/! Run step2_dataset_splitting.py first.")

    print(f"Loading split files from: dataset/processed/")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # 1. Extract unique labels and build mapping
    labels = sorted(train_df["label"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}

    print("\n--- GENERATED LABEL MAPPING ---")
    for k, v in label2id.items():
        print(f"  {k:<15} -> Class ID: {v}")

    # 2. Map textual labels to numeric IDs
    train_df["label"] = train_df["label"].map(label2id)
    val_df["label"] = val_df["label"].map(label2id)
    test_df["label"] = test_df["label"].map(label2id)

    # 3. Save mapping JSON and encoded datasets strictly to dataset/processed/
    mapping_dict = {
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()}
    }

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(PROCESSED_DIR, "train_encoded.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, "validation_encoded.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test_encoded.csv"), index=False)

    with open(os.path.join(PROCESSED_DIR, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(mapping_dict, f, indent=4)

    print("-" * 75)
    print("Label encoding complete! Saved label_mapping.json & *_encoded.csv in dataset/processed/\n")
    return label2id, id2label

if __name__ == "__main__":
    encode_labels()
