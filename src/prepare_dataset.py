import pandas as pd
import json

# Load datasets
train_df = pd.read_csv("dataset/train.csv")
val_df = pd.read_csv("dataset/validation.csv")
test_df = pd.read_csv("dataset/test.csv")

# Get all unique labels
labels = sorted(train_df["label"].unique())

# Create mapping
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

print("Label mapping:")
print(label2id)

# Convert labels to numbers
train_df["label"] = train_df["label"].map(label2id)
val_df["label"] = val_df["label"].map(label2id)
test_df["label"] = test_df["label"].map(label2id)

# Save encoded datasets
train_df.to_csv("dataset/train_encoded.csv", index=False)
val_df.to_csv("dataset/validation_encoded.csv", index=False)
test_df.to_csv("dataset/test_encoded.csv", index=False)

# Save mapping
with open("dataset/label_mapping.json", "w") as f:
    json.dump(
        {
            "label2id": label2id,
            "id2label": {str(k): v for k, v in id2label.items()}
        },
        f,
        indent=4
    )

print("\nEncoding complete!")