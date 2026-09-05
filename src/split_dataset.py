import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_excel("dataset/legal_domain_dataset.xlsx")

print("Original dataset:", len(df))

# 80% Train, 20% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["label"],
    random_state=42
)

# Temporary 50-50 split:
# 10% Validation + 10% Test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

# Save splits inside dataset folder
train_df.to_csv("dataset/train.csv", index=False)
val_df.to_csv("dataset/validation.csv", index=False)
test_df.to_csv("dataset/test.csv", index=False)

print("\nTrain:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

print("\nTrain distribution:")
print(train_df["label"].value_counts())

print("\nValidation distribution:")
print(val_df["label"].value_counts())

print("\nTest distribution:")
print(test_df["label"].value_counts())