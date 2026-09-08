import pandas as pd
import re

# -----------------------------
# 1. Load dataset
# -----------------------------
input_file = "../dataset/legal_domain_dataset.xlsx"
output_file = "../dataset/legal_domain_dataset_cleaned.xlsx"

df = pd.read_excel(input_file)

print("========== BEFORE CLEANING ==========")
print("Total rows:", len(df))
# print("Missing values:")
# print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
print("Duplicate texts:", df["text"].duplicated().sum())


# -----------------------------
# 2. Remove completely empty rows
# -----------------------------
df = df.dropna(how="all")


# -----------------------------
# 3. Remove rows with missing
#    text or label
# -----------------------------
df = df.dropna(subset=["text", "label"])


# -----------------------------
# 4. Clean text formatting
# -----------------------------
df["text"] = df["text"].astype(str)

# Remove leading and trailing spaces
df["text"] = df["text"].str.strip()

# Replace multiple spaces/newlines/tabs with single space
df["text"] = df["text"].apply(
    lambda x: re.sub(r"\s+", " ", x)
)


# -----------------------------
# 5. Remove empty text
# -----------------------------
df = df[df["text"] != ""]


# -----------------------------
# 6. Remove exact duplicate rows
# -----------------------------
df = df.drop_duplicates()


# -----------------------------
# 7. Remove duplicate legal
#    statements based on text
# -----------------------------
df = df.drop_duplicates(subset=["text"])


# -----------------------------
# 8. Reset index
# -----------------------------
df = df.reset_index(drop=True)


# -----------------------------
# 9. Save cleaned dataset
# -----------------------------
df.to_excel(output_file, index=False)


# -----------------------------
# 10. Final verification
# -----------------------------
print("\n========== AFTER CLEANING ==========")
print("Total rows:", len(df))

# print("\nMissing values:")
# print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nDuplicate texts:", df["text"].duplicated().sum())

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nCleaned dataset saved to:")
print(output_file)