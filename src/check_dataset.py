import pandas as pd

df = pd.read_excel("dataset/legal_domain_dataset.xlsx")

print("Total rows:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nDuplicate texts:", df["text"].duplicated().sum())

print("\nClass distribution:")
print(df["label"].value_counts())