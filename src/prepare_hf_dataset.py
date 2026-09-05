from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "law-ai/InLegalBERT"

# Load datasets
dataset = load_dataset(
    "csv",
    data_files={
        "train": "dataset/train_encoded.csv",
        "validation": "dataset/validation_encoded.csv",
        "test": "dataset/test_encoded.csv"
    }
)

print(dataset)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# Tokenize all datasets
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True
)

print("\nTokenized dataset:")
print(tokenized_dataset)

print("\nFirst example:")
print(tokenized_dataset["train"][0])