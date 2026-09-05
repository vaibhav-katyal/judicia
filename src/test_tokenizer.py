from transformers import AutoTokenizer

MODEL_NAME = "law-ai/InLegalBERT"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

text = "My employer has not paid my salary."

tokens = tokenizer.tokenize(text)

print("Original:")
print(text)

print("\nTokens:")
print(tokens)

encoded = tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=128
)

print("\nInput IDs:")
print(encoded["input_ids"])

print("\nAttention Mask:")
print(encoded["attention_mask"])