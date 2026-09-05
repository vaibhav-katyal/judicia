from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "models/judicia-domain-model"

print("Loading Judicia model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()

print("Model loaded successfully!\n")


while True:

    text = input("Enter your legal query (or type 'exit'): ")

    if text.lower() == "exit":
        break

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    probabilities = torch.softmax(outputs.logits, dim=1)
    confidence = probabilities[0][prediction].item()

    label = model.config.id2label[prediction]

    print("\nPredicted domain:", label)
    print("Confidence:", f"{confidence * 100:.2f}%\n")