from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "law-ai/InLegalBERT"

label2id = {
    "Civil": 0,
    "Constitutional": 1,
    "Consumer": 2,
    "Criminal": 3,
    "Cyber": 4,
    "Family": 5,
    "Labour": 6,
    "Property": 7
}

id2label = {
    0: "Civil",
    1: "Constitutional",
    2: "Consumer",
    3: "Criminal",
    4: "Cyber",
    5: "Family",
    6: "Labour",
    7: "Property"
}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=8,
    label2id=label2id,
    id2label=id2label
)

print("Model loaded successfully!")
print("Number of labels:", model.config.num_labels)