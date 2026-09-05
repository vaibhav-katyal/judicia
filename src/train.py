import numpy as np

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ==========================================
# 1. Configuration
# ==========================================

MODEL_NAME = "law-ai/InLegalBERT"

NUM_LABELS = 8

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


# ==========================================
# 2. Load Dataset
# ==========================================

dataset = load_dataset(
    "csv",
    data_files={
        "train": "dataset/train_encoded.csv",
        "validation": "dataset/validation_encoded.csv",
        "test": "dataset/test_encoded.csv"
    }
)

print(dataset)


# ==========================================
# 3. Tokenizer
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True
)


# ==========================================
# 4. Load Model
# ==========================================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    label2id=label2id,
    id2label=id2label
)


# ==========================================
# 5. Metrics
# ==========================================

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted"
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ==========================================
# 6. Training Arguments
# ==========================================

training_args = TrainingArguments(

    output_dir="./models/judicia-domain",

    num_train_epochs=3,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    learning_rate=2e-5,

    weight_decay=0.01,

    eval_strategy="epoch",

    save_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    logging_dir="./logs",

    logging_steps=50,

    report_to="none"
)


# ==========================================
# 7. Trainer
# ==========================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_dataset["train"],

    eval_dataset=tokenized_dataset["validation"],

    processing_class=tokenizer,

    compute_metrics=compute_metrics
)


# ==========================================
# 8. Train
# ==========================================

print("\nStarting training...\n")

trainer.train()


# ==========================================
# 9. Save Best Model
# ==========================================

trainer.save_model("./models/judicia-domain")

tokenizer.save_pretrained("./models/judicia-domain")

print("\nTraining complete!")
print("Model saved to: ./models/judicia-domain")