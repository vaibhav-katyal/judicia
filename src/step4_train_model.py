import os
import json
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

"""
=============================================================================
STAGE 4: TRANSFORMER MODEL FINE-TUNING PIPELINE
Project: Judicia — Legal Domain Classification System
=============================================================================
This script fine-tunes InLegalBERT (law-ai/InLegalBERT) for sequence classification:
1. Loads encoded train, validation, and test sets.
2. Tokenizes text sequences using InLegalBERT WordPiece tokenizer (max_length=128).
3. Fine-tunes Transformer weights with Cross-Entropy Loss & AdamW optimizer.
4. Evaluates weighted F1-Score, Precision, Recall, and Accuracy.
5. Exports best model weights & tokenizer to models/judicia-domain-model/
"""

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "dataset", "processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "judicia-domain-model")
MODEL_NAME = "law-ai/InLegalBERT"

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

def train_model():
    print("=" * 75)
    print("STAGE 4: TRANSFORMER MODEL FINE-TUNING (InLegalBERT)")
    print("=" * 75)

    train_file = os.path.join(PROCESSED_DIR, "train_encoded.csv")
    val_file = os.path.join(PROCESSED_DIR, "validation_encoded.csv")
    test_file = os.path.join(PROCESSED_DIR, "test_encoded.csv")
    mapping_file = os.path.join(PROCESSED_DIR, "label_mapping.json")

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Encoded train file not found at {train_file}! Run step3_label_encoding.py first.")

    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    label2id = mapping["label2id"]
    id2label = {int(k): v for k, v in mapping["id2label"].items()}
    num_labels = len(label2id)

    print(f"Loaded {num_labels} legal domain classes from: dataset/processed/label_mapping.json")

    # 1. Load HuggingFace Dataset
    dataset = load_dataset(
        "csv",
        data_files={
            "train": train_file,
            "validation": val_file,
            "test": test_file
        }
    )

    # 2. Initialize Tokenizer & Tokenize Dataset
    print(f"Initializing InLegalBERT Tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # 3. Load Pretrained Sequence Classification Model
    print(f"Loading Base Model Weights: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label
    )

    # 4. Define Training Arguments
    checkpoint_dir = os.path.join(PROJECT_ROOT, "models", "checkpoint_runs")
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
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
        logging_steps=50,
        report_to="none"
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )

    # 6. Train Model
    print("\nStarting InLegalBERT Fine-Tuning Process...")
    trainer.train()

    # 7. Save Model & Tokenizer
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("-" * 75)
    print(f"Training Complete! Model & Tokenizer saved to: models/judicia-domain-model/\n")

if __name__ == "__main__":
    train_model()
