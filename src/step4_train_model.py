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

# Configuration Parameters
MODEL_NAME = "law-ai/InLegalBERT"
OUTPUT_DIR = "models/judicia-domain-model"

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
    print("=" * 70)
    print("STAGE 4: TRANSFORMER MODEL FINE-TUNING (InLegalBERT)")
    print("=" * 70)

    # 1. Locate encoded datasets & mapping
    base_dir = "dataset/processed" if os.path.exists("dataset/processed/train_encoded.csv") else "dataset"
    
    train_file = os.path.join(base_dir, "train_encoded.csv")
    val_file = os.path.join(base_dir, "validation_encoded.csv")
    test_file = os.path.join(base_dir, "test_encoded.csv")
    mapping_file = os.path.join(base_dir, "label_mapping.json")

    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    
    label2id = mapping["label2id"]
    id2label = {int(k): v for k, v in mapping["id2label"].items()}
    num_labels = len(label2id)

    print(f"Loaded {num_labels} classes from {mapping_file}")
    print(f"Dataset path: {base_dir}/")

    # 2. Load HuggingFace Dataset
    dataset = load_dataset(
        "csv",
        data_files={
            "train": train_file,
            "validation": val_file,
            "test": test_file
        }
    )

    # 3. Initialize Tokenizer & Tokenize Dataset
    print(f"Initializing Tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # 4. Load Pretrained Sequence Classification Model
    print(f"Loading Base Model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label
    )

    # 5. Define Training Arguments
    training_args = TrainingArguments(
        output_dir="./models/checkpoint_runs",
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

    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )

    # 7. Train Model
    print("\nStarting InLegalBERT Fine-Tuning Process...")
    trainer.train()

    # 8. Save Model & Tokenizer
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("-" * 70)
    print(f"Training Complete! Model & Tokenizer saved to: {OUTPUT_DIR}\n")

if __name__ == "__main__":
    train_model()
