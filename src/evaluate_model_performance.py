import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

"""
=============================================================================
STAGE 5: MODEL EVALUATION & METRICS PERFORMANCE REPORT
Project: Judicia — Legal Domain Classification System
=============================================================================
This script evaluates the trained InLegalBERT model on Validation and Test datasets:
1. Calculates Test Accuracy & Validation Accuracy
2. Calculates Weighted & Macro Precision, Recall, F1-Score
3. Prints Scikit-Learn Classification Report
4. Plots & Saves 8x8 Confusion Matrix Heatmap
5. Plots & Saves Per-Class Performance Breakdown Bar Chart
"""

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "judicia-domain-model")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "dataset", "processed")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "reports", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def evaluate_dataset(model, tokenizer, df, device, batch_size=16):
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    
    preds = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            preds.extend(batch_preds)
            
    return np.array(preds), np.array(labels)

def run_evaluation():
    print("=" * 80)
    print(" JUDICIA ML SYSTEM — FULL MODEL EVALUATION & METRICS REPORT")
    print("=" * 80)

    # 1. Load Model and Tokenizer
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model path not found at {MODEL_PATH}!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Model from: {MODEL_PATH} (Device: {device})")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    # 2. Load Label Mapping
    mapping_path = os.path.join(PROCESSED_DIR, "label_mapping.json")
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    
    id2label = {int(k): v for k, v in mapping["id2label"].items()}
    class_names = [id2label[i] for i in range(len(id2label))]

    # 3. Load Validation and Test Datasets
    val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "validation_encoded.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test_encoded.csv"))

    print(f"Loaded Validation Set ({len(val_df)} samples) and Test Set ({len(test_df)} samples).")

    # 4. Evaluate Validation Set
    val_preds, val_labels = evaluate_dataset(model, tokenizer, val_df, device)
    val_acc = accuracy_score(val_labels, val_preds)

    # 5. Evaluate Test Set
    test_preds, test_labels = evaluate_dataset(model, tokenizer, test_df, device)
    test_acc = accuracy_score(test_labels, test_preds)

    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average="weighted")

    print("\n" + "=" * 80)
    print(" OVERALL METRICS SUMMARY")
    print("=" * 80)
    print(f"  Validation Accuracy  : {val_acc * 100:.2f}%")
    print(f"  Test Set Accuracy    : {test_acc * 100:.2f}%")
    print(f"  Weighted Precision   : {precision * 100:.2f}%")
    print(f"  Weighted Recall      : {recall * 100:.2f}%")
    print(f"  Weighted F1-Score    : {f1 * 100:.2f}%")

    print("\n" + "=" * 80)
    print(" PER-CLASS CLASSIFICATION REPORT")
    print("=" * 80)
    report = classification_report(test_labels, test_preds, target_names=class_names, digits=4)
    print(report)

    # 6. Plot & Save Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted Legal Domain", fontweight="bold")
    ax.set_ylabel("True Legal Domain", fontweight="bold")
    ax.set_title(f"Judicia InLegalBERT Model — Confusion Matrix (Test Accuracy: {test_acc*100:.1f}%)", pad=15)
    plt.xticks(rotation=30)
    plt.tight_layout()
    cm_path = os.path.join(PLOTS_DIR, "05_eval_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    print(f"Confusion Matrix heatmap saved to: {cm_path}")
    plt.close()

    # 7. Plot & Save Per-Class Metrics Bar Chart
    per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(test_labels, test_preds, average=None)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(class_names))
    width = 0.25
    ax.bar(x - width, per_class_p, width, label="Precision", color="#3498db")
    ax.bar(x, per_class_r, width, label="Recall", color="#2ecc71")
    ax.bar(x + width, per_class_f1, width, label="F1-Score", color="#9b59b6")
    ax.set_ylabel("Score (0.0 to 1.0)", fontweight="bold")
    ax.set_title("Test Set Evaluation Metrics by Legal Domain")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=15)
    ax.set_ylim(0.7, 1.05)
    ax.legend(loc="lower right")
    plt.tight_layout()
    metrics_path = os.path.join(PLOTS_DIR, "06_eval_per_class_metrics.png")
    plt.savefig(metrics_path, dpi=300)
    print(f"Per-class metrics plot saved to: {metrics_path}")
    plt.close()

    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_evaluation()
