# ⚖️ JUDICIA: Intelligent Legal Domain Classification & Vector RAG System

Judicia is an end-to-end Machine Learning system that combines fine-tuned Transformer classification (**InLegalBERT**) with **Dynamic Vector Retrieval-Augmented Generation (RAG)** to classify plain-language legal queries into target legal domains and retrieve governing statutory provisions (including **BNS 2023**, **BSA 2023**, **IPC**, **IT Act 2000**) and court precedents.

---

## 📁 Structured Project Directory

```text
test-model/
├── dataset/                      # Dataset Storage & Data Splits
│   ├── raw/
│   │   └── legal_domain_dataset.xlsx  # Raw uncleaned dataset (4,000 samples)
│   ├── processed/
│   │   ├── legal_domain_dataset_cleaned.xlsx
│   │   ├── train.csv             # 80% Training Split (3,200 samples)
│   │   ├── validation.csv        # 10% Validation Split (400 samples)
│   │   ├── test.csv              # 10% Test Split (400 samples)
│   │   ├── train_encoded.csv     # Integer-encoded training dataset
│   │   ├── validation_encoded.csv
│   │   ├── test_encoded.csv
│   │   └── label_mapping.json    # Bidirectional Label-ID dictionary (0..7)
│
├── src/                          # Machine Learning & Data Pipeline (Numbered Steps)
│   ├── step1_data_cleaning.py    # 1. Missing values, text whitespace & deduplication
│   ├── step2_dataset_splitting.py# 2. Stratified 80/10/10 Train-Val-Test split
│   ├── step3_label_encoding.py   # 3. Text label to numeric ID encoding & JSON mapping
│   ├── step4_train_model.py      # 4. InLegalBERT Transformer fine-tuning script
│   └── run_pipeline.py           # Master ETL Pipeline Runner
│
├── backend/                      # Python Flask API & Evaluation Engine
│   ├── app.py                    # Production Flask API + ML Inference + Vector RAG Engine
│   ├── inference.py              # Interactive CLI Inference Tool
│   ├── stress_test.py            # Model Evaluation, Confusion Matrix & Metrics Report
│   ├── legal_corpus.json         # Statutory Knowledgebase (BNS 2023, BSA 2023, IPC, IT Act)
│   └── requirements.txt          # Backend dependencies
│
├── models/                       # Model Weights & Artifacts
│   └── judicia-domain-model/     # Fine-tuned Transformer & Tokenizer (.safetensors)
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── tokenizer_config.json
│
├── frontend/                     # Vite React + TypeScript Web Application
│   ├── src/
│   │   ├── App.tsx               # Interactive Legal AI UI & Confidence Visualizer
│   │   └── lib/judicia-data.ts   # API Client & Backend Connection Layer
│   ├── vite.config.ts            # Proxy configuration to API server (port 8000)
│   └── package.json
│
└── .gitignore                    # Version control rules
```

---

## 🔄 Sequential Data Pipeline

### 1. Data Cleaning (`src/step1_data_cleaning.py`)
- Removes completely empty rows and missing text/label fields.
- Normalizes whitespace (`re.sub(r'\s+', ' ', text)`) and trims text strings.
- Eliminates exact duplicate rows and duplicate text statements to prevent data leakage.

### 2. Stratified Splitting (`src/step2_dataset_splitting.py`)
- Splits data into **80% Training**, **10% Validation**, and **10% Testing**.
- Enforces equal distribution across all 8 target legal domains (`stratify=df['label']`).

### 3. Label Encoding (`src/step3_label_encoding.py`)
- Maps categorical labels (`Civil`, `Constitutional`, `Consumer`, `Criminal`, `Cyber`, `Family`, `Labour`, `Property`) into numerical class IDs `0..7`.
- Exports `label_mapping.json`.

### 4. Transformer Training (`src/step4_train_model.py`)
- Fine-tunes **InLegalBERT** (`law-ai/InLegalBERT`) sequence classifier.
- Saves fine-tuned model weights in `.safetensors` format in `models/judicia-domain-model/`.

---

## 🚀 How to Run the Project

### Option A: Run the Preprocessing Pipeline
```bash
python src/run_pipeline.py
```

### Option B: Start the Backend Model & Vector RAG Server
```bash
python backend/app.py
```
*(Runs on `http://localhost:8000`)*

### Option C: Launch the Frontend Web UI
```bash
cd frontend
npm run dev
```
*(Runs on `http://localhost:5173`)*

### Option D: Run Real-World Stress Test Evaluation
```bash
python backend/stress_test.py
```
*(Generates Precision, Recall, F1-Score, and Confusion Matrix across 50 test cases)*
