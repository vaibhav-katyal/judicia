import os
import json
import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# 1. Locate judicia-domain-model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSSIBLE_PATHS = [
    os.path.join(BASE_DIR, "models", "judicia-domain-model"),
    os.path.join(BASE_DIR, "model", "judicia-domain-model"),
    "models/judicia-domain-model",
    "model/judicia-domain-model",
    r"c:\Users\HIMANKSHA\Desktop\test-model\models\judicia-domain-model"
]

MODEL_PATH = None
for path in POSSIBLE_PATHS:
    if os.path.exists(path):
        MODEL_PATH = path
        break

if not MODEL_PATH:
    raise FileNotFoundError("Could not find judicia-domain-model directory!")

print(f"Loading Judicia BERT Model from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
print("Judicia Model loaded into memory successfully!")

# 2. Load Legal Corpus Database for RAG Retrieval Engine
CORPUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "legal_corpus.json")
if os.path.exists(CORPUS_PATH):
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        LEGAL_CORPUS = json.load(f)
    print(f"Loaded {len(LEGAL_CORPUS)} statutory sections/precedents into Legal Corpus.")
else:
    LEGAL_CORPUS = []
    print("Warning: legal_corpus.json not found!")

# 3. Build TF-IDF Vector Index for Semantic RAG Search
corpus_texts = []
for item in LEGAL_CORPUS:
    # Combine title, snippet, passages, and keywords for rich semantic indexing
    combined_text = f"{item['title']} {item['snippet']} {' '.join(item['passages'])} {item.get('keywords', '')}"
    corpus_texts.append(combined_text)

tfidf_vectorizer = TfidfVectorizer(stop_words="english")
if corpus_texts:
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_texts)
    print("TF-IDF Vector Index initialized successfully!")
else:
    tfidf_matrix = None

def retrieve_dynamic_passages(query: str, predicted_domain: str, model_confidence: float):
    """
    RAG Engine: Computes semantic cosine similarity between query vector
    and corpus items, filtered and ranked dynamically for the user query.
    """
    if not LEGAL_CORPUS or tfidf_matrix is None:
        return []

    # Vectorize user query
    query_vec = tfidf_vectorizer.transform([query])
    
    # Calculate similarity against all corpus items
    sim_scores = cosine_similarity(query_vec, tfidf_matrix)[0]

    # Rank and score corpus items
    scored_items = []
    for idx, item in enumerate(LEGAL_CORPUS):
        raw_sim = float(sim_scores[idx])
        domain_match = (item["domain"].lower() == predicted_domain.lower())
        
        # Boost score if domain matches predicted ML domain
        final_score = raw_sim + (0.35 if domain_match else 0.0)
        
        scored_items.append({
            "item": item,
            "raw_sim": raw_sim,
            "domain_match": domain_match,
            "final_score": final_score
        })

    # Sort descending by final score
    scored_items.sort(key=lambda x: x["final_score"], reverse=True)

    # Return top 2-3 matching results
    results = []
    for entry in scored_items[:3]:
        item = entry["item"]
        score = entry["final_score"]
        # Format match percentage
        match_pct = min(0.99, max(0.60, round(score if entry["domain_match"] else score * 0.7, 2)))

        results.append({
            "id": item["id"],
            "category": item["category"],
            "jurisdiction": item["jurisdiction"],
            "year": item["year"],
            "title": item["title"],
            "citation": item["citation"],
            "snippet": item["snippet"],
            "passages": item["passages"],
            "why": f"[Dynamic Semantic Vector Match: {match_pct * 100:.0f}%] Predicted Domain '{predicted_domain}' ({model_confidence * 100:.1f}% confidence)",
            "match": match_pct
        })

    return results

@app.route("/api/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    text = data.get("query", "").strip()

    if not text:
        return jsonify({"error": "Query string is required"}), 400

    # Pass text through fine-tuned BERT Model
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)[0]
    prediction_idx = torch.argmax(outputs.logits, dim=1).item()

    if prediction_idx in model.config.id2label:
        domain = model.config.id2label[prediction_idx]
    else:
        domain = model.config.id2label[str(prediction_idx)]

    confidence = float(probabilities[prediction_idx].item())

    # Build dictionary of all class probabilities
    all_probs = {}
    for idx, prob in enumerate(probabilities):
        label = model.config.id2label.get(idx) or model.config.id2label.get(str(idx))
        all_probs[label] = float(prob.item())

    # Dynamic RAG Retrieval of Legal Sections & Precedents
    dynamic_results = retrieve_dynamic_passages(text, domain, confidence)

    return jsonify({
        "query": text,
        "domain": domain,
        "confidence": confidence,
        "probabilities": all_probs,
        "results": dynamic_results
    })

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "model_path": MODEL_PATH,
        "corpus_items": len(LEGAL_CORPUS),
        "rag_engine": "TF-IDF + Cosine Similarity Vector Index"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Judicia ML + Vector RAG Server on http://localhost:{port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
