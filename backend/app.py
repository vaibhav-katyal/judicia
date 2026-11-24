"""
Judicia backend API.

Endpoints
---------
GET  /api/health   Service + RAG status (safe to expose)
POST /api/ask      Classify a legal question, retrieve law, generate a first-aid plan
POST /api/predict  Backwards-compatible alias for /api/ask

The classifier (InLegalBERT) always runs locally. Retrieval uses Pinecone when
configured and falls back to a local TF-IDF index. Drafting uses Groq when
configured and falls back to a rule-based playbook. The API therefore stays
usable even before API keys are added.
"""

from __future__ import annotations

import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rag_engine import LegalRAG

APP_VERSION = "2.0.0"
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


def _find_model_path() -> Path:
    candidates = [
        Path(os.environ.get("JUDICIA_MODEL_PATH", "")).expanduser()
        if os.environ.get("JUDICIA_MODEL_PATH")
        else None,
        PROJECT_ROOT / "models" / "judicia-domain-model",
        PROJECT_ROOT / "model" / "judicia-domain-model",
        BASE_DIR / "models" / "judicia-domain-model",
        Path("models/judicia-domain-model"),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate.resolve()
    searched = "\n".join(f"  - {candidate}" for candidate in candidates if candidate)
    raise FileNotFoundError(
        "Could not find the judicia-domain-model directory. Searched:\n" + searched
    )


class JudiciaClassifier:
    """Thin, defensive wrapper around the fine-tuned InLegalBERT classifier."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model_name = model_path.name
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        self.model.eval()

        raw_labels = getattr(self.model.config, "id2label", {}) or {}
        self.id2label: dict[int, str] = {}
        for key, value in raw_labels.items():
            try:
                self.id2label[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        if not self.id2label:
            self.id2label = {
                0: "Civil",
                1: "Constitutional",
                2: "Consumer",
                3: "Criminal",
                4: "Cyber",
                5: "Family",
                6: "Labour",
                7: "Property",
            }
        self.labels = [self.id2label[index] for index in sorted(self.id2label)]

    def predict(self, text: str) -> tuple[str, float, dict[str, float]]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0].tolist()

        best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        domain = self.id2label.get(best_index, f"Class {best_index}")
        confidence = float(probabilities[best_index])

        all_probabilities = {
            self.id2label.get(index, f"Class {index}"): float(score)
            for index, score in enumerate(probabilities)
        }
        return domain, confidence, all_probabilities


print("[Judicia] Loading domain classifier...")
MODEL_PATH = _find_model_path()
print(f"[Judicia] Model: {MODEL_PATH}")
classifier = JudiciaClassifier(MODEL_PATH)
print(f"[Judicia] Classifier ready ({len(classifier.labels)} labels).")

print("[Judicia] Initialising retrieval + generation engine...")
rag = LegalRAG(corpus_path=BASE_DIR / "legal_corpus.json")
print(
    f"[Judicia] RAG ready with {len(rag.corpus)} provisions. "
    f"Pinecone={'configured' if rag.config.pinecone_api_key else 'local fallback'}, "
    f"Groq={'configured' if rag.config.groq_api_key else 'rule-based fallback'}."
)

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB is plenty for a question


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _error(message: str, status: int = 400):
    return jsonify({"error": message}), status


@app.get("/")
def index():
    return jsonify(
        {
            "name": "Judicia API",
            "version": APP_VERSION,
            "endpoints": ["/api/health", "/api/ask", "/api/predict"],
            "docs": "POST /api/ask with {\"query\": \"...\"}",
        }
    )


@app.get("/api/health")
def health():
    rag_status = rag.status()
    return jsonify(
        {
            "status": "online",
            "version": APP_VERSION,
            "classifier": {
                "loaded": True,
                "model": classifier.model_name,
                "labels": classifier.labels,
            },
            "rag": rag_status,
            "corpus_items": len(rag.corpus),
            "timestamp": _utc_now(),
        }
    )


def _run_pipeline(payload: dict[str, Any]):
    query = str(payload.get("query", "")).strip()
    if not query:
        return _error("Please describe your legal situation in the 'query' field.")
    if len(query) > 2000:
        return _error("Please keep your question under 2,000 characters.")

    total_started = time.perf_counter()

    classify_started = time.perf_counter()
    try:
        domain, confidence, probabilities = classifier.predict(query)
    except Exception as exc:
        traceback.print_exc()
        return _error(f"The domain classifier could not process this question: {exc}", 500)
    classify_ms = round((time.perf_counter() - classify_started) * 1000, 1)

    try:
        sources, retrieval_meta = rag.search(query, domain, confidence)
    except Exception as exc:
        traceback.print_exc()
        sources = []
        retrieval_meta = {
            "engine": "none",
            "reason": f"Retrieval failed: {exc}",
            "candidates": 0,
            "latency_ms": 0,
        }

    try:
        answer, llm_meta = rag.answer(query, domain, confidence, sources)
    except Exception as exc:
        traceback.print_exc()
        # This should not happen because the engine has its own fallback, but
        # never leave the user without an answer.
        answer = {
            "summary": (
                "We could not generate a detailed plan for this question. Please describe "
                "what happened, keep all documents safe, and contact the District Legal "
                "Services Authority for free legal aid."
            ),
            "urgency": "urgent",
            "immediate_steps": [
                "Move to a safe place and call 112 if there is immediate danger.",
                "Write down what happened with dates, names, and amounts.",
                "Preserve all messages, documents, receipts, and photographs.",
                "Contact the District Legal Services Authority or a qualified advocate.",
            ],
            "rights": [],
            "options": [],
            "documents": [],
            "where_to_complain": [
                {
                    "name": "District Legal Services Authority",
                    "detail": "Free legal aid for eligible persons.",
                    "phone": None,
                    "link": "https://nalsa.gov.in",
                }
            ],
            "key_deadlines": [],
            "disclaimer": (
                "Judicia provides legal information, not legal advice. Consult a qualified "
                "advocate for your specific situation. Call 112 in an emergency."
            ),
        }
        llm_meta = {
            "used": False,
            "model": "rule-based playbook",
            "reason": f"Generation failed: {exc}",
            "latency_ms": 0,
        }

    total_ms = round((time.perf_counter() - total_started) * 1000, 1)

    return jsonify(
        {
            "query": query,
            "domain": domain,
            "confidence": confidence,
            "probabilities": probabilities,
            "answer": answer,
            "sources": sources,
            "meta": {
                "classifier": {
                    "model": classifier.model_name,
                    "labels": classifier.labels,
                    "latency_ms": classify_ms,
                },
                "retrieval": retrieval_meta,
                "llm": llm_meta,
                "corpus_items": len(rag.corpus),
                "latency_ms": total_ms,
            },
            "generated_at": _utc_now(),
        }
    )


@app.post("/api/ask")
def ask():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _error("Send a JSON body like {\"query\": \"...\"}.")
    return _run_pipeline(payload)


@app.post("/api/predict")
@app.post("/predict")
def predict():
    """Backwards-compatible alias retained for older frontend builds."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _error("Send a JSON body like {\"query\": \"...\"}.")
    return _run_pipeline(payload)


@app.errorhandler(404)
def not_found(_error_obj):
    if request.path.startswith("/api/") or request.path == "/predict":
        return jsonify({"error": "Endpoint not found."}), 404
    return jsonify({"error": "Not found."}), 404


@app.errorhandler(405)
def method_not_allowed(_error_obj):
    return jsonify({"error": "Method not allowed for this endpoint."}), 405


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"[Judicia] API listening on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
