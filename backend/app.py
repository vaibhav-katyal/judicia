import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)
CORS(app)

# Locate judicia-domain-model
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

print(f"Loading Judicia Model from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
print("Judicia Model successfully loaded into backend memory!")

# Rich domain context data for UI representation
DOMAIN_PASSAGES = {
    "Cyber": [
        {
            "id": "cy-1",
            "category": "Statute",
            "jurisdiction": "India",
            "year": 2000,
            "title": "Section 66C & 66D, Information Technology Act, 2000",
            "citation": "IT Act 2000, s.66C/66D",
            "snippet": "Punishment for identity theft and cheating by personation by using computer resources, unauthorized account access, phishing, or stealing digital credentials.",
            "passages": ["s.66C - Identity Theft", "s.66D - Cheating by Personation"],
            "why": "Directly governs unauthorized access, hacking, online account compromise, social media impersonation, and phishing."
        },
        {
            "id": "cy-2",
            "category": "Statute",
            "jurisdiction": "India",
            "year": 2000,
            "title": "Section 65B, Indian Evidence Act, 1872",
            "citation": "Indian Evidence Act, 1872, s.65B",
            "snippet": "Any information contained in an electronic record is deemed a document and admissible in evidence subject to conditions and certificate under Section 65B(4).",
            "passages": ["65B(1) - Electronic Record", "65B(4) - Mandatory Certificate"],
            "why": "Governs evidentiary proof for electronic records, server logs, chats, emails, and digital forensics."
        }
    ],
    "Criminal": [
        {
            "id": "cr-1",
            "category": "Statute",
            "jurisdiction": "India",
            "year": 1860,
            "title": "Section 503 & 506, Indian Penal Code",
            "citation": "IPC 1860, s.503/506",
            "snippet": "Criminal intimidation — Whoever threatens another with any injury to his person, reputation, or property with intent to cause alarm.",
            "passages": ["s.503 - Criminal Intimidation", "s.506 - Punishment"],
            "why": "Applies to death threats, verbal threats, harassment, and intentional intimidation."
        },
        {
            "id": "cr-2",
            "category": "Statute",
            "jurisdiction": "India",
            "year": 1860,
            "title": "Section 323 & 351, IPC — Voluntary Hurt & Assault",
            "citation": "IPC 1860, s.323/351",
            "snippet": "Punishment for voluntarily causing hurt and assault by making gestures or preparations to use criminal force.",
            "passages": ["s.323 - Voluntarily Causing Hurt", "s.351 - Assault Defined"],
            "why": "Pertains to physical attack, bodily violence, assault, and armed threat."
        }
    ],
    "Property": [
        {
            "id": "pr-1",
            "category": "Statute",
            "jurisdiction": "India",
            "year": 1882,
            "title": "Section 105 & 108, Transfer of Property Act, 1882",
            "citation": "Transfer of Property Act, 1882, s.105",
            "snippet": "Lease of immovable property, rights and liabilities of lessor and lessee, return of security deposits, and tenancy determination.",
            "passages": ["s.105 - Lease Defined", "s.108 - Rights & Liabilities"],
            "why": "Governs landlord-tenant disputes, non-payment of rent, and security deposit retention."
        },
        {
            "id": "pr-2",
            "category": "Statute",
            "jurisdiction": "India",
            "year": 1963,
            "title": "Section 5 & 6, Specific Relief Act, 1963",
            "citation": "Specific Relief Act, 1963, s.5/6",
            "snippet": "Recovery of specific immovable property and summary suits for persons dispossessed of land or real estate.",
            "passages": ["s.5 - Title-based Recovery", "s.6 - Suit for Dispossessed Person"],
            "why": "Applies to ancestral land claims, illegal boundary encroachment, unauthorized property sale, and partition disputes."
        }
    ],
    "Family": [
        {
            "id": "fa-1",
            "category": "Statute",
            "jurisdiction": "India",
            "year": 1955,
            "title": "Section 13 & 24, Hindu Marriage Act, 1955",
            "citation": "Hindu Marriage Act, 1955, s.13/24",
            "snippet": "Dissolution of marriage by decree of divorce and grant of maintenance pendente lite and expenses of proceedings.",
            "passages": ["s.13 - Grounds for Divorce", "s.24 - Spousal Maintenance"],
            "why": "Governs divorce applications, judicial separation, and spousal maintenance claims."
        },
        {
            "id": "fa-2",
            "category": "Statute",
            "jurisdiction": "India",
            "year": 1956,
            "title": "Section 6 & 13, Hindu Minority & Guardianship Act",
            "citation": "HMG Act 1956, s.6/13",
            "snippet": "Natural guardians of a Hindu minor and welfare of the minor to be the paramount consideration in custody orders.",
            "passages": ["s.6 - Natural Guardians", "s.13 - Welfare of Minor"],
            "why": "Governs child custody disputes, visitation rights, separation arrangements, and adoption."
        }
    ],
    "Labour": [
        {
            "id": "la-1",
            "category": "Statute",
            "jurisdiction": "India",
            "year": 1947,
            "title": "Section 25F, Industrial Disputes Act & Payment of Wages Act",
            "citation": "ID Act 1947, s.25F",
            "snippet": "Conditions precedent to retrenchment of workmen, notice requirements, retrenchment compensation, and recovery of unpaid salary.",
            "passages": ["s.25F - Termination Notice", "s.15 - Wages Claim"],
            "why": "Governs unpaid salary, arbitrary termination without notice, illegal wage deductions, and relieving letter refusal."
        },
        {
            "id": "la-2",
            "category": "Statute",
            "jurisdiction": "India",
            "year": 1961,
            "title": "Section 5 & 12, Maternity Benefit Act, 1961",
            "citation": "Maternity Benefit Act, 1961, s.5/12",
            "snippet": "Right to payment of maternity benefit and prohibition of dismissal during absence on maternity leave.",
            "passages": ["s.5 - Maternity Benefit", "s.12 - Unlawful Dismissal Protection"],
            "why": "Protects employee entitlement to statutory benefits, overtime compliance, and leave rights."
        }
    ],
    "Civil": [
        {
            "id": "ci-1",
            "category": "Statute",
            "jurisdiction": "India",
            "year": 1872,
            "title": "Section 73 & 74, Indian Contract Act, 1872",
            "citation": "Indian Contract Act, 1872, s.73",
            "snippet": "Compensation for loss or damage caused by breach of contract, naturally arising in usual course of things.",
            "passages": ["s.73 - Damages for Breach", "s.74 - Penalty Clause"],
            "why": "Foundational law on remedies for breach of written agreement, service default, and financial recovery."
        },
        {
            "id": "ci-2",
            "category": "Statute",
            "jurisdiction": "India",
            "year": 1908,
            "title": "Order XXXVII, Code of Civil Procedure, 1908",
            "citation": "CPC 1908, O.37",
            "snippet": "Summary procedure for suits upon contracts, bills of exchange, promissory notes, and monetary recovery.",
            "passages": ["O.37 R.1 - Summary Suits Scope", "O.37 R.2 - Leave to Defend"],
            "why": "Governs legal debt recovery, summary suits, and contractual enforcement."
        }
    ],
    "Consumer": [
        {
            "id": "co-1",
            "category": "Statute",
            "jurisdiction": "India",
            "year": 2019,
            "title": "Section 2(11) & 35, Consumer Protection Act, 2019",
            "citation": "Consumer Protection Act, 2019, s.35",
            "snippet": "Manner in which complaint shall be filed before Consumer Commission regarding deficiency in service, defective product, or unfair trade practice.",
            "passages": ["s.2(11) - Deficiency in Service", "s.2(47) - Unfair Trade Practice"],
            "why": "Governs defective e-commerce goods, warranty disputes, wrong charges, cancellation refund denials, and product returns."
        }
    ],
    "Constitutional": [
        {
            "id": "cn-1",
            "category": "Statute",
            "jurisdiction": "India",
            "year": 1950,
            "title": "Articles 14, 19 & 21, Constitution of India",
            "citation": "Constitution of India, Art. 14/19/21",
            "snippet": "Fundamental Rights: Equality before law, protection of freedom of speech, assembly, movement, and protection of life & personal liberty.",
            "passages": ["Art. 14 - Right to Equality", "Art. 19 - Freedom of Speech", "Art. 21 - Personal Liberty"],
            "why": "Applies to government actions restricting fundamental rights, arbitrary state discrimination, and unlawful surveillance."
        }
    ]
}

@app.route("/api/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    text = data.get("query", "").strip()

    if not text:
        return jsonify({"error": "Query string is required"}), 400

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

    matched_passages = DOMAIN_PASSAGES.get(domain, DOMAIN_PASSAGES["Civil"])

    formatted_results = []
    for item in matched_passages:
        formatted_results.append({
            "id": item["id"],
            "category": item["category"],
            "jurisdiction": item["jurisdiction"],
            "year": item["year"],
            "title": item["title"],
            "citation": item["citation"],
            "snippet": item["snippet"],
            "passages": item["passages"],
            "why": f"[ML Model Classification: {domain} ({confidence * 100:.1f}% confidence)] {item['why']}",
            "match": round(confidence, 2)
        })

    return jsonify({
        "query": text,
        "domain": domain,
        "confidence": confidence,
        "probabilities": all_probs,
        "results": formatted_results
    })

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "model_path": MODEL_PATH,
        "architecture": model.config.architectures[0] if hasattr(model.config, 'architectures') else 'BERT'
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Judicia ML Backend Server on http://localhost:{port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
