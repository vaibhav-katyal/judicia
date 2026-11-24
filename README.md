# ⚖️ Judicia — Legal First Aid with RAG

Judicia is an India-focused legal first-aid assistant. A user describes what happened,
and Judicia:

1. **Classifies** the question into one of eight legal domains with a fine-tuned
   InLegalBERT model (`models/judicia-domain-model`).
2. **Retrieves** the closest statutory provisions and precedents using **Pinecone**
   vector search (with a local TF-IDF fallback).
3. **Drafts** a plain-language, step-by-step first-aid plan using **Groq**, grounded
   only in the retrieved sources (with a rule-based fallback).

The UI shows the answer, the exact sources, the classifier distribution, and which
engine handled retrieval and drafting.

> Judicia gives legal information and first-aid guidance, not legal advice.
> Call **112** if there is immediate danger.

---

## 1. What you need

### Accounts / keys

| Service | What to bring | Where to get it |
| --- | --- | --- |
| **Pinecone** | API key. The index is created automatically. | https://app.pinecone.io → API Keys |
| **Groq** | API key | https://console.groq.com/keys |

Pinecone's free tier and Groq's free tier are enough for development and a demo.

### Local tools

- **Python 3.10+** (tested on 3.12)
- **Node.js 20+** and npm
- Internet access for Pinecone and Groq
- The model files in `models/judicia-domain-model/`:
  - `config.json`
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `model.safetensors` (~418 MB — this file is git-ignored, so keep a local copy)

The classifier is loaded **inside the backend**. You do not start it as a separate
service or script.

---

## 2. Add the API keys

From the project root:

```powershell
Copy-Item backend\.env.example backend\.env
notepad backend\.env
```

Fill in the two values:

```env
PINECONE_API_KEY=pcsk_your_real_key
GROQ_API_KEY=gsk_your_real_key
```

Leave these defaults alone unless you have a reason to change them:

```env
PINECONE_INDEX_NAME=judicia-legal
PINECONE_NAMESPACE=legal-corpus
PINECONE_EMBEDDING_MODEL=multilingual-e5-large
PINECONE_EMBEDDING_DIMENSION=1024
GROQ_MODEL=llama-3.3-70b-versatile
RAG_TOP_K=6
```

`backend/.env` is already git-ignored. Never commit it or share the keys.

> The app still starts without keys: retrieval falls back to a local index and
> drafting falls back to a rule-based playbook. The header will say
> **“Local fallback mode”** instead of **“AI engine ready.”**

---

## 3. Install dependencies

### Backend

```powershell
python -m pip install -r backend\requirements.txt
```

This installs the classifier stack (`torch`, `transformers`), Flask, Pinecone, Groq,
and python-dotenv.

### Frontend

```powershell
cd frontend
npm install
cd ..
```

---

## 4. Seed Pinecone (first time only)

```powershell
python backend\ingest_pinecone.py
```

What this does:

- Reads `backend/legal_corpus.json` (25 provisions).
- Creates the `judicia-legal` index if it does not exist (1024 dimensions, cosine,
  AWS `us-east-1`).
- Embeds every provision with Pinecone hosted inference.
- Upserts the vectors into the `legal-corpus` namespace.
- Prints the final vector count.

Useful commands:

```powershell
# See what would be embedded, without calling Pinecone
python backend\ingest_pinecone.py --dry-run

# Rebuild the index after changing the corpus or embedding model
python backend\ingest_pinecone.py --recreate
```

You do **not** need to create the index manually in the Pinecone dashboard.

---

## 5. Start the backend (classifier + RAG API)

From the project root:

```powershell
python backend\app.py
```

Expected output:

```text
[Judicia] Loading domain classifier...
[Judicia] Model: ...\models\judicia-domain-model
[Judicia] Classifier ready (8 labels).
[Judicia] Initialising retrieval + generation engine...
[Judicia] RAG ready with 25 provisions. Pinecone=configured, Groq=configured.
[Judicia] API listening on http://localhost:8000
```

Check it:

- Health: http://localhost:8000/api/health
- API docs: http://localhost:8000/
- Ask endpoint: `POST http://localhost:8000/api/ask`

The health response tells you whether Pinecone and Groq are configured, and which
retrieval engine was used last.

---

## 6. Start the frontend

Open a **second terminal**:

```powershell
cd frontend
npm run dev
```

Open:

- http://localhost:5173

Vite already proxies `/api` to `http://localhost:8000` (see `frontend/vite.config.ts`).
If you deploy the frontend elsewhere, set `VITE_API_BASE_URL` to the backend URL.

---

## 7. Verify the full RAG path

1. The header should show **“AI engine ready”** with three green dots.
   - Classifier = local InLegalBERT model
   - Pinecone = key configured
   - Groq = key configured
2. Ask: *“My landlord refuses to return my rental security deposit.”*
3. You should see:
   - **Retrieved: 5 sources · Pinecone**
   - **Drafted: llama-3.3-70b-versatile**
   - Source cards with citations and “The sources behind this answer.”
4. Click a source card to see its snippet, section labels, and match explanation.

If any service is down, the answer still appears but the header and answer metadata
show the fallback used.

---

## 8. API

### `POST /api/ask`

Request:

```json
{ "query": "Someone hacked my Instagram account and changed my password." }
```

Response shape (abridged):

```json
{
  "query": "...",
  "domain": "Cyber",
  "confidence": 0.95,
  "probabilities": { "Cyber": 0.95, "Criminal": 0.02 },
  "answer": {
    "summary": "...",
    "urgency": "urgent",
    "immediate_steps": ["..."],
    "rights": ["..."],
    "options": ["..."],
    "documents": ["..."],
    "where_to_complain": [
      { "name": "National Cyber Crime Helpline", "detail": "...", "phone": "1930", "link": null }
    ],
    "key_deadlines": ["..."],
    "disclaimer": "..."
  },
  "sources": [
    {
      "id": "cy-101",
      "chunk_id": "cy-101::main",
      "domain": "Cyber",
      "category": "Statute",
      "title": "Section 66C, Information Technology Act, 2000",
      "citation": "IT Act 2000, s.66C",
      "snippet": "...",
      "passages": ["..."],
      "match": 0.98,
      "score": 0.91,
      "why": "Semantic match in the Cyber domain..."
    }
  ],
  "meta": {
    "classifier": { "model": "judicia-domain-model", "latency_ms": 120 },
    "retrieval": { "engine": "pinecone", "index": "judicia-legal", "latency_ms": 400 },
    "llm": { "used": true, "model": "llama-3.3-70b-versatile", "latency_ms": 1800 },
    "corpus_items": 25,
    "latency_ms": 2400
  },
  "generated_at": "2026-11-24T17:00:00Z"
}
```

### `POST /api/predict`

Backwards-compatible alias for `/api/ask` (older frontend builds).

### `GET /api/health`

Returns classifier status, Pinecone/Groq configuration status, last retrieval engine,
and corpus size. Safe to expose — it never returns keys.

---

## 9. How the fallbacks work

| Component | Primary | Fallback |
| --- | --- | --- |
| Domain classifier | `models/judicia-domain-model` | No fallback; the model must be present |
| Retrieval | Pinecone hosted embeddings + vector search | Local TF-IDF + keyword index over `legal_corpus.json` |
| Drafting | Groq chat completion (JSON mode) | Rule-based domain playbook in `backend/legal_knowledge.py` |

The API stays up even if Pinecone or Groq is unreachable. The UI labels which mode
was used, and the answer's `meta` object explains any fallback reason.

---

## 10. Troubleshooting

| Symptom | Fix |
| --- | --- |
| Header says **Engine offline** | The backend is not running. Start `python backend/app.py` and check http://localhost:8000/api/health |
| Header says **Local fallback mode** | `PINECONE_API_KEY` / `GROQ_API_KEY` are missing or still placeholders. Edit `backend/.env`, then **restart the backend** |
| `PINECONE_API_KEY is missing` during ingest | Copy `backend/.env.example` to `backend/.env` and add the real key |
| `The Pinecone index was not found` | Run `python backend/ingest_pinecone.py` |
| Pinecone 401/403 | Wrong key, or the key belongs to a different Pinecone project |
| Groq 401/403 | Wrong key or no access to the configured model |
| Groq model decommissioned | Change `GROQ_MODEL` in `backend/.env`, or rely on `GROQ_FALLBACK_MODELS` |
| Port 8000 already in use | Stop the other process or start with `$env:PORT=8001; python backend/app.py` (then update the Vite proxy) |
| Frontend shows no results | Check the browser console and the backend terminal; make sure `/api/ask` returns 200 |
| `Could not find the judicia-domain-model directory` | Run from the project root and keep the `models/judicia-domain-model/` files in place |
| Slow first request | The InLegalBERT model loads at backend startup; subsequent requests are faster |

---

## 11. Project structure (updated)

```text
test-model/
├── backend/
│   ├── app.py                  # Flask API: classifier + RAG endpoint
│   ├── rag_engine.py           # Pinecone retrieval, Groq generation, fallbacks
│   ├── legal_knowledge.py      # corpus loading, chunks, playbooks, fallback answers
│   ├── ingest_pinecone.py      # one-time Pinecone index creation + ingestion
│   ├── legal_corpus.json       # 25 statutory provisions / precedents
│   ├── inference.py            # standalone classifier CLI
│   ├── stress_test.py          # classifier evaluation
│   ├── requirements.txt
│   └── .env.example            # copy to .env; never commit .env
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx             # landing page, loading/error states, header status
│   │   ├── components/
│   │   │   ├── Composer.tsx    # question box
│   │   │   ├── ResultsView.tsx # plan, sources, probability panel, copy/print
│   │   │   └── Icons.tsx
│   │   ├── lib/judicia-data.ts # typed API client + response normalisation
│   │   └── index.css           # design system
│   └── vite.config.ts          # /api proxy -> localhost:8000
│
├── models/judicia-domain-model/
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── model.safetensors       # ~418 MB, keep locally (git-ignored)
│
├── dataset/                    # training data
├── src/                        # ML training pipeline
└── notebooks/                  # EDA and evaluation notebooks
```

---

## 12. Security notes

- Keep `backend/.env` private. It is git-ignored.
- The frontend never receives the keys.
- The `/api/health` endpoint reports only whether services are configured, not secrets.
- Do not paste real keys into source files, screenshots, issues, or chat.
- For production, put the backend behind HTTPS, add rate limiting, and restrict CORS
  to your frontend origin.
