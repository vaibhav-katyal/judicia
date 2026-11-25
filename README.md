# âš–ï¸ Judicia â€” Legal First Aid with RAG

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
| **Pinecone** | API key. The index is created automatically. | https://app.pinecone.io â†’ API Keys |
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
  - `model.safetensors` (~418 MB â€” this file is git-ignored, so keep a local copy)

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
GROQ_MODEL=openai/gpt-oss-120b
RAG_TOP_K=6
```

`backend/.env` is already git-ignored. Never commit it or share the keys.

> The app still starts without keys: retrieval falls back to a local index and
> drafting falls back to a rule-based playbook. The header will say
> **â€œLocal fallback modeâ€** instead of **â€œAI engine ready.â€**

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
- Extracts and chunks every PDF in `backend/docs/`:
  - **Bare Acts** are split per **section**, so `Provided that` clauses stay with
    their rule, and are tagged `doc_type: "bare_act"`.
  - **Handbooks / SOPs** are split into ~1,000-character procedural chunks and
    tagged `doc_type: "handbook"`.
- Creates the `judicia-legal` index if it does not exist (1024 dimensions, cosine,
  AWS `us-east-1`).
- Embeds every chunk with the same Pinecone hosted model (`multilingual-e5-large`).
- Upserts the vectors into the `legal-corpus` namespace, keeping the existing
  corpus vectors intact.
- Caches the extracted PDF chunks in `backend/docs/pdf_chunks.json` so the local
  fallback can search them when Pinecone is unreachable.
- Prints per-track counts and the final vector count.

Useful commands:

```powershell
# See what would be embedded, without calling Pinecone
python backend\ingest_pinecone.py --dry-run

# Only the PDF tracks (leaves existing corpus vectors untouched)
python backend\ingest_pinecone.py --pdf-only

# Only the JSON corpus (ignores backend/docs)
python backend\ingest_pinecone.py --corpus-only

# Rebuild the index after changing the corpus, PDFs or embedding model
python backend\ingest_pinecone.py --recreate

# Custom folder / manifest / batch size
python backend\ingest_pinecone.py --docs-dir backend\docs --batch-size 32
```

You do **not** need to create the index manually in the Pinecone dashboard.

### Where to put PDFs

Drop files in `backend/docs/` and tag them in `backend/docs_manifest.json` as
either `bare_act` or `handbook`. Unlisted PDFs in `backend/docs/bare_act/` or
`backend/docs/handbook/` are picked up automatically.

The manifest already maps the expected files:

| File | Track | Short code |
| --- | --- | --- |
| `Bharatiya_Nagarik_Suraksha_Sanhita,_2023.pdf` | `bare_act` | `BNSS` |
| `BNS2023.pdf` | `bare_act` | `BNS` |
| `CP Act 2019_1732700731.pdf` | `bare_act` | `CPA 2019` |
| `it_act_2000_updated.pdf` | `bare_act` | `IT Act` |
| `DOMESTICVIOLENCEACT,2005.pdf` | `bare_act` | `PWDVA` |
| `rti_act_2005.pdf` | `bare_act` | `RTI Act` |
| `Legal-Aid-to-Legal-Rights.pdf` | `handbook` | `LEGAL AID` |
| `20250408448855620.pdf` (NALSA SOP) | `handbook` | `NALSA SOP` |

The IT Act PDF is not in the repo yet. Add it and re-run
`python backend\ingest_pinecone.py --pdf-only`; the rest ingest normally without it.
See `backend/docs/README.md` for the full manifest format.


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
[Judicia] RAG ready with 25 provisions and 1987 PDF chunks (Bare Acts=1292, handbooks=695). Pinecone=configured, Groq=configured.
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

1. The header should show **â€œAI engine readyâ€** with three green dots.
   - Classifier = local InLegalBERT model
   - Pinecone = key configured
   - Groq = key configured
2. Ask: *â€œMy landlord refuses to return my rental security deposit.â€*
3. You should see:
   - **Retrieved: 5 sources Â· Pinecone**
   - **Drafted: openai/gpt-oss-120b**
   - Source cards with citations and â€œThe sources behind this answer.â€
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
    "llm": { "used": true, "model": "openai/gpt-oss-120b", "latency_ms": 1800 },
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
and corpus size. Safe to expose â€” it never returns keys.

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
â”œâ”€â”€ backend/
â”‚   â”œâ”€â”€ app.py                  # Flask API: classifier + RAG endpoint
â”‚   â”œâ”€â”€ rag_engine.py           # Pinecone retrieval, Groq generation, fallbacks
â”‚   â”œâ”€â”€ legal_knowledge.py      # corpus loading, chunks, playbooks, fallback answers
â”‚   â”œâ”€â”€ pdf_ingest.py           # PDF extraction + Bare Act / handbook chunking
â”‚   â”œâ”€â”€ docs_manifest.json      # maps PDFs to doc_type, short_code, domain
â”‚   â”œâ”€â”€ docs/                   # source PDFs + pdf_chunks.json cache + README
â”‚   â”œâ”€â”€ ingest_pinecone.py      # Pinecone index creation + corpus/PDF ingestion
â”‚   â”œâ”€â”€ legal_corpus.json       # 25 statutory provisions / precedents
â”‚   â”œâ”€â”€ inference.py            # standalone classifier CLI
â”‚   â”œâ”€â”€ stress_test.py          # classifier evaluation
â”‚   â”œâ”€â”€ requirements.txt
â”‚   â””â”€â”€ .env.example            # copy to .env; never commit .env
â”‚
â”œâ”€â”€ frontend/
â”‚   â”œâ”€â”€ src/
â”‚   â”‚   â”œâ”€â”€ App.tsx             # landing page, loading/error states, header status
â”‚   â”‚   â”œâ”€â”€ components/
â”‚   â”‚   â”‚   â”œâ”€â”€ Composer.tsx    # question box
â”‚   â”‚   â”‚   â”œâ”€â”€ ResultsView.tsx # plan, sources, probability panel, copy/print
â”‚   â”‚   â”‚   â””â”€â”€ Icons.tsx
â”‚   â”‚   â”œâ”€â”€ lib/judicia-data.ts # typed API client + response normalisation
â”‚   â”‚   â””â”€â”€ index.css           # design system
â”‚   â””â”€â”€ vite.config.ts          # /api proxy -> localhost:8000
â”‚
â”œâ”€â”€ models/judicia-domain-model/
â”‚   â”œâ”€â”€ config.json
â”‚   â”œâ”€â”€ tokenizer.json
â”‚   â”œâ”€â”€ tokenizer_config.json
â”‚   â””â”€â”€ model.safetensors       # ~418 MB, keep locally (git-ignored)
â”‚
â”œâ”€â”€ dataset/                    # training data
â”œâ”€â”€ src/                        # ML training pipeline
â””â”€â”€ notebooks/                  # EDA and evaluation notebooks
```

---

## 12. Security notes

- Keep `backend/.env` private. It is git-ignored.
- The frontend never receives the keys.
- The `/api/health` endpoint reports only whether services are configured, not secrets.
- Do not paste real keys into source files, screenshots, issues, or chat.
- For production, put the backend behind HTTPS, add rate limiting, and restrict CORS
  to your frontend origin.
