# backend/docs — PDF source folder

Drop the source PDFs here, then run the PDF ingestion:

```powershell
python backend/ingest_pinecone.py
```

## Two tracks

| Track | What goes here | How it is chunked |
| --- | --- | --- |
| `bare_act` | Official Acts, Sanhitas, Codes (BNS, BNSS, CPA, IT Act, PWDVA, RTI) | One chunk per **section**. `Provided that` clauses and `Explanation` blocks stay attached to their section. Sections longer than ~1,800 characters are split into parts. |
| `handbook` | Legal-aid handbooks, SOPs, FAQs, step-by-step guides | `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=200`), preferring paragraph, `Q.`, and `Step ` boundaries. |

## Two ways to categorise a PDF

1. **Manifest (recommended).** Add an entry to `backend/docs_manifest.json`
   with `doc_type`, `act_name`, `short_code` / `source_name`, `domain`, and `year`.
2. **Subfolder.** Put the file in `backend/docs/bare_act/` or
   `backend/docs/handbook/`. Unlisted PDFs in those folders are picked up
   automatically. Unlisted PDFs in `backend/docs/` are guessed from the filename.

## File-name matching

The manifest matches after normalising case, spaces, commas, underscores and
hyphens, so `Bharatiya_Nagarik_Suraksha_Sanhita,_2023.pdf` matches the entry
written as `Bharatiya_Nagarik_Suraksha_Sanhita_2023.pdf`. Each entry can also
define `patterns` (regular expressions) as a fallback.

## Commands

```powershell
# Extract + chunk everything and print a report (no network)
python backend/ingest_pinecone.py --dry-run

# Embed and upsert both the JSON corpus and every PDF
python backend/ingest_pinecone.py

# Only the PDFs (leaves existing corpus vectors untouched)
python backend/ingest_pinecone.py --pdf-only

# Only the JSON corpus (PDFs are ignored)
python backend/ingest_pinecone.py --corpus-only

# Custom folders / batch size
python backend/ingest_pinecone.py --docs-dir backend/docs --manifest backend/docs_manifest.json --batch-size 32

# Rebuild the index, then ingest both tracks
python backend/ingest_pinecone.py --recreate
```

Extracted chunks are cached in `backend/docs/pdf_chunks.json` so the local
fallback (used when Pinecone is unavailable) can search them too. That file is
generated — do not edit it by hand. Override the locations with the optional
`DOCS_DIR`, `PDF_MANIFEST_PATH` and `PDF_CHUNKS_PATH` environment variables in
`backend/.env`.

## Currently expected files

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

Missing files are reported as warnings; the rest still ingest normally.
