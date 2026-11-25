#!/usr/bin/env python3
"""
Seed the Pinecone index with Judicia's legal knowledge.

Two tracks are ingested into the same index, namespace and embedding space:

* the existing JSON corpus (``backend/legal_corpus.json``)
* PDFs in ``backend/docs/`` — Bare Acts split per section and handbooks
  split per procedural chunk (see ``backend/docs/README.md``)

Usage (from the project root):
    python backend/ingest_pinecone.py                # corpus + PDFs
    python backend/ingest_pinecone.py --pdf-only     # only the PDF tracks
    python backend/ingest_pinecone.py --corpus-only  # only the JSON corpus
    python backend/ingest_pinecone.py --dry-run      # extract + chunk, no network
    python backend/ingest_pinecone.py --recreate     # rebuild the index first

The script reads ``backend/.env`` (or the project-root ``.env``) and expects:
    PINECONE_API_KEY=...
    PINECONE_INDEX_NAME=judicia-legal        # optional
    PINECONE_NAMESPACE=legal-corpus          # optional
    PINECONE_EMBEDDING_MODEL=multilingual-e5-large
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Make sibling imports work when the script is launched from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from legal_knowledge import build_chunks, load_corpus  # noqa: E402
from pdf_ingest import (  # noqa: E402
    DEFAULT_CACHE_PATH,
    DEFAULT_DOCS_DIR,
    DEFAULT_MANIFEST_PATH,
    build_pdf_chunks,
    save_chunks_cache,
)
from rag_engine import (  # noqa: E402
    Pinecone,
    RAGConfig,
    _env,
    _field,
    _friendly_error,
    detect_embedding_dimension,
    ensure_pinecone_index,
    upsert_chunks,
)


def _env_path(name: str, default: Path) -> Path:
    value = _env(name)
    return Path(value) if value else default


def _print_stats(pc, config: RAGConfig) -> None:
    try:
        description = pc.describe_index(config.pinecone_index)
        host = _field(description, "host")
        if not host:
            return
        index = pc.Index(host=host)
        stats = index.describe_index_stats()
        total = _field(stats, "total_vector_count", 0)
        namespaces = _field(stats, "namespaces", {}) or {}
        print(f"\nIndex '{config.pinecone_index}' now has {total} vectors.")
        if namespaces:
            for name, details in namespaces.items():
                count = _field(details, "vector_count", 0)
                print(f"  namespace '{name}': {count} vectors")
    except Exception as exc:
        print(f"(Could not read index stats: {_friendly_error(exc)})")


def _summarise_chunks(chunks: list[dict], label: str) -> None:
    by_type = Counter(chunk.get("doc_type", "corpus") for chunk in chunks)
    by_document = Counter(
        chunk.get("short_code") or chunk.get("source_name") or "corpus" for chunk in chunks
    )
    print(f"\n{label}: {len(chunks)} chunks")
    for doc_type in ("bare_act", "handbook", "corpus"):
        if by_type.get(doc_type):
            print(f"  {doc_type}: {by_type[doc_type]}")
    for name, count in by_document.most_common():
        print(f"  - {name}: {count}")


def _load_pdf_chunks(docs_dir: Path, manifest: Path) -> list[dict]:
    print(f"Extracting PDFs from {docs_dir} ...")
    try:
        chunks, warnings = build_pdf_chunks(docs_dir, manifest, verbose=True)
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        print(
            "Install the PDF dependencies with:\n"
            "  python -m pip install -r backend/requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(1)
    for warning in warnings:
        print(f"  warning: {warning}")
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest the Judicia legal corpus and PDF knowledge base into Pinecone."
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--pdf-only",
        action="store_true",
        help="Only ingest the PDF tracks in backend/docs (existing corpus vectors are kept).",
    )
    source_group.add_argument(
        "--corpus-only",
        action="store_true",
        help="Only ingest the JSON corpus (PDFs are ignored).",
    )
    parser.add_argument(
        "--docs-dir",
        default=None,
        help=f"Folder containing the source PDFs (default: {DEFAULT_DOCS_DIR}).",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help=f"PDF manifest path (default: {DEFAULT_MANIFEST_PATH}).",
    )
    parser.add_argument(
        "--cache",
        default=None,
        help=f"Where to write the extracted chunk cache (default: {DEFAULT_CACHE_PATH}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Vectors per embedding/upsert batch (default: 32).",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the index before ingesting (use after changing embedding models).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow --recreate together with --pdf-only, which also deletes corpus vectors.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and chunk everything, print a report, and do not call Pinecone.",
    )
    args = parser.parse_args()

    if args.recreate and args.pdf_only and not args.force:
        print(
            "--recreate deletes the whole index, including the JSON corpus vectors.\n"
            "Use --corpus-only for a full rebuild, drop --recreate, or add --force.",
            file=sys.stderr,
        )
        return 1

    config = RAGConfig.from_env()
    docs_dir = Path(args.docs_dir) if args.docs_dir else _env_path("DOCS_DIR", DEFAULT_DOCS_DIR)
    manifest = (
        Path(args.manifest)
        if args.manifest
        else _env_path("PDF_MANIFEST_PATH", DEFAULT_MANIFEST_PATH)
    )
    cache_path = Path(args.cache) if args.cache else _env_path("PDF_CHUNKS_PATH", DEFAULT_CACHE_PATH)

    # -- build chunks ----------------------------------------------------
    corpus_chunks: list[dict] = []
    pdf_chunks: list[dict] = []

    if not args.pdf_only:
        corpus = load_corpus()
        corpus_chunks = build_chunks(corpus)
        if not corpus_chunks:
            print(
                "No corpus chunks found. Check backend/legal_corpus.json.",
                file=sys.stderr,
            )
            return 1
        _summarise_chunks(corpus_chunks, f"JSON corpus ({len(corpus)} provisions)")

    if not args.corpus_only:
        pdf_chunks = _load_pdf_chunks(docs_dir, manifest)
        if pdf_chunks:
            _summarise_chunks(pdf_chunks, "PDF tracks")
        elif args.pdf_only:
            print(
                "\nNo PDF chunks were produced. Add PDFs to backend/docs/ and check "
                "backend/docs_manifest.json.",
                file=sys.stderr,
            )
            return 1
        else:
            print("\nNo PDF chunks were produced; continuing with the JSON corpus only.")

    all_chunks = [*corpus_chunks, *pdf_chunks]
    _summarise_chunks(all_chunks, "TOTAL")

    if args.dry_run:
        print("\nSample chunks:")
        for chunk in all_chunks[:8]:
            print(
                f"  - {chunk['id']} | {chunk.get('doc_type', 'corpus')} | "
                f"{chunk.get('citation') or chunk.get('title', '')[:80]}"
            )
        print("\nDry run complete. Nothing was sent to Pinecone.")
        return 0

    if pdf_chunks:
        # Save before touching Pinecone: the cache powers the local fallback
        # even if the network call fails.
        saved = save_chunks_cache(pdf_chunks, cache_path)
        print(f"\nSaved {len(pdf_chunks)} PDF chunks to {saved}")

    # -- connect ---------------------------------------------------------
    if not config.pinecone_api_key:
        print(
            "PINECONE_API_KEY is missing.\n"
            "Copy backend/.env.example to backend/.env and paste your Pinecone key.",
            file=sys.stderr,
        )
        return 1

    if Pinecone is None:
        print(
            "The pinecone package is not installed. Run: pip install -r backend/requirements.txt",
            file=sys.stderr,
        )
        return 1

    print(
        f"\nConnecting to Pinecone (index={config.pinecone_index}, "
        f"namespace={config.pinecone_namespace}, model={config.embedding_model})..."
    )

    try:
        pc = Pinecone(api_key=config.pinecone_api_key)
        index = ensure_pinecone_index(pc, config, recreate=args.recreate)

        written_pdf = 0
        written_corpus = 0
        if pdf_chunks:
            print(f"\nUpserting PDF chunks (dimension={detect_embedding_dimension(pc, config)})...")
            written_pdf = upsert_chunks(
                pc,
                config,
                pdf_chunks,
                index=index,
                batch_size=args.batch_size,
                label="PDF vectors",
            )
        if corpus_chunks:
            print("\nUpserting JSON corpus chunks...")
            written_corpus = upsert_chunks(
                pc,
                config,
                corpus_chunks,
                index=index,
                batch_size=args.batch_size,
                label="corpus vectors",
            )

        print(
            f"\nDone. Wrote {written_pdf} PDF vectors and {written_corpus} corpus vectors."
        )
        _print_stats(pc, config)
        print("\nStart the backend and ask a question to test retrieval:")
        print("  python backend/app.py")
        return 0
    except Exception as exc:
        print(f"\nIngestion failed: {_friendly_error(exc)}", file=sys.stderr)
        print(
            "Check that the Pinecone key is valid, the index name is correct, and the "
            "embedding model is available to your Pinecone project.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
