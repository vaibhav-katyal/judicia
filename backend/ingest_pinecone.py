#!/usr/bin/env python3
"""
Seed the Pinecone index with Judicia's statutory corpus.

Usage (from the project root):
    python backend/ingest_pinecone.py
    python backend/ingest_pinecone.py --recreate
    python backend/ingest_pinecone.py --dry-run

The script reads ``backend/.env`` (or the project-root ``.env``) and expects:
    PINECONE_API_KEY=...
    PINECONE_INDEX_NAME=judicia-legal   # optional
    PINECONE_NAMESPACE=legal-corpus     # optional
    PINECONE_EMBEDDING_MODEL=multilingual-e5-large
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sibling imports work when the script is launched from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from legal_knowledge import build_chunks, load_corpus  # noqa: E402
from rag_engine import (  # noqa: E402
    Pinecone,
    RAGConfig,
    _field,
    _friendly_error,
    ingest_corpus,
)


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest the Judicia legal corpus into Pinecone.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the index before ingesting (use after changing embedding models).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the chunks that would be embedded, without calling Pinecone.",
    )
    args = parser.parse_args()

    corpus = load_corpus()
    chunks = build_chunks(corpus)

    if not chunks:
        print("No corpus chunks found. Check backend/legal_corpus.json.", file=sys.stderr)
        return 1

    print(f"Loaded {len(corpus)} provisions -> {len(chunks)} Pinecone chunks.")

    if args.dry_run:
        for chunk in chunks:
            print(f"- {chunk['id']} | {chunk['domain']} | {chunk['title'][:80]}")
        print("\nDry run complete. Nothing was sent to Pinecone.")
        return 0

    config = RAGConfig.from_env()
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
        f"Connecting to Pinecone (index={config.pinecone_index}, "
        f"namespace={config.pinecone_namespace}, model={config.embedding_model})..."
    )

    try:
        pc = Pinecone(api_key=config.pinecone_api_key)
        written = ingest_corpus(pc, config, recreate=args.recreate)
        print(f"\nDone. Wrote {written} vectors to Pinecone.")
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
