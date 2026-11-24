"""
Judicia RAG engine.

Pipeline:
  1. Pinecone hosted inference embeds the user's question.
  2. Pinecone returns the closest statutory provisions (optionally filtered by
     the classifier's predicted legal domain).
  3. Groq writes a grounded, structured first-aid plan from those provisions.

Every external call has a graceful local fallback:
  * Pinecone unavailable  -> TF-IDF + keyword retrieval over ``legal_corpus.json``
  * Groq unavailable      -> rule-based domain playbook from ``legal_knowledge``

No API key is ever logged or returned to the browser.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from legal_knowledge import (
    DEFAULT_DISCLAIMER,
    build_chunks,
    compose_fallback_answer,
    get_playbook,
    load_corpus,
)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


def _load_env_files() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:  # pragma: no cover - python-dotenv is optional at runtime
        return

    for candidate in (PROJECT_ROOT / ".env", BASE_DIR / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)


_load_env_files()


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name, str(default))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class RAGConfig:
    pinecone_api_key: str
    pinecone_index: str
    pinecone_namespace: str
    pinecone_cloud: str
    pinecone_region: str
    embedding_model: str
    embedding_dimension: int
    groq_api_key: str
    groq_model: str
    groq_fallback_models: tuple[str, ...]
    top_k: int
    groq_timeout: float

    @classmethod
    def from_env(cls) -> "RAGConfig":
        fallback_models = tuple(
            model.strip()
            for model in _env(
                "GROQ_FALLBACK_MODELS",
                "openai/gpt-oss-20b,qwen/qwen3.8-27b",
            ).split(",")
            if model.strip()
        )
        return cls(
            pinecone_api_key=_env("PINECONE_API_KEY"),
            pinecone_index=_env("PINECONE_INDEX_NAME", "judicia-legal"),
            pinecone_namespace=_env("PINECONE_NAMESPACE", "legal-corpus"),
            pinecone_cloud=_env("PINECONE_CLOUD", "aws"),
            pinecone_region=_env("PINECONE_REGION", "us-east-1"),
            embedding_model=_env("PINECONE_EMBEDDING_MODEL", "multilingual-e5-large"),
            embedding_dimension=_env_int("PINECONE_EMBEDDING_DIMENSION", 1024),
            groq_api_key=_env("GROQ_API_KEY"),
            groq_model=_env("GROQ_MODEL", "openai/gpt-oss-120b"),
            groq_fallback_models=fallback_models,
            top_k=max(3, min(10, _env_int("RAG_TOP_K", 6))),
            groq_timeout=float(_env_int("GROQ_TIMEOUT_SECONDS", 45)),
        )


# ---------------------------------------------------------------------------
# Optional clients
# ---------------------------------------------------------------------------

try:
    from pinecone import Pinecone, ServerlessSpec  # type: ignore
except Exception:  # pragma: no cover - only hit when dependency is missing
    Pinecone = None  # type: ignore[assignment]
    ServerlessSpec = None  # type: ignore[assignment]

try:
    from groq import Groq  # type: ignore
except Exception:  # pragma: no cover
    Groq = None  # type: ignore[assignment]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from dict-like and object-like SDK responses."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    return fallback


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = re.split(r"\s*\|\s*|\n+", value)
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in (_text(v) for v in value) if item]


def _shorten(value: str, limit: int) -> str:
    value = _text(value)
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def _friendly_error(exc: Exception) -> str:
    """Return a browser-safe error summary that never contains credentials."""
    message = str(exc).replace("\n", " ").strip()
    lowered = message.lower()

    if "401" in lowered or "unauthorized" in lowered or "invalid api key" in lowered:
        return "Authentication failed. Check the API key in backend/.env."
    if "403" in lowered or "forbidden" in lowered:
        return "The API key does not have permission for this request."
    if "429" in lowered or "rate limit" in lowered or "quota" in lowered:
        return "The API rate limit or quota was reached. Try again shortly."
    if "not found" in lowered and "index" in lowered:
        return "The Pinecone index was not found. Run: python backend/ingest_pinecone.py"
    if "timed out" in lowered or "timeout" in lowered:
        return "The service timed out. The local fallback was used."
    if "connection" in lowered or "network" in lowered or "dns" in lowered:
        return "The service could not be reached. Check the internet connection."
    if len(message) > 240:
        return message[:239].rstrip() + "…"
    return message or "The service is unavailable."


_ALLOWED_LINK_HOSTS = (
    "cybercrime.gov.in",
    "consumerhelpline.gov.in",
    "edaakhil.nic.in",
    "nalsa.gov.in",
    "nhrc.nic.in",
    "rtionline.gov.in",
    "indiacode.nic.in",
    "shramsuvidha.gov.in",
    "epfindia.gov.in",
    "pgportal.gov.in",
    "ncw.nic.in",
    "childlineindia.org.in",
    "wcd.nic.in",
    "india.gov.in",
)


def _safe_link(value: Any) -> str | None:
    link = _text(value)
    if not link:
        return None
    if not re.match(r"^https?://", link, re.IGNORECASE):
        return None
    try:
        host = (urlparse(link).hostname or "").lower()
    except ValueError:
        return None
    for allowed in _ALLOWED_LINK_HOSTS:
        if host == allowed or host.endswith("." + allowed):
            return link
    return None


def _safe_phone(value: Any) -> str | None:
    phone = re.sub(r"[^\d+\- ]", "", _text(value)).strip()
    if not phone or len(phone) > 20:
        return None
    if not re.search(r"\d{3,}", phone):
        return None
    return phone


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class LegalRAG:
    def __init__(self, config: RAGConfig | None = None, corpus_path: str | Path | None = None):
        self.config = config or RAGConfig.from_env()
        self.corpus = load_corpus(corpus_path or BASE_DIR / "legal_corpus.json")
        self.chunks = build_chunks(self.corpus)

        self._tfidf: TfidfVectorizer | None = None
        self._tfidf_matrix: Any = None
        self._token_sets: list[set[str]] = []
        self._build_local_index()

        self._pc: Any = None
        self._index: Any = None
        self._groq: Any = None
        self._pinecone_ready = False
        self._groq_ready = False
        self._pinecone_error: str | None = None
        self._groq_error: str | None = None
        self._last_engine: str | None = None

    # -- local index -----------------------------------------------------
    def _build_local_index(self) -> None:
        texts = [chunk["text"] for chunk in self.chunks]
        if not texts:
            return
        self._tfidf = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=20000,
        )
        self._tfidf_matrix = self._tfidf.fit_transform(texts)
        self._token_sets = [self._tokens(chunk["text"]) for chunk in self.chunks]

    @staticmethod
    def _tokens(value: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-zA-Z0-9]{3,}", value.lower())
            if token not in {"the", "and", "for", "with", "this", "that", "from", "your", "you"}
        }

    # -- clients ---------------------------------------------------------
    def _pinecone_client(self) -> Any:
        if Pinecone is None:
            raise RuntimeError("The pinecone package is not installed. Run: pip install pinecone")
        if not self.config.pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY is not configured.")
        if self._pc is None:
            self._pc = Pinecone(api_key=self.config.pinecone_api_key)
        return self._pc

    def _pinecone_index(self) -> Any:
        if self._index is not None:
            return self._index
        pc = self._pinecone_client()
        description = pc.describe_index(self.config.pinecone_index)
        host = _field(description, "host")
        if not host:
            raise RuntimeError("Pinecone did not return an index host.")
        self._index = pc.Index(host=host)
        return self._index

    def _groq_client(self) -> Any:
        if Groq is None:
            raise RuntimeError("The groq package is not installed. Run: pip install groq")
        if not self.config.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")
        if self._groq is None:
            self._groq = Groq(
                api_key=self.config.groq_api_key,
                timeout=self.config.groq_timeout,
                max_retries=1,
            )
        return self._groq

    # -- embeddings ------------------------------------------------------
    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        if not texts:
            return []
        pc = self._pinecone_client()
        parameters = {"input_type": input_type, "truncate": "END"}
        try:
            response = pc.inference.embed(
                model=self.config.embedding_model,
                inputs=texts,
                parameters=parameters,
            )
        except Exception as exc:
            # Some embedding models do not accept input_type. Retry without it.
            if "input_type" in str(exc).lower() or "parameter" in str(exc).lower():
                response = pc.inference.embed(
                    model=self.config.embedding_model,
                    inputs=texts,
                    parameters={"truncate": "END"},
                )
            else:
                raise

        data = _field(response, "data", response)
        if data is None:
            data = []
        vectors: list[list[float]] = []
        for embedding in data:
            values = _field(embedding, "values")
            if values is not None:
                vectors.append([float(value) for value in values])
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Embedding provider returned {len(vectors)} vectors for {len(texts)} inputs."
            )
        return vectors

    # -- retrieval -------------------------------------------------------
    def _query_pinecone(
        self,
        vector: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        index = self._pinecone_index()
        result = index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            namespace=self.config.pinecone_namespace,
            filter=metadata_filter,
        )
        matches = _field(result, "matches", []) or []
        rows: list[dict[str, Any]] = []
        for match in matches:
            metadata = _field(match, "metadata", {}) or {}
            if not isinstance(metadata, dict):
                metadata = {}
            rows.append(
                {
                    "chunk_id": _text(_field(match, "id")),
                    "score": float(_field(match, "score", 0.0) or 0.0),
                    "metadata": metadata,
                }
            )
        return rows

    def _source_from_metadata(
        self, chunk_id: str, score: float, metadata: dict[str, Any], engine: str, domain: str
    ) -> dict[str, Any]:
        passages = _string_list(metadata.get("passages"))
        source_domain = _text(metadata.get("domain"), "General")
        citation = _text(metadata.get("citation"))
        title = _text(metadata.get("title"))
        if engine == "pinecone":
            why = (
                f"Semantic match in the {source_domain} domain "
                f"(vector similarity {score:.2f})."
            )
        else:
            why = (
                f"Local keyword index matched this provision to your question "
                f"({source_domain} domain)."
            )
        if source_domain.lower() == domain.lower():
            why += " The predicted category also aligns with this source."

        try:
            year = int(metadata.get("year", 0))
        except (TypeError, ValueError):
            year = 0

        return {
            "id": _text(metadata.get("source_id")) or chunk_id.split("::")[0],
            "chunk_id": chunk_id,
            "domain": source_domain,
            "category": _text(metadata.get("category"), "Statute"),
            "jurisdiction": _text(metadata.get("jurisdiction"), "India"),
            "year": year,
            "title": title,
            "citation": citation,
            "snippet": _text(metadata.get("snippet")),
            "passages": passages,
            "score": round(score, 4),
            "why": why,
        }

    def _finalise_sources(
        self, sources: Iterable[dict[str, Any]], engine: str
    ) -> list[dict[str, Any]]:
        deduped: dict[str, dict[str, Any]] = {}
        for source in sources:
            key = source.get("id") or source.get("chunk_id")
            if not key:
                continue
            previous = deduped.get(key)
            if previous is None or source.get("score", 0) > previous.get("score", 0):
                deduped[key] = source

        ordered = sorted(
            deduped.values(), key=lambda item: item.get("score", 0.0), reverse=True
        )
        for source in ordered:
            score = float(source.get("score", 0.0))
            if engine == "pinecone":
                match = 0.55 + 0.44 * _clamp(score, 0.0, 1.0)
            else:
                match = 0.5 + 0.48 * _clamp(score * 2.5, 0.0, 1.0)
            source["match"] = round(_clamp(match, 0.5, 0.99), 3)
        return ordered

    def _local_search(self, query: str, domain: str, top_k: int) -> list[dict[str, Any]]:
        if self._tfidf is None or self._tfidf_matrix is None or not self.chunks:
            return []

        query_vector = self._tfidf.transform([query])
        similarities = cosine_similarity(query_vector, self._tfidf_matrix)[0]
        query_tokens = self._tokens(query)

        scored: list[tuple[float, int]] = []
        for index, chunk in enumerate(self.chunks):
            raw = float(similarities[index])
            domain_match = chunk["domain"].lower() == domain.lower()
            overlap = 0.0
            if query_tokens:
                overlap = len(query_tokens & self._token_sets[index]) / len(query_tokens)
            score = raw * (1.3 if domain_match else 1.0) + overlap * 0.08
            if raw > 0.002 or domain_match:
                scored.append((score, index))

        scored.sort(reverse=True)
        rows = [
            self._source_from_metadata(
                chunk_id=self.chunks[index]["id"],
                score=score,
                metadata=self.chunks[index],
                engine="local",
                domain=domain,
            )
            for score, index in scored[: max(top_k * 2, 8)]
        ]

        if not rows:
            # Last resort: show the provisions from the predicted domain so the
            # answer still has something concrete to cite.
            fallback_rows = [
                self._source_from_metadata(
                    chunk_id=chunk["id"],
                    score=0.01,
                    metadata=chunk,
                    engine="local",
                    domain=domain,
                )
                for chunk in self.chunks
                if chunk["domain"].lower() == domain.lower()
            ]
            rows = fallback_rows[:top_k]

        return self._finalise_sources(rows, "local")[:top_k]

    def search(
        self, query: str, domain: str, confidence: float
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Retrieve the most relevant statutory sources for a query."""
        started = time.perf_counter()
        top_k = self.config.top_k

        if self.config.pinecone_api_key and Pinecone is not None:
            try:
                vector = self._embed([query], "query")[0]
                rows: list[dict[str, Any]] = []

                # The classifier is a useful signal, not a hard gate. Query the
                # predicted domain first, then merge in an unfiltered search so
                # a misclassification cannot hide the right statute.
                if confidence >= 0.40:
                    rows += self._query_pinecone(
                        vector, top_k=max(top_k * 2, 10), metadata_filter={"domain": {"$eq": domain}}
                    )
                rows += self._query_pinecone(
                    vector, top_k=max(top_k * 2, 10), metadata_filter=None
                )

                sources = [
                    self._source_from_metadata(
                        chunk_id=row["chunk_id"],
                        score=row["score"],
                        metadata=row["metadata"],
                        engine="pinecone",
                        domain=domain,
                    )
                    for row in rows
                ]
                for source in sources:
                    if source["domain"].lower() == domain.lower():
                        source["score"] = round(source["score"] * 1.03 + 0.01, 4)

                sources = self._finalise_sources(sources, "pinecone")[:top_k]
                if sources:
                    self._pinecone_ready = True
                    self._pinecone_error = None
                    self._last_engine = "pinecone"
                    return sources, {
                        "engine": "pinecone",
                        "index": self.config.pinecone_index,
                        "namespace": self.config.pinecone_namespace,
                        "embedding_model": self.config.embedding_model,
                        "candidates": len(rows),
                        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                    }
                raise RuntimeError("Pinecone returned no matches.")
            except Exception as exc:
                self._pinecone_ready = False
                self._pinecone_error = _friendly_error(exc)

        sources = self._local_search(query, domain, top_k)
        self._last_engine = "local"
        return sources, {
            "engine": "local",
            "reason": self._pinecone_error
            or (
                "PINECONE_API_KEY is not configured; using the local keyword index."
                if not self.config.pinecone_api_key
                else "Pinecone was unavailable; using the local keyword index."
            ),
            "candidates": len(sources),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    # -- answer generation ----------------------------------------------
    def _context_for_llm(self, sources: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for index, source in enumerate(sources, start=1):
            passages = "; ".join(source.get("passages") or []) or "Not available"
            blocks.append(
                f"[S{index}] {source.get('citation') or source.get('title')}"
                f" ({source.get('category')}, {source.get('year')})\n"
                f"Title: {source.get('title')}\n"
                f"Summary: {_shorten(str(source.get('snippet', '')), 650)}\n"
                f"Key provisions: {_shorten(passages, 650)}"
            )
        return "\n\n".join(blocks) or "No statutes were retrieved."

    def _system_prompt(self) -> str:
        return (
            "You are Judicia, an Indian legal first-aid assistant. You help ordinary people "
            "understand what to do next after a legal problem. You are calm, practical, and "
            "empathetic. You are not a lawyer and you must not give a final legal opinion.\n\n"
            "GROUNDING RULES:\n"
            "- Use ONLY the retrieved context for legal provisions. Cite section numbers and "
            "statute names exactly as they appear in the context.\n"
            "- Never invent a section number, case name, deadline, fee, phone number, or authority.\n"
            "- If the context does not cover an important point, say so plainly and recommend a "
            "qualified advocate.\n"
            "- Do not promise a result or predict how a court will decide.\n"
            "- If there is any risk to life or safety, tell the person to call 112 first.\n"
            "- Use short, plain sentences. Avoid legal jargon; explain any term you must use.\n\n"
            "OUTPUT: return a single JSON object only, with exactly these keys:\n"
            '{"summary": string, "urgency": "emergency" | "urgent" | "normal", '
            '"immediate_steps": string[], "rights": string[], "options": string[], '
            '"documents": string[], "where_to_complain": '
            '[{"name": string, "detail": string, "phone": string | null}], '
            '"key_deadlines": string[], "disclaimer": string}\n\n'
            "Give 4-7 immediate_steps, 3-6 rights, 2-5 options, 3-6 documents, "
            "1-4 where_to_complain entries, and 0-4 key_deadlines. Do not add markdown. "
            "Do not add any text outside the JSON object."
        )

    def _user_prompt(
        self,
        query: str,
        domain: str,
        confidence: float,
        sources: list[dict[str, Any]],
    ) -> str:
        playbook = get_playbook(domain)
        verified = "\n".join(
            f"- {entry.get('name')}: {entry.get('detail')}"
            + (f" (phone {entry.get('phone')})" if entry.get("phone") else "")
            for entry in playbook.get("where_to_complain", [])
        )
        return (
            f"SITUATION:\n{query}\n\n"
            f"CLASSIFIER:\nPredicted domain: {domain}\n"
            f"Model confidence: {confidence * 100:.1f}%\n\n"
            f"RETRIEVED CONTEXT:\n{self._context_for_llm(sources)}\n\n"
            f"VERIFIED HELPLINES/AUTHORITIES (you may reuse the relevant ones):\n{verified}\n\n"
            "Write the first-aid plan for the person in the SITUATION. Be specific to their "
            "facts, but ground every legal point in the CONTEXT. Put the most urgent action "
            "first. Explain what evidence to preserve. Tell them where to go and what to say."
        )

    def _call_groq(self, messages: list[dict[str, str]]) -> tuple[str, str]:
        client = self._groq_client()
        models: list[str] = []
        for model in (self.config.groq_model, *self.config.groq_fallback_models):
            if model and model not in models:
                models.append(model)

        last_error: Exception | None = None
        for model in models:
            try:
                try:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.2,
                        max_tokens=1800,
                        response_format={"type": "json_object"},
                    )
                except Exception as exc:
                    # A few Groq models do not support JSON mode. Retry once
                    # without it instead of failing the whole request.
                    if "response_format" not in str(exc).lower() and "json" not in str(exc).lower():
                        raise
                    completion = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.2,
                        max_tokens=1800,
                    )

                choices = _field(completion, "choices", []) or []
                message = _field(choices[0], "message", {}) if choices else {}
                content = _text(_field(message, "content"))
                if not content:
                    raise RuntimeError("Groq returned an empty response.")
                return content, model
            except Exception as exc:
                last_error = exc
                if not self._model_unavailable(exc):
                    break
        raise last_error or RuntimeError("Groq request failed.")

    @staticmethod
    def _model_unavailable(exc: Exception) -> bool:
        message = str(exc).lower()
        if "model" not in message:
            return False
        return any(
            phrase in message
            for phrase in (
                "not found",
                "does not exist",
                "decommission",
                "not supported",
                "invalid model",
                "no longer supported",
            )
        )

    def _parse_json(self, content: str) -> dict[str, Any]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start == -1 or end <= start:
                raise RuntimeError("The language model did not return valid JSON.")
            parsed = json.loads(cleaned[start : end + 1])
        if not isinstance(parsed, dict):
            raise RuntimeError("The language model returned an unexpected JSON shape.")
        return parsed

    def _sanitize_answer(
        self, raw: dict[str, Any], fallback: dict[str, Any]
    ) -> dict[str, Any]:
        summary = _text(raw.get("summary"))
        if len(summary) < 40:
            summary = fallback["summary"]

        urgency = _text(raw.get("urgency")).lower()
        if urgency not in {"emergency", "urgent", "normal"}:
            urgency = fallback["urgency"]

        steps = _string_list(raw.get("immediate_steps"))
        if len(steps) < 2:
            steps = list(fallback["immediate_steps"])

        rights = self._unique_strings(
            _string_list(raw.get("rights")) + list(fallback.get("rights", [])), 8
        )
        options = self._unique_strings(
            _string_list(raw.get("options")) or list(fallback.get("options", [])), 6
        )
        documents = self._unique_strings(
            _string_list(raw.get("documents")) + list(fallback.get("documents", [])), 8
        )
        deadlines = self._unique_strings(
            _string_list(raw.get("key_deadlines")) or list(fallback.get("key_deadlines", [])), 4
        )
        places = self._merge_places(
            raw.get("where_to_complain"), fallback.get("where_to_complain", [])
        )

        return {
            "summary": _shorten(summary, 900),
            "urgency": urgency,
            "immediate_steps": [ _shorten(item, 700) for item in steps[:8] ],
            "rights": [ _shorten(item, 600) for item in rights ],
            "options": [ _shorten(item, 600) for item in options ],
            "documents": [ _shorten(item, 400) for item in documents ],
            "where_to_complain": places,
            "key_deadlines": [ _shorten(item, 500) for item in deadlines ],
            "disclaimer": DEFAULT_DISCLAIMER,
        }

    @staticmethod
    def _unique_strings(items: Iterable[str], limit: int) -> list[str]:
        seen: set[str] = set()
        output: list[str] = []
        for item in items:
            cleaned = _text(item)
            key = cleaned.lower()
            if cleaned and key not in seen:
                seen.add(key)
                output.append(cleaned)
            if len(output) >= limit:
                break
        return output

    def _merge_places(
        self, generated: Any, fallback_places: Iterable[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []

        if isinstance(generated, (list, tuple)):
            for item in generated:
                entry = self._place_entry(item)
                if entry:
                    entries.append(entry)

        index_by_name: dict[str, int] = {}
        for position, entry in enumerate(entries):
            index_by_name[entry["name"].lower()] = position

        for item in fallback_places:
            entry = self._place_entry(item)
            if not entry:
                continue
            key = entry["name"].lower()
            existing_position = index_by_name.get(key)
            if existing_position is None:
                entries.append(entry)
                index_by_name[key] = len(entries) - 1
            else:
                existing = entries[existing_position]
                if not existing.get("phone") and entry.get("phone"):
                    existing["phone"] = entry["phone"]
                if not existing.get("link") and entry.get("link"):
                    existing["link"] = entry["link"]

        return entries[:6]

    def _place_entry(self, item: Any) -> dict[str, Any] | None:
        if isinstance(item, str):
            name = _text(item)
            detail = ""
            phone = None
            link = None
        elif isinstance(item, dict):
            name = _text(item.get("name") or item.get("authority") or item.get("title"))
            detail = _text(item.get("detail") or item.get("description"))
            phone = _safe_phone(item.get("phone"))
            link = _safe_link(item.get("link"))
        else:
            return None

        if not name:
            return None
        return {
            "name": _shorten(name, 160),
            "detail": _shorten(detail, 320),
            "phone": phone,
            "link": link,
        }

    def answer(
        self,
        query: str,
        domain: str,
        confidence: float,
        sources: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Generate a structured first-aid answer, always returning a plan."""
        started = time.perf_counter()
        fallback = compose_fallback_answer(query, domain, sources)

        if not self.config.groq_api_key or Groq is None:
            self._groq_ready = False
            self._groq_error = "GROQ_API_KEY is not configured."
            return fallback, {
                "used": False,
                "model": "rule-based playbook",
                "reason": "GROQ_API_KEY is not configured; showing the rule-based first-aid plan.",
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }

        messages = [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": self._user_prompt(query, domain, confidence, sources),
            },
        ]

        try:
            content, model = self._call_groq(messages)
            parsed = self._parse_json(content)
            answer = self._sanitize_answer(parsed, fallback)
            self._groq_ready = True
            self._groq_error = None
            return answer, {
                "used": True,
                "model": model,
                "reason": None,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        except Exception as exc:
            self._groq_ready = False
            self._groq_error = _friendly_error(exc)
            return fallback, {
                "used": False,
                "model": "rule-based playbook",
                "reason": f"{self._groq_error} Showing the rule-based first-aid plan.",
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }

    # -- status ----------------------------------------------------------
    def status(self) -> dict[str, Any]:
        return {
            "pinecone": {
                "configured": bool(self.config.pinecone_api_key),
                "index": self.config.pinecone_index,
                "namespace": self.config.pinecone_namespace,
                "embedding_model": self.config.embedding_model,
                "ready": self._pinecone_ready,
                "last_error": self._pinecone_error,
            },
            "groq": {
                "configured": bool(self.config.groq_api_key),
                "model": self.config.groq_model,
                "ready": self._groq_ready,
                "last_error": self._groq_error,
            },
            "retrieval": {
                "engine": self._last_engine
                or ("pinecone" if self.config.pinecone_api_key else "local")
            },
            "corpus_items": len(self.corpus),
        }


# ---------------------------------------------------------------------------
# Ingest helpers (used by ingest_pinecone.py)
# ---------------------------------------------------------------------------


def detect_embedding_dimension(pc: Any, config: RAGConfig) -> int:
    """Ask Pinecone for the model's default dimension, with an env fallback."""
    try:
        info = pc.inference.get_model(model=config.embedding_model)
        dimension = _field(info, "default_dimension")
        if dimension:
            return int(dimension)
    except Exception:
        pass
    return config.embedding_dimension


def ensure_pinecone_index(
    pc: Any,
    config: RAGConfig,
    *,
    recreate: bool = False,
    wait_seconds: int = 90,
) -> Any:
    """Create the Pinecone index when needed and return a data-plane client."""
    if ServerlessSpec is None:
        raise RuntimeError("The pinecone package is not installed correctly.")

    existing_names = {_field(index, "name") for index in pc.list_indexes()}
    if recreate and config.pinecone_index in existing_names:
        print(f"Deleting existing index '{config.pinecone_index}'...")
        pc.delete_index(config.pinecone_index)
        deadline = time.time() + wait_seconds
        while time.time() < deadline:
            if config.pinecone_index not in {_field(index, "name") for index in pc.list_indexes()}:
                break
            time.sleep(2)

    if config.pinecone_index not in existing_names or recreate:
        dimension = detect_embedding_dimension(pc, config)
        print(
            f"Creating Pinecone index '{config.pinecone_index}' "
            f"({dimension} dimensions, cosine, {config.pinecone_cloud}/{config.pinecone_region})..."
        )
        pc.create_index(
            name=config.pinecone_index,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=config.pinecone_cloud,
                region=config.pinecone_region,
            ),
        )

    deadline = time.time() + wait_seconds
    description: Any = None
    while time.time() < deadline:
        description = pc.describe_index(config.pinecone_index)
        status = _field(description, "status")
        ready = _field(status, "ready")
        if ready is None:
            ready = True
        if ready:
            break
        time.sleep(2)

    host = _field(description, "host")
    if not host:
        raise RuntimeError("Pinecone index exists but no host was returned.")
    return pc.Index(host=host)


def ingest_corpus(
    pc: Any,
    config: RAGConfig,
    *,
    recreate: bool = False,
    batch_size: int = 32,
) -> int:
    """Embed and upsert every corpus chunk. Returns the number of vectors written."""
    corpus = load_corpus()
    chunks = build_chunks(corpus)
    if not chunks:
        raise RuntimeError("The legal corpus is empty or could not be loaded.")

    index = ensure_pinecone_index(pc, config, recreate=recreate)

    vectors_written = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [chunk["text"] for chunk in batch]
        try:
            response = pc.inference.embed(
                model=config.embedding_model,
                inputs=texts,
                parameters={"input_type": "passage", "truncate": "END"},
            )
        except Exception as exc:
            if "input_type" in str(exc).lower() or "parameter" in str(exc).lower():
                response = pc.inference.embed(
                    model=config.embedding_model,
                    inputs=texts,
                    parameters={"truncate": "END"},
                )
            else:
                raise
        data = _field(response, "data", response) or []
        if len(data) != len(batch):
            raise RuntimeError(
                f"Embedding provider returned {len(data)} vectors for {len(batch)} chunks."
            )
        vectors: list[dict[str, Any]] = []
        for chunk, embedding in zip(batch, data):
            values = _field(embedding, "values")
            if values is None:
                continue
            metadata = {
                key: chunk[key]
                for key in (
                    "source_id",
                    "domain",
                    "category",
                    "jurisdiction",
                    "year",
                    "title",
                    "citation",
                    "snippet",
                    "passages",
                    "keywords",
                )
            }
            metadata["text"] = chunk["text"]
            vectors.append(
                {
                    "id": chunk["id"],
                    "values": [float(value) for value in values],
                    "metadata": metadata,
                }
            )

        if vectors:
            index.upsert(vectors=vectors, namespace=config.pinecone_namespace)
            vectors_written += len(vectors)
            print(f"  upserted {vectors_written}/{len(chunks)} vectors")

    return vectors_written
