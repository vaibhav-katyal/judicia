"""PDF ingestion for Judicia.

Two independent tracks are supported:

``bare_act``
    Official statutes, Sanhitas and Codes. Text is split on section boundaries
    (``Section 12.`` / ``12. Title.``) so a rule always stays with its
    ``Provided that`` / ``Explanation`` clauses. A section is only sub-chunked
    when it exceeds ``max_chars`` (~1800 by default).

``handbook``
    Legal-aid handbooks, SOPs and FAQs. Text is split with LangChain's
    ``RecursiveCharacterTextSplitter`` (1000 / 200) preferring paragraph, ``Q.``
    and ``Step`` boundaries.

Both tracks return chunks in the same shape as ``legal_knowledge.build_chunks``
so ``rag_engine.upsert_chunks`` can embed them with the project's existing
Pinecone embedding model (``multilingual-e5-large``, 1024 dimensions) and
upsert them into the same index and namespace.

PyMuPDF and langchain-text-splitters are imported lazily so the rest of the
backend still starts when only the chatbot dependencies are installed.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DOCS_DIR = BASE_DIR / "docs"
DEFAULT_MANIFEST_PATH = BASE_DIR / "docs_manifest.json"
DEFAULT_CACHE_PATH = DEFAULT_DOCS_DIR / "pdf_chunks.json"

DEFAULT_BARE_ACT_MAX_CHARS = 1800
DEFAULT_HANDBOOK_CHUNK_SIZE = 1000
DEFAULT_HANDBOOK_CHUNK_OVERLAP = 200

CACHE_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Documents and manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PDFSpec:
    path: Path
    doc_type: str  # "bare_act" | "handbook"
    act_name: str
    short_code: str
    source_name: str
    domain: str
    category: str
    year: int
    jurisdiction: str = "India"


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return str(value)
    return str(value).strip()


def _slug(value: str, limit: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return (slug or "document")[:limit].strip("-") or "document"


def _normalise_name(value: str) -> str:
    stem = Path(value).stem if value.lower().endswith(".pdf") else value
    return re.sub(r"[^a-z0-9]+", " ", stem.lower()).strip()


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read the PDF manifest {path}: {exc}") from exc
    documents = payload.get("documents") if isinstance(payload, dict) else payload
    if not isinstance(documents, list):
        return []
    return [item for item in documents if isinstance(item, dict)]


def _manifest_spec(entry: dict[str, Any], pdf: Path) -> PDFSpec:
    doc_type = _text(entry.get("doc_type"), "").lower()
    if doc_type not in {"bare_act", "handbook"}:
        doc_type = _infer_doc_type(pdf)
    act_name = _text(entry.get("act_name")) or pdf.stem.replace("_", " ")
    short_code = _text(entry.get("short_code")) or act_name.upper()[:20]
    source_name = _text(entry.get("source_name")) or act_name
    domain = _text(entry.get("domain"), "General") or "General"
    try:
        year = int(entry.get("year", 0))
    except (TypeError, ValueError):
        year = 0
    category = "Bare Act" if doc_type == "bare_act" else "Handbook"
    return PDFSpec(
        path=pdf,
        doc_type=doc_type,
        act_name=act_name,
        short_code=short_code,
        source_name=source_name,
        domain=domain,
        category=category,
        year=year,
        jurisdiction=_text(entry.get("jurisdiction"), "India") or "India",
    )


def _infer_doc_type(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "handbook" in parts or "handbooks" in parts:
        return "handbook"
    if "bare_act" in parts or "bare-acts" in parts or "acts" in parts:
        return "bare_act"
    name = path.stem.lower()
    if any(token in name for token in ("handbook", "guide", "sop", "manual", "faq", "legal-aid")):
        return "handbook"
    return "bare_act"


def _infer_spec(pdf: Path) -> PDFSpec:
    doc_type = _infer_doc_type(pdf)
    name = re.sub(r"[_\-,]+", " ", pdf.stem)
    name = re.sub(r"\s+", " ", name).strip()
    year_match = re.search(r"(19|20)\d{2}", name)
    year = int(year_match.group(0)) if year_match else 0
    short_code = re.sub(r"\b(19|20)\d{2}\b", "", name).strip()[:18].upper() or "DOC"
    return PDFSpec(
        path=pdf,
        doc_type=doc_type,
        act_name=name,
        short_code=short_code,
        source_name=name,
        domain="General",
        category="Bare Act" if doc_type == "bare_act" else "Handbook",
        year=year,
    )


def _matches_entry(entry: dict[str, Any], pdf: Path) -> bool:
    candidates = [entry.get("filename"), *(entry.get("aliases") or [])]
    stem = _normalise_name(pdf.name)
    for candidate in candidates:
        if candidate and _normalise_name(_text(candidate)) == stem:
            return True
    for pattern in entry.get("patterns") or []:
        try:
            if re.search(str(pattern), pdf.name, re.IGNORECASE):
                return True
        except re.error:
            continue
    return False


def resolve_pdf_specs(
    docs_dir: str | Path = DEFAULT_DOCS_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> tuple[list[PDFSpec], list[str]]:
    """Match PDFs on disk to manifest entries.

    Returns ``(specs, warnings)``. Unmatched PDFs are inferred from their
    subfolder (``bare_act`` / ``handbook``) or filename.
    """
    directory = Path(docs_dir)
    warnings: list[str] = []
    if not directory.exists():
        return [], [f"The docs folder does not exist: {directory}"]

    pdfs = sorted(path for path in directory.rglob("*.pdf") if path.is_file())
    if not pdfs:
        return [], [f"No PDF files found in {directory}. Drop your PDFs there first."]

    entries = _load_manifest(Path(manifest_path))
    for entry in entries:
        filename = _text(entry.get("filename"))
        if filename and not any(_matches_entry(entry, pdf) for pdf in pdfs):
            warnings.append(f"Manifest entry not found on disk: {filename}")

    specs: list[PDFSpec] = []
    claimed: set[Path] = set()
    for entry in entries:
        match = next((pdf for pdf in pdfs if pdf not in claimed and _matches_entry(entry, pdf)), None)
        if match is None:
            continue
        claimed.add(match)
        specs.append(_manifest_spec(entry, match))

    for pdf in pdfs:
        if pdf not in claimed:
            specs.append(_infer_spec(pdf))

    return specs, warnings


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------


def _load_fitz():
    try:
        import pymupdf as fitz  # type: ignore

        return fitz
    except Exception:
        pass
    try:
        import fitz  # type: ignore

        return fitz
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "PyMuPDF is not installed. Run: python -m pip install pymupdf"
        ) from exc


@dataclass
class PdfLine:
    idx: int
    page: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    bold: bool


_NOISE_PATTERNS = [
    re.compile(r"^\s*(?:page\s*)?\d{1,4}\s*(?:of\s*\d{1,4})?\s*$", re.IGNORECASE),
    re.compile(r"^\s*[ivxlcdm]{1,6}\s*$", re.IGNORECASE),
    re.compile(r"^\s*the\s+gazette\s+of\s+india", re.IGNORECASE),
    re.compile(r"^\s*\[?\s*part\s*[-–—]?\s*ii\b", re.IGNORECASE),
    re.compile(r"^\s*\[?\s*(?:part|sec\.?)\b.*\]\s*$", re.IGNORECASE),
    re.compile(r"^\s*sec\.?\s*\d+\s*\]\s*$", re.IGNORECASE),
    re.compile(r"^\s*no\.?\s*\d+\s*\]\s*$", re.IGNORECASE),
    re.compile(r"^\s*published\s+by\s+authority", re.IGNORECASE),
    re.compile(r"^\s*registered\s+no\.?", re.IGNORECASE),
    re.compile(r"^\s*ministry\s+of\s+law\s+and\s+justice", re.IGNORECASE),
    re.compile(r"^\s*\(?\s*legislative\s+department\s*\)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*new\s+delhi,\s+the\b", re.IGNORECASE),
    re.compile(r"^\s*the\s+following\s+act\s+of\s+parliament\s+received", re.IGNORECASE),
    re.compile(r"^\s*separate\s+paging\s+is\s+given", re.IGNORECASE),
    re.compile(r"^\s*collected\s+by\s+the\s+all\s+india\s+christian\s+council", re.IGNORECASE),
    re.compile(r"^\s*homepage\s*$", re.IGNORECASE),
    re.compile(r"^\s*chapters?\s+and\s+sections\s*$", re.IGNORECASE),
    re.compile(r"^\s*index\s*$", re.IGNORECASE),
    re.compile(r"^\s*jftlv", re.IGNORECASE),
    re.compile(r"^\s*vlk/", re.IGNORECASE),
    re.compile(r"^\s*hkkx\b", re.IGNORECASE),
    re.compile(r"^\s*izkf/", re.IGNORECASE),
    re.compile(r"^\s*ubz\s+fnyyh", re.IGNORECASE),
    re.compile(r"^\s*bl\s+hkkx", re.IGNORECASE),
    re.compile(r"^\s*lai?[ñn]\s*\d+", re.IGNORECASE),
    re.compile(r"^\s*[—–\-_=]{2,}\s*$"),
]

_CONTENT_START = re.compile(
    r"^(?:\d{1,3}[a-z]{0,2}\.|\(|provided\b|explanation\b|chapter\b|section\b|clauses?\b)",
    re.IGNORECASE,
)


def _is_noise_line(value: str) -> bool:
    text = value.strip()
    if not text or len(text) < 2:
        return True
    return any(pattern.match(text) for pattern in _NOISE_PATTERNS)


def _looks_like_content(value: str) -> bool:
    return bool(_CONTENT_START.match(value)) or len(value) > 120


def _extract_layout_lines(path: Path) -> tuple[list[PdfLine], int]:
    fitz = _load_fitz()
    document = fitz.open(path)
    lines: list[PdfLine] = []
    try:
        for page_number, page in enumerate(document):
            payload = page.get_text("dict")
            for block in payload.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans") or []
                    text = "".join(_text(span.get("text")) for span in spans).strip()
                    if not text:
                        continue
                    bold = any(
                        "bold" in _text(span.get("font")).lower()
                        or (int(span.get("flags", 0) or 0) & 16)
                        for span in spans
                    )
                    x0, y0, x1, y1 = line.get("bbox", (0, 0, 0, 0))
                    lines.append(
                        PdfLine(
                            idx=len(lines),
                            page=page_number,
                            text=text,
                            x0=float(x0),
                            y0=float(y0),
                            x1=float(x1),
                            y1=float(y1),
                            bold=bold,
                        )
                    )
        page_count = document.page_count
    finally:
        document.close()
    return lines, page_count


def _extract_page_texts(path: Path) -> list[str]:
    fitz = _load_fitz()
    document = fitz.open(path)
    try:
        return [page.get_text("text") for page in document]
    finally:
        document.close()


def _clean_layout_lines(lines: list[PdfLine], page_count: int) -> list[PdfLine]:
    by_page: dict[int, list[PdfLine]] = defaultdict(list)
    for line in lines:
        if _is_noise_line(line.text):
            continue
        by_page[line.page].append(line)

    counts: Counter[str] = Counter()
    for page_lines in by_page.values():
        seen = {
            _normalise_name(line.text)
            for line in page_lines
            if 4 <= len(line.text) <= 110
        }
        counts.update(seen)

    threshold = max(5, int(page_count * 0.45))
    running = {
        key
        for key, count in counts.items()
        if count >= threshold and key and not _looks_like_content(key)
    }

    kept: list[PdfLine] = []
    for page_number in sorted(by_page):
        for line in by_page[page_number]:
            if _normalise_name(line.text) in running:
                continue
            kept.append(line)
    kept.sort(key=lambda line: (line.page, round(line.y0, 1), line.x0))
    for position, line in enumerate(kept):
        line.idx = position
    return kept


def _clean_page_texts(page_texts: Iterable[str]) -> str:
    by_page: list[list[str]] = []
    for raw in page_texts:
        cleaned: list[str] = []
        for raw_line in raw.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                cleaned.append("")
                continue
            if _is_noise_line(line):
                continue
            cleaned.append(line)
        by_page.append(cleaned)

    counts: Counter[str] = Counter()
    for cleaned in by_page:
        seen = {
            _normalise_name(line)
            for line in cleaned
            if 4 <= len(line) <= 110
        }
        counts.update(seen)
    threshold = max(4, int(len(by_page) * 0.45))
    running = {
        key
        for key, count in counts.items()
        if count >= threshold and key and not _looks_like_content(key)
    }

    pages: list[str] = []
    for cleaned in by_page:
        kept = [
            line
            for line in cleaned
            if not line or _normalise_name(line) not in running
        ]
        text = "\n".join(kept)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if text:
            pages.append(text)
    merged = "\n\n".join(pages)
    return re.sub(r"[ \t]{2,}", " ", merged).strip()


# ---------------------------------------------------------------------------
# Bare-act section splitting
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(?:Section\s+)?(\d{1,3})([A-Z]{0,2})\.\s*(.*)$")
_INLINE_TITLE_RE = re.compile(
    r"^(?P<title>[A-Z][^—–]{2,130}?)\s*(?:[—–]{1,2}|\.\s*[-–—])\s*(?P<rest>.*)$"
)
_AMENDMENT_HEADING_RE = re.compile(
    r"^\s*(?:Section\s+)?\d{1,3}[A-Z]{0,2}\.\s*\[?\s*"
    r"(?:Subs\.|Ins\.|Omitted|Added|Repealed|Amendment)\b",
    re.IGNORECASE,
)
_AMENDMENT_FOOTNOTE_RE = re.compile(
    r"^\s*\d{1,3}\.\s*(?:Subs\.|Ins\.|Omitted|Added|Repealed|Amendment)\b"
    r"|\bw\.e\.f\.\b",
    re.IGNORECASE,
)
_APPENDIX_RE = re.compile(
    r"^(?:the\s+)?(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)?\s*"
    r"(?:schedule|statement\s+of\s+objects|notes\s+on\s+clauses|memorandum)\b",
    re.IGNORECASE,
)


def _is_appendix_line(value: str) -> bool:
    text = value.strip()
    if len(text) > 90 or not _APPENDIX_RE.match(text):
        return False
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(character.isupper() for character in letters) / len(letters)
    return upper_ratio >= 0.7


def _clean_title(value: str) -> str:
    title = re.sub(r"\s+", " ", value).strip()
    title = title.strip(" .:;-–—")
    return title[:180]


def _split_inline_title(rest: str) -> tuple[str, str]:
    match = _INLINE_TITLE_RE.match(rest)
    if not match:
        return "", rest
    title = _clean_title(match.group("title"))
    tail = match.group("rest").strip()
    words = title.split()
    if not (1 <= len(words) <= 16) or len(title) > 130:
        return "", rest
    if title.endswith(",") or "(" in title or ")" in title:
        return "", rest
    lowered = title.lower()
    if lowered.startswith(
        (
            "in this act",
            "in this sanhita",
            "in this code",
            "unless the context",
            "save as otherwise",
            "nothing in this",
            "subject to",
        )
    ):
        return "", rest
    return title, tail


def _parse_toc(prefix_lines: list[PdfLine]) -> dict[str, str]:
    titles: dict[str, str] = {}
    pending_number: str | None = None
    pending_parts: list[str] = []
    stop = re.compile(
        r"statement\s+of\s+objects|notes\s+on\s+clauses",
        re.IGNORECASE,
    )

    def flush() -> None:
        nonlocal pending_number, pending_parts
        if pending_number and pending_parts:
            title = _clean_title(" ".join(pending_parts))
            if 2 <= len(title) <= 180:
                titles.setdefault(pending_number, title)
        pending_number = None
        pending_parts = []

    for line in prefix_lines:
        value = line.text.strip()
        if stop.search(value):
            break
        if re.match(r"^(chapter|section|clauses?)\b", value, re.IGNORECASE):
            flush()
            continue
        number_only = re.match(r"^(\d{1,3}[A-Z]{0,2})\.\s*$", value)
        if number_only:
            flush()
            pending_number = number_only.group(1)
            continue
        inline = re.match(r"^(\d{1,3}[A-Z]{0,2})\.\s+(.+)$", value)
        if inline:
            flush()
            title = _clean_title(inline.group(2))
            if 2 <= len(title) <= 180 and not re.fullmatch(r"[\d\s.]+", title):
                titles.setdefault(inline.group(1), title)
            continue
        if pending_number:
            if (
                len(value) > 100
                or value.isupper()
                or re.match(
                    r"^(note|copyright|corresponding|comparative|statement|index|see|also)\b",
                    value,
                    re.IGNORECASE,
                )
            ):
                flush()
                continue
            pending_parts.append(value)
    flush()
    return titles


def _find_body_start(lines: list[PdfLine]) -> int:
    body_marker = re.compile(
        r"\b(?:this\s+)?(?:act|sanhita|code)\s+may\s+be\s+called\b", re.IGNORECASE
    )
    for position, line in enumerate(lines):
        if body_marker.search(line.text):
            return position
    for position, line in enumerate(lines):
        if re.match(r"^1\.\s", line.text):
            return position
    return 0


def _page_bounds(page_lines: list[PdfLine]) -> tuple[float, float]:
    wide = [line for line in page_lines if (line.x1 - line.x0) > 200]
    if wide:
        return min(line.x0 for line in wide), max(line.x1 for line in wide)
    if not page_lines:
        return 0.0, 0.0
    return min(line.x0 for line in page_lines), max(line.x1 for line in page_lines)


_MARGIN_FOOTNOTE_RE = re.compile(
    r"^\s*\d{1,4}\s+of\s+\d{4}\.?\s*$|^\s*\[\s*\d+\s*\]\s*$", re.IGNORECASE
)


def _margin_titles_for_headings(
    margin_lines: list[PdfLine],
    headings: list[dict[str, Any]],
) -> dict[int, str]:
    """Map ``heading line idx -> margin title`` for Acts with side headings."""
    by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for heading in headings:
        by_page[heading["line"].page].append(heading)

    collected: dict[int, list[PdfLine]] = defaultdict(list)
    for line in margin_lines:
        if re.fullmatch(r"[\d\s.]+", line.text) or _MARGIN_FOOTNOTE_RE.match(line.text):
            continue
        best: tuple[float, dict[str, Any]] | None = None
        for heading in by_page.get(line.page, []):
            heading_line = heading["line"]
            if line.y0 < heading_line.y0 - 30:
                continue
            distance = abs(line.y0 - heading_line.y0)
            if distance <= 100 and (best is None or distance < best[0]):
                best = (distance, heading)
        if best is not None:
            collected[best[1]["line"].idx].append(line)

    output: dict[int, str] = {}
    for key, lines in collected.items():
        lines.sort(key=lambda item: item.y0)
        title = _clean_title(" ".join(item.text for item in lines))
        if 2 <= len(title) <= 180:
            output[key] = title
    return output


def _build_bare_act_chunks(
    spec: PDFSpec,
    raw_lines: list[PdfLine],
    page_count: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    cleaned = _clean_layout_lines(raw_lines, page_count)
    if not cleaned:
        return []

    pages: dict[int, list[PdfLine]] = defaultdict(list)
    for line in cleaned:
        pages[line.page].append(line)

    margin_ids: set[int] = set()
    for page_lines in pages.values():
        left, right = _page_bounds(page_lines)
        if not left and not right:
            continue
        for line in page_lines:
            if line.x1 < left - 6 or line.x0 > right + 4:
                margin_ids.add(line.idx)

    body_start = _find_body_start(cleaned)
    toc_titles = _parse_toc(cleaned[:body_start])

    body_lines = [line for line in cleaned[body_start:] if line.idx not in margin_ids]
    margin_lines = [line for line in cleaned[body_start:] if line.idx in margin_ids]

    # Schedules, statements of objects, memoranda, and the Hindi version at the
    # end of some gazettes are not sections. Stop the bare-act track there.
    limit = len(body_lines)
    for position, line in enumerate(body_lines):
        if _is_appendix_line(line.text):
            limit = position
            break

    headings: list[dict[str, Any]] = []
    last_number = 0
    last_suffix = ""
    for position, line in enumerate(body_lines[:limit]):
        match = _HEADING_RE.match(line.text)
        if not match:
            continue
        number = int(match.group(1))
        suffix = match.group(2) or ""
        rest = match.group(3).strip()
        if number > 999 or _AMENDMENT_HEADING_RE.search(line.text):
            continue

        inline_title, inline_rest = _split_inline_title(rest)
        if not headings:
            accepted = (
                number == 1
                and not suffix
                and (line.bold or bool(inline_title) or rest.startswith("("))
            )
        else:
            sequential = (
                (number == last_number + 1 and not suffix)
                or (number == last_number and bool(suffix) and suffix > last_suffix)
                or (last_number < number <= last_number + 2)
            )
            accepted = sequential and (
                line.bold or bool(inline_title) or number == last_number + 1
            )
        if not accepted:
            continue

        headings.append(
            {
                "position": position,
                "line": line,
                "number": number,
                "suffix": suffix,
                "section": f"{number}{suffix}",
                "rest": rest,
                "inline_title": inline_title,
                "inline_rest": inline_rest,
            }
        )
        last_number = number
        last_suffix = suffix

    margin_titles = _margin_titles_for_headings(margin_lines, headings)

    consumed_title_ids: set[int] = set()
    sections: list[dict[str, Any]] = []
    for index, heading in enumerate(headings):
        position = heading["position"]
        next_position = (
            headings[index + 1]["position"] if index + 1 < len(headings) else limit
        )

        preceding, consumed = _preceding_title(body_lines, position, heading["line"])
        consumed_title_ids.update(consumed)

        title = (
            heading["inline_title"]
            or toc_titles.get(heading["section"], "")
            or preceding
            or margin_titles.get(heading["line"].idx, "")
        )

        parts = []
        first = heading["inline_rest"] or heading["rest"]
        if first:
            parts.append(first)
        for line in body_lines[position + 1 : next_position]:
            if line.idx in consumed_title_ids:
                continue
            if re.match(r"^CHAPTER\b", line.text, re.IGNORECASE):
                continue
            if _AMENDMENT_FOOTNOTE_RE.search(line.text):
                continue
            parts.append(line.text)
        body = _join_lines(parts)
        if len(body) < 25:
            continue

        if not title:
            title = _fallback_title(heading["section"], body)
        title = _clean_title(title)

        label = f"{spec.short_code} - Section {heading['section']}"
        if title:
            label += f": {title}"
        header = f"[{label}]"

        chunks = _split_long_text(body, max_chars)
        total = len(chunks)
        for part_number, part_body in enumerate(chunks, start=1):
            part_marker = f" [part {part_number}/{total}]" if total > 1 else ""
            embedded_text = f"{header}{part_marker}\n{part_body}"
            source_id = f"pdf-{_slug(spec.short_code)}-s{heading['section']}"
            chunk_id = source_id if total == 1 else f"{source_id}-p{part_number}"
            sections.append(
                {
                    "id": chunk_id,
                    "source_id": source_id,
                    "doc_type": "bare_act",
                    "act_name": spec.act_name,
                    "short_code": spec.short_code,
                    "section": heading["section"],
                    "title": f"Section {heading['section']}: {title}"
                    if title
                    else f"Section {heading['section']}",
                    "source_name": spec.act_name,
                    "domain": spec.domain,
                    "category": "Bare Act",
                    "jurisdiction": spec.jurisdiction,
                    "year": spec.year,
                    "citation": f"{spec.short_code}, s.{heading['section']}",
                    "snippet": _clip(part_body, 420),
                    "passages": [],
                    "keywords": "",
                    "text": embedded_text,
                    "part": part_number,
                }
            )
    return sections


def _preceding_title(
    body_lines: list[PdfLine], position: int, heading: PdfLine
) -> tuple[str, list[int]]:
    """Collect the bold title lines printed immediately above a BNS-style heading."""
    collected: list[PdfLine] = []
    cursor = position - 1
    reference_y = heading.y0

    while cursor >= 0 and len(collected) < 2:
        candidate = body_lines[cursor]
        if not candidate.bold:
            break
        if _HEADING_RE.match(candidate.text) or re.match(
            r"^(chapter|section|clauses?)\b", candidate.text, re.IGNORECASE
        ):
            break
        if candidate.text.isupper() or len(candidate.text) > 180:
            break
        if candidate.text.rstrip().endswith((":", ";")):
            break
        if candidate.text[:1].isupper():
            pass
        elif collected:
            pass
        else:
            # A lowercase line immediately above the heading is usually the
            # tail of the previous section, not the start of a title.
            break
        if candidate.page == heading.page:
            if reference_y - candidate.y1 > 45:
                break
        elif candidate.page != heading.page - 1:
            break
        collected.append(candidate)
        reference_y = candidate.y0
        cursor -= 1

    if not collected:
        return "", []
    collected.reverse()
    title = _clean_title(" ".join(line.text for line in collected))
    return title, [line.idx for line in collected]


def _fallback_title(section: str, body: str) -> str:
    if section == "1":
        return "Short title, extent and commencement"
    if re.search(r"in this (?:act|sanhita|code),\s*unless the context", body[:240], re.I):
        return "Definitions"
    return ""


def _join_lines(parts: Iterable[str]) -> str:
    output = ""
    for part in parts:
        value = re.sub(r"\s+", " ", part).strip()
        if not value:
            continue
        if not output:
            output = value
        elif output.endswith("-") and value[:1].islower():
            output = output[:-1] + value
        else:
            output += " " + value
    return re.sub(r"\s+", " ", output).strip()


def _clip(value: str, limit: int) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def _split_long_text(text: str, limit: int, overlap: int = 120) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + limit, length)
        if end < length:
            cut = text.rfind(" ", start + limit // 2, end)
            if cut > start:
                end = cut
        parts.append(text[start:end].strip())
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return [part for part in parts if part]


# ---------------------------------------------------------------------------
# Handbook chunking
# ---------------------------------------------------------------------------


def _build_handbook_chunks(
    spec: PDFSpec,
    page_texts: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "langchain-text-splitters is not installed. "
            "Run: python -m pip install langchain-text-splitters"
        ) from exc

    text = _clean_page_texts(page_texts)
    if not text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\nQ.", "\nStep ", "\n"],
        keep_separator=True,
    )
    pieces = splitter.split_text(text)

    header = f"[Guide: {spec.source_name}]"
    source_id = f"pdf-guide-{_slug(spec.short_code or spec.source_name)}"
    chunks: list[dict[str, Any]] = []
    for part_number, piece in enumerate(pieces, start=1):
        body = piece.strip()
        if len(body) < 40:
            continue
        chunks.append(
            {
                "id": f"{source_id}-c{part_number:04d}",
                "source_id": source_id,
                "doc_type": "handbook",
                "act_name": spec.act_name,
                "short_code": spec.short_code,
                "section": "",
                "title": spec.source_name,
                "source_name": spec.source_name,
                "domain": spec.domain,
                "category": "Handbook",
                "jurisdiction": spec.jurisdiction,
                "year": spec.year,
                "citation": spec.short_code or spec.source_name,
                "snippet": _clip(body, 420),
                "passages": [],
                "keywords": "",
                "text": f"{header}\n{body}",
                "part": part_number,
            }
        )
    return chunks


# ---------------------------------------------------------------------------
# Public builder + local cache
# ---------------------------------------------------------------------------


def build_pdf_chunks(
    docs_dir: str | Path = DEFAULT_DOCS_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    *,
    max_chars: int = DEFAULT_BARE_ACT_MAX_CHARS,
    chunk_size: int = DEFAULT_HANDBOOK_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_HANDBOOK_CHUNK_OVERLAP,
    verbose: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Extract, clean and chunk every configured PDF.

    Returns ``(chunks, warnings)``. Missing dependencies raise ``RuntimeError``
    with the exact ``pip install`` command.
    """
    specs, warnings = resolve_pdf_specs(docs_dir, manifest_path)
    chunks: list[dict[str, Any]] = []

    for spec in specs:
        try:
            if spec.doc_type == "bare_act":
                lines, page_count = _extract_layout_lines(spec.path)
                document_chunks = _build_bare_act_chunks(spec, lines, page_count, max_chars)
            else:
                page_texts = _extract_page_texts(spec.path)
                document_chunks = _build_handbook_chunks(
                    spec, page_texts, chunk_size, chunk_overlap
                )
        except Exception as exc:
            warnings.append(f"Could not read {spec.path.name}: {exc}")
            continue

        if not document_chunks:
            warnings.append(f"No chunks were produced for {spec.path.name}.")
            continue
        if verbose:
            print(
                f"  {spec.path.name}: {len(document_chunks)} chunks "
                f"({spec.doc_type}, {spec.short_code})"
            )
        chunks.extend(document_chunks)

    return chunks, warnings


def save_chunks_cache(
    chunks: list[dict[str, Any]],
    path: str | Path = DEFAULT_CACHE_PATH,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "count": len(chunks),
        "chunks": chunks,
    }
    target.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return target


def load_cached_pdf_chunks(path: str | Path = DEFAULT_CACHE_PATH) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(payload, dict):
        chunks = payload.get("chunks")
    else:
        chunks = payload
    if not isinstance(chunks, list):
        return []
    return [chunk for chunk in chunks if isinstance(chunk, dict) and chunk.get("text")]
