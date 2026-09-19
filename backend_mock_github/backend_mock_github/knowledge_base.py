"""Safe loading, metadata extraction, and chunking for approved RAG sources.

This module deliberately has no awareness of assessment records.  Only paths
explicitly listed in ``knowledge_sources.yaml`` may be ingested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from hashlib import sha256
from pathlib import Path
import csv
import re
import zipfile
from xml.etree import ElementTree


SUPPORTED_SUFFIXES = {".md", ".txt", ".csv", ".pdf", ".docx", ".xlsx"}
BLOCKED_NAMES = {"intern_assessments.db", ".env", ".env.example"}


@dataclass(frozen=True)
class SourceConfig:
    path: Path
    category: str
    access_level: str


@dataclass(frozen=True)
class SourceDocument:
    source_path: str
    title: str
    text: str
    category: str
    access_level: str
    content_hash: str
    topics: tuple[str, ...] = ()
    effective_date: str | None = None
    version: str | None = None
    location: str | None = None


@dataclass(frozen=True)
class KnowledgeChunk:
    id: str
    source_path: str
    title: str
    text: str
    category: str
    access_level: str
    content_hash: str
    chunk_index: int
    topics: tuple[str, ...] = ()
    effective_date: str | None = None
    version: str | None = None
    location: str | None = None


def parse_source_config(path: str | Path) -> tuple[SourceConfig, ...]:
    """Read the intentionally small YAML subset used by knowledge_sources.yaml."""
    config_path = Path(path)
    entries: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw in config_path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line == "knowledge_sources:":
            continue
        if line.startswith("- "):
            if current:
                entries.append(current)
            current = {}
            line = line[2:].strip()
        if current is not None and ":" in line:
            key, value = line.split(":", 1)
            current[key.strip()] = value.strip().strip("\"'")
    if current:
        entries.append(current)
    if not entries:
        raise ValueError("No knowledge_sources entries found in configuration.")
    base = config_path.parent
    sources = []
    for entry in entries:
        if not {"path", "category", "access_level"}.issubset(entry):
            raise ValueError("Every knowledge source needs path, category, and access_level.")
        candidate = (base / entry["path"]).resolve()
        if not candidate.is_dir():
            raise ValueError(f"Configured knowledge source is not a directory: {candidate}")
        sources.append(SourceConfig(candidate, entry["category"], entry["access_level"]))
    return tuple(sources)


def discover_documents(sources: tuple[SourceConfig, ...]) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    for source in sources:
        for file_path in source.path.rglob("*"):
            if not file_path.is_file() or not _safe_file(file_path) or file_path.suffix.casefold() not in SUPPORTED_SUFFIXES:
                continue
            documents.extend(_load_file(file_path, source))
    return documents


def chunk_document(document: SourceDocument, chunk_size: int = 900, overlap: int = 150) -> list[KnowledgeChunk]:
    """Chunk at paragraph boundaries, without blending material from files."""
    if chunk_size < 200 or not 0 <= overlap < chunk_size:
        raise ValueError("chunk_size must be at least 200 and overlap must be smaller than it.")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", document.text) if p.strip()]
    pieces: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip()
        if current and len(candidate) > chunk_size:
            pieces.append(current)
            current = (current[-overlap:] + "\n\n" + paragraph).strip() if overlap else paragraph
        else:
            current = candidate
    if current:
        pieces.append(current)
    return [
        KnowledgeChunk(
            id=_hash(f"{document.source_path}:{document.content_hash}:{index}"),
            source_path=document.source_path,
            title=document.title,
            text=piece,
            category=document.category,
            access_level=document.access_level,
            content_hash=document.content_hash,
            chunk_index=index,
            topics=document.topics,
            effective_date=document.effective_date,
            version=document.version,
            location=document.location,
        )
        for index, piece in enumerate(pieces)
    ]


def _safe_file(path: Path) -> bool:
    lowered = {part.casefold() for part in path.parts}
    return path.name.casefold() not in BLOCKED_NAMES and not {".git", "__pycache__", "tests", "fixtures"}.intersection(lowered)


def _load_file(path: Path, source: SourceConfig) -> list[SourceDocument]:
    suffix = path.suffix.casefold()
    if suffix in {".md", ".txt"}:
        text, metadata = _parse_text(path.read_text(encoding="utf-8"))
        return [_document(path, source, text, metadata)]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            text = "\n".join(" | ".join(row) for row in csv.reader(handle) if any(cell.strip() for cell in row))
        return [_document(path, source, text, {})]
    if suffix == ".docx":
        return [_document(path, source, _read_docx(path), {})]
    if suffix == ".xlsx":
        return [_document(path, source, text, {"location": sheet}) for sheet, text in _read_xlsx(path)]
    if suffix == ".pdf":
        return [_document(path, source, text, {"location": page}) for page, text in _read_pdf(path)]
    return []


def _document(path: Path, source: SourceConfig, text: str, metadata: dict[str, str]) -> SourceDocument:
    cleaned = _clean(text)
    if not cleaned:
        return SourceDocument(str(path.resolve()), path.stem, "", source.category, source.access_level, _hash(""))
    title = metadata.get("title") or path.stem.replace("_", " ").replace("-", " ").title()
    topics = tuple(term.strip() for term in metadata.get("topics", "").split(",") if term.strip())
    return SourceDocument(str(path.resolve()), title, cleaned, source.category, source.access_level, _hash(cleaned), topics, metadata.get("effective_date"), metadata.get("version"), metadata.get("location"))


def _parse_text(text: str) -> tuple[str, dict[str, str]]:
    metadata: dict[str, str] = {}
    if text.startswith("---\n"):
        _, header, remainder = text.split("---\n", 2)
        for line in header.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()
        return remainder, metadata
    return text, metadata


def _read_docx(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    return "\n\n".join("".join(node.itertext()) for node in root.iter() if node.tag.endswith("}p"))


def _read_xlsx(path: Path) -> list[tuple[str, str]]:
    with zipfile.ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(node.itertext()) for node in root if node.tag.endswith("}si")]
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        sheet_names = [node.attrib.get("name", "Sheet") for node in workbook.iter() if node.tag.endswith("}sheet")]
        sheets = sorted(name for name in archive.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name))
        output = []
        for index, sheet_file in enumerate(sheets):
            root = ElementTree.fromstring(archive.read(sheet_file))
            rows = []
            for row in (node for node in root.iter() if node.tag.endswith("}row")):
                values = []
                for cell in (node for node in row if node.tag.endswith("}c")):
                    value = next((child.text for child in cell if child.tag.endswith("}v")), None)
                    if value and cell.attrib.get("t") == "s" and value.isdigit():
                        value = shared[int(value)]
                    if value and not value.replace(".", "", 1).isdigit():
                        values.append(value)
                if values:
                    rows.append(" | ".join(values))
            text = "\n".join(rows)
            if text:
                output.append((sheet_names[index] if index < len(sheet_names) else f"Sheet {index + 1}", text))
        return output


def _read_pdf(path: Path) -> list[tuple[str, str]]:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(f"PDF support requires optional dependency pypdf: {path}") from error
    reader = PdfReader(str(path))
    return [(f"page {index + 1}", page.extract_text() or "") for index, page in enumerate(reader.pages)]


def _clean(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n")).strip()


def _hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()
