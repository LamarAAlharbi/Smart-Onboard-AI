"""Provider-neutral embeddings, a development vector store, and policy retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
from typing import Protocol, Sequence

from knowledge_base import KnowledgeChunk


class EmbeddingClient(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[KnowledgeChunk], embeddings: Sequence[Sequence[float]]) -> int: ...
    def search(self, embedding: Sequence[float], limit: int) -> list[tuple[KnowledgeChunk, float]]: ...


class DeterministicEmbeddingClient:
    """Credential-free hashed-token vectors for development and repeatable tests."""
    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in re.findall(r"[a-z0-9]+", text.casefold()):
                vector[int(sha256(token.encode()).hexdigest(), 16) % self.dimensions] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


class OpenAIEmbeddingClient:
    """Optional production adapter. Configuration is supplied only through env vars."""
    def __init__(self) -> None:
        self.model = os.environ.get("RAG_EMBEDDING_MODEL")
        if not self.model:
            raise ValueError("RAG_EMBEDDING_MODEL is required for the OpenAI embedding adapter.")
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError("Install the optional openai package to use this embedding adapter.") from error
        self.client = OpenAI()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=list(texts))
        return [item.embedding for item in response.data]


def configured_embedding_client() -> EmbeddingClient:
    provider = os.environ.get("RAG_EMBEDDING_PROVIDER", "offline").casefold()
    if provider == "offline":
        return DeterministicEmbeddingClient()
    if provider == "openai":
        return OpenAIEmbeddingClient()
    raise ValueError(f"Unsupported RAG_EMBEDDING_PROVIDER: {provider}")


@dataclass(frozen=True)
class StoredVector:
    chunk: KnowledgeChunk
    embedding: list[float]


class JsonVectorStore:
    """Persistent local development store; use pgvector behind this boundary in production."""
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._vectors = self._load()

    def upsert(self, chunks: Sequence[KnowledgeChunk], embeddings: Sequence[Sequence[float]], replace_sources: set[str] | None = None) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        indexed = {item.chunk.id: item for item in self._vectors if not replace_sources or item.chunk.source_path not in replace_sources}
        for chunk, embedding in zip(chunks, embeddings):
            indexed[chunk.id] = StoredVector(chunk, list(embedding))
        self._vectors = list(indexed.values())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([{"chunk": asdict(item.chunk), "embedding": item.embedding} for item in self._vectors]), encoding="utf-8")
        return len(chunks)

    def indexed_hashes(self) -> set[str]:
        return {item.chunk.content_hash for item in self._vectors}

    def document_hashes(self) -> dict[str, set[str]]:
        hashes: dict[str, set[str]] = {}
        for item in self._vectors:
            hashes.setdefault(item.chunk.source_path, set()).add(item.chunk.content_hash)
        return hashes

    def search(self, embedding: Sequence[float], limit: int) -> list[tuple[KnowledgeChunk, float]]:
        matches = [(item.chunk, _cosine(embedding, item.embedding)) for item in self._vectors]
        return sorted(matches, key=lambda item: item[1], reverse=True)[:limit]

    def _load(self) -> list[StoredVector]:
        if not self.path.exists():
            return []
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return [StoredVector(KnowledgeChunk(**{**entry["chunk"], "topics": tuple(entry["chunk"].get("topics", ())) }), entry["embedding"]) for entry in raw]


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: KnowledgeChunk
    score: float

    @property
    def citation(self) -> str:
        location = f", {self.chunk.location}" if self.chunk.location else ""
        return f"{self.chunk.title} ({self.chunk.source_path}{location})"


class PolicyRetriever:
    def __init__(self, store: VectorStore, embeddings: EmbeddingClient, minimum_score: float = 0.05) -> None:
        self.store, self.embeddings, self.minimum_score = store, embeddings, minimum_score

    def retrieve(self, weak_topics: tuple[str, ...], priority: object | None = None, limit: int = 3, categories: set[str] | None = None, access_levels: set[str] | None = None) -> list[RetrievedChunk]:
        """Priority is accepted for pipeline compatibility but never changes relevance."""
        query = " ".join(weak_topics)
        query_terms = set(re.findall(r"[a-z0-9]+", query.casefold()))
        if not query.strip():
            return []
        candidates = self.store.search(self.embeddings.embed([query])[0], max(limit * 5, 10))
        seen_documents: set[tuple[str, int]] = set()
        results = []
        for chunk, score in candidates:
            if score < self.minimum_score or not _topically_related(chunk, query, query_terms) or (categories and chunk.category not in categories) or (access_levels and chunk.access_level not in access_levels):
                continue
            if chunk.effective_date and chunk.effective_date > date.today().isoformat():
                continue
            key = (chunk.source_path, chunk.chunk_index)
            if key not in seen_documents:
                results.append(RetrievedChunk(chunk, score))
                seen_documents.add(key)
            if len(results) == limit:
                break
        return results


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def _topically_related(chunk: KnowledgeChunk, query: str, query_terms: set[str]) -> bool:
    """Use explicit document topics as a precision guard around semantic ranking."""
    if chunk.topics:
        return any(topic.casefold() in query.casefold() or query_terms.intersection(re.findall(r"[a-z0-9]+", topic.casefold())) for topic in chunk.topics)
    return bool(query_terms.intersection(re.findall(r"[a-z0-9]+", chunk.text.casefold())))
