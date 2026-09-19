"""A deterministic, in-memory retriever that can later be replaced by a vector DB."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from rule_engine import Priority


@dataclass(frozen=True)
class KnowledgeDocument:
    title: str
    content: str
    topics: tuple[str, ...]
    minimum_priority: Priority
    source_path: str = "built-in demo"
    location: str | None = None
    score: float | None = None


class Retriever(Protocol):
    def retrieve(self, weak_topics: tuple[str, ...], priority: Priority, limit: int = 3) -> list[KnowledgeDocument]: ...


DEFAULT_KNOWLEDGE_BASE: tuple[KnowledgeDocument, ...] = (
    KnowledgeDocument("Financial Statement Fundamentals", "Review the income statement, balance sheet, and cash-flow statement; reconcile their connections with a worked example.", ("financial statements", "accounting", "cash flow"), Priority.LOW),
    KnowledgeDocument("Excel Controls for Finance", "Practice XLOOKUP, SUMIFS, error checks, and formula-audit techniques before submitting any model.", ("excel", "financial modeling", "spreadsheets"), Priority.LOW),
    KnowledgeDocument("Valuation Essentials", "Build a simple DCF, state assumptions explicitly, and compare implied values against trading multiples.", ("valuation", "dcf", "multiples"), Priority.MEDIUM),
    KnowledgeDocument("Risk and Compliance Escalation", "For material control gaps, document the issue, notify the supervisor promptly, and follow the remediation checklist.", ("risk", "compliance", "controls"), Priority.HIGH),
)


class KnowledgeBaseRetriever:
    """Uses the indexed knowledge base when available, with local samples as a fallback.

    Passing ``documents`` explicitly keeps the deterministic in-memory behavior useful
    for isolated demos and tests.
    """

    def __init__(
        self,
        documents: Iterable[KnowledgeDocument] | None = None,
        index_path: str | Path | None = None,
    ) -> None:
        self.documents = tuple(documents) if documents is not None else DEFAULT_KNOWLEDGE_BASE
        self._indexed_retriever: IndexedPolicyRetriever | None = None

        if documents is None:
            resolved_index = Path(index_path) if index_path is not None else Path(__file__).with_name("data") / "policy_vectors.json"
            if resolved_index.exists():
                self._indexed_retriever = IndexedPolicyRetriever(resolved_index)

    def retrieve(self, weak_topics: tuple[str, ...], priority: Priority, limit: int = 3) -> list[KnowledgeDocument]:
        """Return the highest-relevance materials for the intern's gaps."""
        if limit < 1:
            raise ValueError("limit must be at least 1.")
        if self._indexed_retriever is not None:
            return self._indexed_retriever.retrieve(weak_topics, priority, limit)

        topic_terms = {topic.casefold() for topic in weak_topics}
        ranked: list[tuple[int, KnowledgeDocument]] = []
        for document in self.documents:
            overlap = len(topic_terms.intersection(term.casefold() for term in document.topics))
            # Priority controls the development response, never topic relevance.
            if overlap:
                ranked.append((overlap * 10, document))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [document for _, document in ranked[:limit]]


class IndexedPolicyRetriever:
    """Pipeline-compatible adapter around the persistent policy vector index."""

    def __init__(self, index_path: str = "data/policy_vectors.json", minimum_score: float = 0.05) -> None:
        from vector_store import JsonVectorStore, PolicyRetriever, configured_embedding_client

        self._retriever = PolicyRetriever(JsonVectorStore(index_path), configured_embedding_client(), minimum_score)

    def retrieve(self, weak_topics: tuple[str, ...], priority: Priority, limit: int = 3) -> list[KnowledgeDocument]:
        return [
            KnowledgeDocument(
                title=result.chunk.title,
                content=result.chunk.text,
                topics=result.chunk.topics,
                minimum_priority=Priority.LOW,
                source_path=result.chunk.source_path,
                location=result.chunk.location,
                score=result.score,
            )
            for result in self._retriever.retrieve(weak_topics, priority, limit)
        ]
