from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ingest_knowledge_base import ingest
from knowledge_base import SourceConfig, chunk_document, discover_documents
from llm_generator import build_prompt
from rag_retriever import IndexedPolicyRetriever, KnowledgeDocument
from rule_engine import InternAssessment, evaluate_assessment
from vector_store import DeterministicEmbeddingClient, JsonVectorStore, PolicyRetriever


class RagTests(unittest.TestCase):
    def test_chunk_retains_metadata(self) -> None:
        source = SourceConfig(Path("knowledge_base/training_materials").resolve(), "training", "internal")
        document = next(item for item in discover_documents((source,)) if "Excel" in item.title)
        chunk = chunk_document(document)[0]
        self.assertEqual(chunk.category, "training")
        self.assertEqual(chunk.access_level, "internal")
        self.assertIn("excel", tuple(topic.casefold() for topic in chunk.topics))

    def test_multiple_sources_and_no_database_ingestion(self) -> None:
        sources = (
            SourceConfig(Path("knowledge_base/training_materials").resolve(), "training", "internal"),
            SourceConfig(Path("knowledge_base/finance_policies").resolve(), "finance_policy", "internal"),
        )
        documents = discover_documents(sources)
        self.assertGreaterEqual(len(documents), 3)
        self.assertFalse(any(item.source_path.endswith("intern_assessments.db") for item in documents))

    def test_relevance_does_not_follow_priority(self) -> None:
        with TemporaryDirectory() as directory:
            store = JsonVectorStore(Path(directory) / "vectors.json")
            docs = discover_documents((SourceConfig(Path("knowledge_base/training_materials").resolve(), "training", "internal"), SourceConfig(Path("knowledge_base/finance_policies").resolve(), "finance_policy", "internal")))
            chunks = [chunk for doc in docs for chunk in chunk_document(doc)]
            client = DeterministicEmbeddingClient()
            store.upsert(chunks, client.embed([chunk.text for chunk in chunks]))
            result = PolicyRetriever(store, client).retrieve(("Excel", "Financial Modeling"), limit=3)
            self.assertTrue(result)
            self.assertIn("Excel", result[0].chunk.title)
            self.assertNotIn("Risk", " ".join(item.chunk.title for item in result))

    def test_unknown_topic_returns_no_results(self) -> None:
        with TemporaryDirectory() as directory:
            store = JsonVectorStore(Path(directory) / "vectors.json")
            docs = discover_documents((SourceConfig(Path("knowledge_base/training_materials").resolve(), "training", "internal"),))
            chunks = [chunk for doc in docs for chunk in chunk_document(doc)]
            client = DeterministicEmbeddingClient()
            store.upsert(chunks, client.embed([chunk.text for chunk in chunks]))
            self.assertEqual(PolicyRetriever(store, client).retrieve(("derivatives hedging",)), [])

    def test_prompt_has_citation_and_no_result_text(self) -> None:
        decision = evaluate_assessment(InternAssessment("T-1", "Test", 0.4, ("Excel",)))
        document = KnowledgeDocument("Excel", "Use error checks.", ("excel",), decision.priority, "/approved/excel.md", "page 1")
        self.assertIn("/approved/excel.md", build_prompt(decision, [document]))
        self.assertIn("No approved source", build_prompt(decision, []))

    def test_ingestion_is_idempotent(self) -> None:
        with TemporaryDirectory() as directory:
            config = Path("knowledge_sources.yaml").resolve()
            index = Path(directory) / "vectors.json"
            first = ingest(config, index)
            second = ingest(config, index)
            self.assertGreater(first["indexed"], 0)
            self.assertEqual(second["indexed"], 0)


if __name__ == "__main__":
    unittest.main()
