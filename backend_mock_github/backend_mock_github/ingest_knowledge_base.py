"""Command line ingestion for explicitly approved knowledge-base folders."""

from __future__ import annotations

import argparse
from pathlib import Path

from knowledge_base import chunk_document, discover_documents, parse_source_config
from vector_store import JsonVectorStore, configured_embedding_client


def ingest(config_path: str | Path, index_path: str | Path = "data/policy_vectors.json") -> dict[str, int]:
    sources = parse_source_config(config_path)
    store = JsonVectorStore(index_path)
    documents = discover_documents(sources)
    eligible = [document for document in documents if document.text]
    chunks = [chunk for document in eligible for chunk in chunk_document(document)]
    indexed_documents = store.document_hashes()
    changed_documents = [document for document in eligible if document.content_hash not in indexed_documents.get(document.source_path, set())]
    changed_paths = {document.source_path for document in changed_documents}
    changed = [chunk for chunk in chunks if chunk.source_path in changed_paths]
    indexed = store.upsert(changed, configured_embedding_client().embed([chunk.text for chunk in changed]), changed_paths) if changed else 0
    return {"files": len(documents), "chunks": len(chunks), "indexed": indexed, "skipped": len(chunks) - len(changed), "failed": len(documents) - len(eligible)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Index approved policy and training folders.")
    parser.add_argument("--config", default="knowledge_sources.yaml")
    parser.add_argument("--index", default="data/policy_vectors.json")
    args = parser.parse_args()
    try:
        summary = ingest(args.config, args.index)
    except Exception as error:
        raise SystemExit(f"Ingestion failed: {error}") from error
    print("Ingestion complete: " + ", ".join(f"{key}={value}" for key, value in summary.items()))


if __name__ == "__main__":
    main()
