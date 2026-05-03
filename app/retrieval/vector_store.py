"""
INTERVIEW: "Why ChromaDB over Pinecone/Weaviate/Qdrant?"
1. Free: runs fully locally, no cloud account needed
2. Persistent: data survives restarts (unlike in-memory FAISS)
3. Metadata filtering: supports WHERE clause filtering on chunk metadata
4. Python-native: no separate server process needed (embedded mode)
5. API compatible: same API as Pinecone for common operations — swap later

INTERVIEW: "How does ChromaDB store vectors?"
Uses HNSW (Hierarchical Navigable Small World) for ANN search.
HNSW builds a multi-layer graph where each node links to its nearest neighbors.
Search is O(log n) — significantly faster than brute-force O(n*d) dot product.

INTERVIEW: "How would you scale beyond ChromaDB?"
ChromaDB has a server mode for multi-process access.
For true scale: Qdrant (binary quantization, lower memory) or Weaviate (hybrid built-in).
The ChromaDB client API mirrors others closely — adapter pattern makes it swappable.

INTERVIEW: "How do you handle duplicate documents?"
We use the file hash (SHA256) as part of the doc_id.
On ingest, we check if doc_id exists in metadata — if yes, skip or overwrite.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import chromadb
import numpy as np
from chromadb import Collection
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Wrapper around ChromaDB with two collections:
    - research_papers: for academic PDFs
    - dev_docs: for web documentation

    DESIGN CHOICE: Two collections instead of one with a 'mode' filter.
    Separate collections means:
    1. No cross-contamination (research queries never see docs chunks)
    2. Different distance metrics possible per collection
    3. Easier to clear/rebuild one without affecting the other
    """

    def __init__(self):
        settings = get_settings()
        # DESIGN CHOICE: Persistent client — data survives process restarts
        self._client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(
                anonymized_telemetry=False,  # Don't send usage data
                allow_reset=True,
            ),
        )
        self._research_collection: Collection | None = None
        self._docs_collection: Collection | None = None
        self._settings = settings

    def initialize(self) -> None:
        """
        Create or get existing collections.
        Called once on startup via lifespan.
        INTERVIEW: "Why get_or_create instead of create?"
        Idempotent startup — if the app restarts, existing data is preserved.
        """
        # cosine distance is standard for text embeddings
        # (normalized vectors → cosine similarity = dot product)
        self._research_collection = self._client.get_or_create_collection(
            name=self._settings.CHROMA_RESEARCH_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._docs_collection = self._client.get_or_create_collection(
            name=self._settings.CHROMA_DOCS_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB initialized. "
            f"Research: {self._research_collection.count()} docs. "
            f"DevDocs: {self._docs_collection.count()} docs."
        )

    def get_collection(self, mode: str) -> Collection:
        """Get the appropriate collection for the mode."""
        if mode == "research":
            if self._research_collection is None:
                raise RuntimeError("VectorStore not initialized. Call initialize() first.")
            return self._research_collection
        elif mode == "docs":
            if self._docs_collection is None:
                raise RuntimeError("VectorStore not initialized. Call initialize() first.")
            return self._docs_collection
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'research' or 'docs'.")

    async def add_chunks(
            self,
            mode: str,
            chunk_ids: list[str],
            embeddings: list[list[float]],
            documents: list[str],
            metadatas: list[dict[str, Any]],
    ) -> None:
        """
        Add chunks to ChromaDB in batches.
        INTERVIEW: "Why batch inserts?"
        ChromaDB has an internal batch size limit (~41k items).
        Batching also keeps memory usage bounded during large ingestion jobs.
        """
        collection = self.get_collection(mode)
        batch_size = 500  # Safe batch size for ChromaDB

        loop = asyncio.get_event_loop()

        for i in range(0, len(chunk_ids), batch_size):
            batch_ids = chunk_ids[i: i + batch_size]
            batch_embeddings = embeddings[i: i + batch_size]
            batch_docs = documents[i: i + batch_size]
            batch_metas = metadatas[i: i + batch_size]

            # Sanitize metadata — ChromaDB only supports str, int, float, bool
            batch_metas = [self._sanitize_metadata(m) for m in batch_metas]

            # Run in executor (ChromaDB is sync)
            await loop.run_in_executor(
                None,
                lambda: collection.upsert(  # upsert = insert or update
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_docs,
                    metadatas=batch_metas,
                ),
            )
            logger.debug(f"Upserted batch {i // batch_size + 1}: {len(batch_ids)} chunks")

    async def query_dense(
            self,
            mode: str,
            query_embedding: list[float],
            top_k: int = 20,
            where: dict | None = None,
    ) -> dict[str, Any]:
        """
        Dense vector search (ANN with HNSW).
        INTERVIEW: "What's the difference between dense and sparse retrieval?"
        Dense: query and documents embedded into same vector space.
          Captures semantic similarity — "car" ~ "automobile" ✓
          Misses exact keyword matches — "GPT-4" might not match "GPT4"
        Sparse (BM25): term frequency/inverse document frequency.
          Perfect keyword matching — "GPT-4" matches "GPT-4" exactly
          Misses semantic similarity — "car" ≠ "automobile" ✗
        Hybrid combines both for best of both worlds.
        """
        collection = self.get_collection(mode)
        loop = asyncio.get_event_loop()

        results = await loop.run_in_executor(
            None,
            lambda: collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, collection.count() or 1),
                where=where,
                include=["documents", "metadatas", "distances", "embeddings"],
            ),
        )
        return results

    async def document_exists(self, mode: str, doc_id: str) -> bool:
        """Check if a document is already ingested (deduplication)."""
        collection = self.get_collection(mode)
        loop = asyncio.get_event_loop()

        results = await loop.run_in_executor(
            None,
            lambda: collection.get(
                where={"doc_id": doc_id},
                limit=1,
                include=[],
            ),
        )
        return len(results["ids"]) > 0

    async def delete_document(self, mode: str, doc_id: str) -> int:
        """Delete all chunks for a document."""
        collection = self.get_collection(mode)
        loop = asyncio.get_event_loop()

        existing = await loop.run_in_executor(
            None,
            lambda: collection.get(where={"doc_id": doc_id}, include=[]),
        )
        ids_to_delete = existing["ids"]

        if ids_to_delete:
            await loop.run_in_executor(
                None,
                lambda: collection.delete(ids=ids_to_delete),
            )

        return len(ids_to_delete)

    def get_all_documents_text(self, mode: str) -> list[str]:
        """
        Get all document texts for BM25 index building.
        INTERVIEW: "Why do you need to get all texts?"
        BM25 needs to know the entire corpus to calculate IDF (inverse document frequency).
        IDF = log(N / df) where N = total docs, df = docs containing the term.
        You can't compute IDF without seeing all documents.
        """
        collection = self.get_collection(mode)
        results = collection.get(include=["documents"])
        return results["documents"] or []

    def get_all_documents_with_ids(self, mode: str) -> tuple[list[str], list[str]]:
        """Get all (ids, documents) for BM25 index building."""
        collection = self.get_collection(mode)
        results = collection.get(include=["documents"])
        return results["ids"] or [], results["documents"] or []

    def count(self, mode: str) -> int:
        """Get document count for a collection."""
        try:
            return self.get_collection(mode).count()
        except Exception:
            return 0

    @staticmethod
    def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
        """
        ChromaDB only supports str, int, float, bool in metadata.
        Convert lists to strings, None to "null".
        INTERVIEW: "What metadata limitations does ChromaDB have?"
        No nested objects, no lists. Everything must be a scalar.
        We serialize lists as JSON strings and parse on retrieval.
        """
        import json
        clean: dict[str, Any] = {}
        for k, v in meta.items():
            if v is None:
                clean[k] = "null"
            elif isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif isinstance(v, (list, dict)):
                clean[k] = json.dumps(v)
            else:
                clean[k] = str(v)
        return clean