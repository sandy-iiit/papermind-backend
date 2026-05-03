"""
INTERVIEW: "Explain your hybrid retrieval pipeline end to end."

Step 1 — Dense retrieval (ChromaDB HNSW):
  Embed query with sentence-transformers.
  ANN search returns top-K chunks by cosine similarity.

Step 2 — Sparse retrieval (BM25):
  Tokenize query and corpus.
  BM25 ranks chunks by term overlap + IDF weighting.

Step 3 — RRF Fusion (Reciprocal Rank Fusion):
  Each retriever produces a ranking (1st, 2nd, 3rd...).
  RRF score = Σ 1/(k + rank) for each retriever.
  k=60 is the standard RRF hyperparameter (empirically best).

  INTERVIEW: "Why RRF over weighted average?"
  Weighted average is sensitive to score scale differences between retrieval methods.
  BM25 scores range 0-15, cosine distances 0-2 — raw combination is meaningless.
  RRF only uses RANK POSITION, not raw scores — scale-invariant and robust.
  You don't need to tune weights (k=60 just works for most corpora).

Step 4 — Cross-encoder reranking (see reranker.py):
  Pass top-K candidates through cross-encoder for precise relevance scoring.
  Return top-N for LLM context.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.models.schemas import RetrievedChunk
from app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Combines BM25 (sparse) + sentence-transformers (dense) with RRF fusion.

    DESIGN CHOICE: Build BM25 index lazily on first query, then cache.
    BM25 index is rebuilt when new documents are ingested.
    For production: persist BM25 index to disk (pickle) to avoid recomputing.
    """

    RRF_K = 60  # Standard RRF hyperparameter

    def __init__(self, vector_store: VectorStore):
        settings = get_settings()
        self._vector_store = vector_store
        self._settings = settings

        # Load embedding model once — expensive, keep in memory
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        self._embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")

        # BM25 indexes — rebuilt when documents change
        self._bm25_indexes: dict[str, BM25Okapi | None] = {
            "research": None,
            "docs": None,
        }
        self._bm25_chunk_ids: dict[str, list[str]] = {
            "research": [],
            "docs": [],
        }
        self._bm25_documents: dict[str, list[str]] = {
            "research": [],
            "docs": [],
        }

    async def get_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for query text.
        INTERVIEW: "Why run embeddings in an executor?"
        sentence-transformers uses PyTorch — CPU-bound operation.
        Running it directly in an async function would block the event loop,
        making all other requests wait. Executor moves it to a thread.
        For production: use GPU + batching for higher throughput.
        """
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self._embedding_model.encode(
                text, normalize_embeddings=True  # Unit vector for cosine similarity
            ).tolist(),
        )
        return embedding

    async def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embedding — much faster than one by one."""
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._embedding_model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False,
            ).tolist(),
        )
        return embeddings

    async def build_bm25_index(self, mode: str) -> None:
        """
        Build BM25 index from all documents in the collection.
        INTERVIEW: "When do you rebuild the BM25 index?"
        After every ingestion. For large corpora, use incremental updates.
        BM25 is CPU-fast to build (~1s for 10k docs) so full rebuild is OK here.

        INTERVIEW: "How does BM25 work?"
        BM25 (Best Match 25) scores term relevance.
        score(q, d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
        k1: term frequency saturation (default 1.5 — repeated terms plateau quickly)
        b: length normalization (default 0.75 — penalize long documents)
        """
        loop = asyncio.get_event_loop()
        ids, documents = await loop.run_in_executor(
            None,
            lambda: self._vector_store.get_all_documents_with_ids(mode),
        )

        if not documents:
            logger.warning(f"No documents in {mode} collection — BM25 index empty")
            self._bm25_indexes[mode] = None
            return

        # Tokenize: lowercase, split by whitespace (simple but effective)
        tokenized_corpus = [doc.lower().split() for doc in documents]

        self._bm25_indexes[mode] = BM25Okapi(tokenized_corpus)
        self._bm25_chunk_ids[mode] = ids
        self._bm25_documents[mode] = documents

        logger.info(f"BM25 index built for {mode}: {len(documents)} documents")

    def _bm25_retrieve(
            self, query: str, mode: str, top_k: int
    ) -> list[tuple[str, float]]:
        """
        BM25 retrieval. Returns list of (chunk_id, score) tuples.
        """
        bm25 = self._bm25_indexes[mode]
        if bm25 is None or not self._bm25_chunk_ids[mode]:
            return []

        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        # Get top-K indices by score
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            (self._bm25_chunk_ids[mode][i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0.0  # Filter zero-score results
        ]

    async def retrieve(
            self,
            query: str,
            mode: str,
            top_k: int | None = None,
            rerank_top_n: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Full hybrid retrieval pipeline.
        Returns reranked top-N chunks ready for LLM context.
        """
        settings = self._settings
        top_k = top_k or settings.RETRIEVAL_TOP_K
        rerank_top_n = rerank_top_n or settings.RERANK_TOP_N

        if self._vector_store.count(mode) == 0:
            logger.warning(f"No documents in {mode} collection")
            return []

        # Build BM25 index if not built yet
        if self._bm25_indexes[mode] is None:
            await self.build_bm25_index(mode)

        # ── Step 1: Dense Retrieval ────────────────────────────────────────
        query_embedding = await self.get_embedding(query)
        dense_results = await self._vector_store.query_dense(
            mode=mode,
            query_embedding=query_embedding,
            top_k=top_k,
        )

        dense_chunk_ids: list[str] = dense_results["ids"][0] if dense_results["ids"] else []
        dense_distances: list[float] = dense_results["distances"][0] if dense_results["distances"] else []
        dense_documents: list[str] = dense_results["documents"][0] if dense_results["documents"] else []
        dense_metadatas: list[dict] = dense_results["metadatas"][0] if dense_results["metadatas"] else []

        # Convert cosine distance to similarity score (distance = 1 - similarity)
        dense_scores = {
            chunk_id: 1.0 - dist
            for chunk_id, dist in zip(dense_chunk_ids, dense_distances)
        }

        # ── Step 2: Sparse (BM25) Retrieval ───────────────────────────────
        bm25_results = self._bm25_retrieve(query, mode, top_k)
        bm25_scores = {chunk_id: score for chunk_id, score in bm25_results}

        # ── Step 3: RRF Fusion ─────────────────────────────────────────────
        fused_ids = self._rrf_fusion(
            rankings=[dense_chunk_ids, [r[0] for r in bm25_results]],
            top_k=top_k,
        )

        # ── Step 4: Build RetrievedChunk objects ───────────────────────────
        # Build a lookup for dense results
        dense_lookup = {
            cid: (doc, meta)
            for cid, doc, meta in zip(dense_chunk_ids, dense_documents, dense_metadatas)
        }

        # For chunks only in BM25 results, fetch from vector store
        bm25_only_ids = [
            cid for cid in fused_ids
            if cid not in dense_lookup
        ]
        if bm25_only_ids:
            loop = asyncio.get_event_loop()
            bm25_fetch = await loop.run_in_executor(
                None,
                lambda: self._vector_store.get_collection(mode).get(
                    ids=bm25_only_ids,
                    include=["documents", "metadatas"],
                ),
            )
            for cid, doc, meta in zip(
                    bm25_fetch["ids"],
                    bm25_fetch["documents"],
                    bm25_fetch["metadatas"],
            ):
                dense_lookup[cid] = (doc, meta)

        rrf_scores = self._compute_rrf_scores(fused_ids)

        chunks: list[RetrievedChunk] = []
        for chunk_id in fused_ids:
            if chunk_id not in dense_lookup:
                continue
            doc, meta = dense_lookup[chunk_id]
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    doc_id=meta.get("doc_id", ""),
                    content=doc,
                    metadata=meta,
                    dense_score=dense_scores.get(chunk_id, 0.0),
                    bm25_score=bm25_scores.get(chunk_id, 0.0),
                    rrf_score=rrf_scores.get(chunk_id, 0.0),
                )
            )

        return chunks

    def _rrf_fusion(
            self, rankings: list[list[str]], top_k: int
    ) -> list[str]:
        """
        Reciprocal Rank Fusion.
        INTERVIEW: "Why k=60 specifically?"
        Cormack et al. (2009) showed k=60 is robust across diverse retrieval tasks.
        It balances the contribution of high-ranked vs low-ranked results.
        Smaller k: top-1 dominates too strongly. Larger k: rankings flattened too much.
        """
        scores: dict[str, float] = defaultdict(float)

        for ranking in rankings:
            for rank, doc_id in enumerate(ranking):
                scores[doc_id] += 1.0 / (self.RRF_K + rank + 1)

        # Sort by fused score descending
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_ids[:top_k]

    def _compute_rrf_scores(self, ids: list[str]) -> dict[str, float]:
        """Compute normalized RRF scores for the final ranking."""
        total = len(ids)
        if total == 0:
            return {}
        return {
            chunk_id: 1.0 - (rank / total)
            for rank, chunk_id in enumerate(ids)
        }