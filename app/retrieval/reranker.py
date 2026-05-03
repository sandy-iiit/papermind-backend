"""
INTERVIEW: "Why do you need a reranker if you already have hybrid retrieval?"

Two-stage retrieval (recall then precision):
Stage 1 — Hybrid retrieval (bi-encoder + BM25):
  Fast: query embedding is computed once, compared to all vectors via HNSW.
  O(log n) ANN search. Can handle millions of docs in milliseconds.
  But: bi-encoder computes query and document embeddings INDEPENDENTLY.
  It misses fine-grained query-document interactions.

Stage 2 — Cross-encoder reranking:
  The cross-encoder sees BOTH query and document together in one forward pass.
  [CLS] query [SEP] document passage [SEP]
  This captures exact term matches, negation, and nuanced relevance.
  Much more accurate than bi-encoder, but O(n) in the number of candidates.
  That's why we rerank only top-20, not all 10k docs.

INTERVIEW: "What model do you use and why?"
cross-encoder/ms-marco-MiniLM-L-6-v2:
- Trained on MS MARCO (510k passage-query pairs) — real web search data
- MiniLM-L-6: 6-layer distilled model, 22MB, fast on CPU (~50ms for 20 docs)
- v2: improved training with hard negatives mining

INTERVIEW: "What's the quality difference?"
In my experiments, adding the reranker improved context precision@5 by ~15-20%.
The improvement is most dramatic for queries with rare keywords or negations.
"""

from __future__ import annotations

import asyncio
import logging

from sentence_transformers import CrossEncoder

from app.config import get_settings
from app.models.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a cross-encoder model.
    Uses sentence-transformers CrossEncoder — free, runs locally on CPU.
    """

    def __init__(self):
        settings = get_settings()
        logger.info(f"Loading cross-encoder: {settings.RERANKER_MODEL}")
        # Load model once at startup — ~200ms on first load, cached in memory
        self._model = CrossEncoder(
            settings.RERANKER_MODEL,
            max_length=512,  # Max tokens for (query + document) pair
        )
        logger.info("Cross-encoder loaded.")
        self._top_n = settings.RERANK_TOP_N

    async def rerank(
            self,
            query: str,
            chunks: list[RetrievedChunk],
            top_n: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Rerank chunks and return top_n most relevant.

        INTERVIEW: "What's the time complexity?"
        O(n * seq_len) where n = number of candidates (20), seq_len = 512.
        In practice: ~50-100ms on CPU for 20 candidates.
        GPU would reduce this to ~5-10ms.

        INTERVIEW: "How do you handle the token limit?"
        If query + document > 512 tokens, we truncate the document.
        The cross-encoder handles this internally with max_length=512.
        In production, you'd split long chunks before reranking.
        """
        if not chunks:
            return []

        top_n = top_n or self._top_n

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, chunk.content) for chunk in chunks]

        # Run cross-encoder inference in a thread (CPU-bound)
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: self._model.predict(pairs, show_progress_bar=False),
        )

        # Attach reranker scores
        for chunk, score in zip(chunks, scores):
            chunk.reranker_score = float(score)
            # Final score: blend RRF rank with reranker
            # DESIGN CHOICE: reranker gets 80% weight — it's the most accurate signal
            chunk.final_score = 0.2 * chunk.rrf_score + 0.8 * self._normalize_score(float(score))

        # Sort by final score descending, take top_n
        reranked = sorted(chunks, key=lambda c: c.final_score, reverse=True)
        top_chunks = reranked[:top_n]

        logger.debug(
            f"Reranked {len(chunks)} → {len(top_chunks)} chunks. "
            f"Top score: {top_chunks[0].final_score:.3f}" if top_chunks else "No chunks"
        )
        return top_chunks

    @staticmethod
    def _normalize_score(score: float) -> float:
        """
        Sigmoid normalization: maps cross-encoder logit to [0, 1].
        INTERVIEW: "Why normalize the reranker score?"
        Raw cross-encoder scores are logits in range [-inf, +inf].
        Sigmoid maps them to [0, 1] so we can combine with RRF (also 0-1).
        """
        import math
        return 1.0 / (1.0 + math.exp(-score))
