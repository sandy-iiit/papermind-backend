"""
INTERVIEW: "How do you evaluate a RAG system? What metrics matter?"

RAGAS provides 4 reference-free metrics (no ground truth needed):

1. FAITHFULNESS (0-1):
   "Is the answer supported by the retrieved context?"
   Method: Break answer into claims → check each claim against context → score = claims_supported / total_claims
   Why it matters: Catches hallucinations. Low faithfulness = model invented facts not in context.

2. ANSWER RELEVANCY (0-1):
   "Does the answer actually address the question?"
   Method: Generate N reverse questions from the answer → embed them → compute similarity to original query
   Why it matters: Catches evasive or off-topic answers that are technically faithful.

3. CONTEXT RECALL (0-1):
   "Does the retrieved context contain all information needed to answer?"
   Method: Break ground truth into statements → check how many are in the context
   Requires: ground_truth labels (we generate them synthetically)
   Why it matters: Measures retrieval quality. Low recall = chunking/embedding problem.

4. CONTEXT PRECISION (0-1):
   "Are the retrieved chunks all relevant? Or is there noise?"
   Method: For each retrieved chunk, check if it contains information relevant to the ground truth
   Why it matters: High recall but low precision = retrieving too many irrelevant chunks.

INTERVIEW: "What's your overall pipeline score?"
overall_score = 0.3*faithfulness + 0.3*answer_relevancy + 0.25*context_recall + 0.15*context_precision
Weights: faithfulness and relevancy matter most for user experience.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from app.config import get_settings
from app.models.schemas import EvalMetrics

logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """
    Wraps RAGAS evaluation framework.
    DESIGN CHOICE: Lazy import of ragas to avoid slowing startup.
    ragas imports torch + transformers — heavy, ~3s cold start.
    We only load it when evaluation is actually requested.
    """

    def __init__(self):
        self._settings = get_settings()
        self._ragas_loaded = False
        self._metrics = None

    def _load_ragas(self):
        """Lazy-load RAGAS metrics."""
        if self._ragas_loaded:
            return

        try:
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
            from ragas import evaluate
            from langchain_groq import ChatGroq
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper

            # Configure RAGAS to use our free stack
            # INTERVIEW: "RAGAS needs its own LLM — doesn't that use more API credits?"
            # Yes. For production, cache RAGAS evaluations and run them async / nightly.
            # We use llama3-8b for evaluation (cheaper than 70b) — evaluation is less complex.
            eval_llm = ChatGroq(
                api_key=self._settings.GROQ_API_KEY,
                model="llama3-8b-8192",  # Smaller model for eval — cheaper
                temperature=0.0,
            )
            eval_embeddings = HuggingFaceEmbeddings(
                model_name=self._settings.EMBEDDING_MODEL
            )

            # Wrap for RAGAS compatibility
            self._ragas_llm = LangchainLLMWrapper(eval_llm)
            self._ragas_embeddings = LangchainEmbeddingsWrapper(eval_embeddings)

            self._faithfulness = faithfulness
            self._answer_relevancy = answer_relevancy
            self._context_recall = context_recall
            self._context_precision = context_precision
            self._evaluate = evaluate

            # Configure metrics with our LLM/embeddings
            for metric in [faithfulness, answer_relevancy, context_recall, context_precision]:
                metric.llm = self._ragas_llm
                if hasattr(metric, "embeddings"):
                    metric.embeddings = self._ragas_embeddings

            self._ragas_loaded = True
            logger.info("RAGAS metrics loaded successfully")

        except ImportError as e:
            logger.error(f"RAGAS not installed: {e}. Run: pip install ragas")
            raise

    async def evaluate_pipeline(
            self,
            qa_samples: list[dict[str, Any]],
    ) -> EvalMetrics:
        """
        Run RAGAS evaluation on a set of QA samples.

        Each sample must have:
        {
          "question": str,
          "answer": str,
          "contexts": List[str],  # Retrieved chunk texts
          "ground_truth": str,    # Expected answer (can be synthetic)
        }

        INTERVIEW: "Where do ground truth labels come from?"
        We generate them synthetically using the same LLM on the original documents.
        It's not perfect but gives a reasonable evaluation signal.
        See test_generator.py for the generation logic.
        """
        loop = asyncio.get_event_loop()

        # Load RAGAS in executor (heavy imports)
        await loop.run_in_executor(None, self._load_ragas)

        start = time.time()

        try:
            from datasets import Dataset

            # RAGAS expects a HuggingFace Dataset
            dataset = Dataset.from_list(qa_samples)

            result = await loop.run_in_executor(
                None,
                lambda: self._evaluate(
                    dataset,
                    metrics=[
                        self._faithfulness,
                        self._answer_relevancy,
                        self._context_recall,
                        self._context_precision,
                    ],
                ),
            )

            # Extract scores (RAGAS returns pandas DataFrame)
            scores = result.to_pandas()

            faithfulness_score = float(scores["faithfulness"].mean())
            relevancy_score = float(scores["answer_relevancy"].mean())
            recall_score = float(scores["context_recall"].mean())
            precision_score = float(scores["context_precision"].mean())

            # Weighted overall score
            overall = (
                    0.30 * faithfulness_score
                    + 0.30 * relevancy_score
                    + 0.25 * recall_score
                    + 0.15 * precision_score
            )

            elapsed = (time.time() - start) * 1000
            logger.info(
                f"RAGAS evaluation complete | "
                f"faithfulness={faithfulness_score:.3f} | "
                f"relevancy={relevancy_score:.3f} | "
                f"recall={recall_score:.3f} | "
                f"precision={precision_score:.3f} | "
                f"time={elapsed:.0f}ms"
            )

            return EvalMetrics(
                faithfulness=round(faithfulness_score, 3),
                answer_relevancy=round(relevancy_score, 3),
                context_recall=round(recall_score, 3),
                context_precision=round(precision_score, 3),
                overall_score=round(overall, 3),
            )

        except Exception as e:
            logger.error(f"RAGAS evaluation error: {e}")
            # Return neutral scores on error rather than crashing
            return EvalMetrics(
                faithfulness=0.0,
                answer_relevancy=0.0,
                context_recall=0.0,
                context_precision=0.0,
                overall_score=0.0,
            )