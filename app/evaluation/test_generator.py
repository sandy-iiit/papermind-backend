"""
INTERVIEW: "How do you create evaluation test sets without manual labeling?"

Synthetic test set generation using the same LLM:
1. Sample random chunks from the knowledge base
2. For each chunk, generate a realistic question a user might ask
3. Generate the expected answer from the chunk (ground truth)
4. Add to test set

This is called "LLM-as-judge" or "synthetic QA generation."
It's not perfect (the LLM evaluating what the LLM generated has bias),
but it's much better than no evaluation at all, and practical for a free stack.

For production: use human-curated test sets, or at minimum use a DIFFERENT
model for generation vs evaluation (e.g., generate with GPT-4, evaluate with Groq).

INTERVIEW: "What's the bias risk in synthetic evaluation?"
If you use the same LLM to generate test set AND answer questions,
the model will tend to score itself highly (self-consistency bias).
Mitigation: use different models, different temperature, or add adversarial examples.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from textwrap import dedent
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_llm
from app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

TEST_GENERATION_PROMPT = dedent("""
You are creating evaluation examples for a RAG system test set.
Given a document chunk, generate a realistic user question and expected answer.

The question should:
- Be something a real user would naturally ask
- Be answerable from the chunk content alone
- Not be trivially obvious (not just "what is in this text?")

Return ONLY valid JSON:
{
  "question": "The natural user question",
  "ground_truth": "The expected correct answer based on the chunk"
}
""").strip()


class TestSetGenerator:
    """
    Generates synthetic QA pairs for RAGAS evaluation.
    """

    def __init__(self, vector_store: VectorStore):
        self._vector_store = vector_store

    async def generate(self, mode: str, num_samples: int = 10) -> list[dict[str, Any]]:
        """
        Generate QA pairs from random chunks in the collection.

        INTERVIEW: "Why random sampling for test set generation?"
        We want to cover the knowledge base breadth, not just easy chunks.
        Random sampling gives unbiased coverage. For production, use stratified
        sampling by document and section to ensure even coverage.
        """
        ids, documents = self._vector_store.get_all_documents_with_ids(mode)

        if not ids:
            logger.warning(f"No documents in {mode} collection for test generation")
            return []

        # Sample min(num_samples, available) chunks
        sample_size = min(num_samples, len(ids))
        indices = random.sample(range(len(ids)), sample_size)

        sampled_docs = [documents[i] for i in indices]

        # Generate QA pairs concurrently (respecting Groq rate limits)
        semaphore = asyncio.Semaphore(3)  # 3 concurrent LLM calls max
        tasks = [
            self._generate_qa_pair(chunk, semaphore)
            for chunk in sampled_docs
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        qa_pairs: list[dict[str, Any]] = []
        for result in results:
            if isinstance(result, dict) and "question" in result:
                qa_pairs.append(result)
            elif isinstance(result, Exception):
                logger.warning(f"QA generation failed: {result}")

        logger.info(f"Generated {len(qa_pairs)} QA pairs for {mode}")
        return qa_pairs

    async def _generate_qa_pair(
            self, chunk_text: str, semaphore: asyncio.Semaphore
    ) -> dict[str, Any]:
        """Generate a single QA pair from a chunk."""
        async with semaphore:
            llm = get_llm()
            messages = [
                SystemMessage(content=TEST_GENERATION_PROMPT),
                HumanMessage(content=f"CHUNK:\n{chunk_text[:1500]}"),
            ]

            try:
                response = await llm.ainvoke(messages)
                content = response.content.strip()

                # Parse JSON response
                import re
                json_match = re.search(r"\{[\s\S]*\}", content)
                if json_match:
                    parsed = json.loads(json_match.group())
                    return {
                        "question": parsed.get("question", ""),
                        "ground_truth": parsed.get("ground_truth", ""),
                        "source_chunk": chunk_text[:500],
                    }
            except Exception as e:
                logger.debug(f"QA pair generation error: {e}")

            return {}