"""
INTERVIEW: "What does the critic agent do and why is it valuable?"

The critic is a quality gate between retrieval and synthesis.
It evaluates the researcher's analysis on three dimensions:
1. Completeness: Does the retrieved context fully answer the query?
2. Relevance: Are the retrieved chunks actually about what was asked?
3. Consistency: Are there contradictions between chunks?

Why this matters:
- Without a critic, a poorly retrieved context gets synthesized into a
  confident-sounding but incorrect answer (hallucination).
- The critic can recommend retry with a refined query.
- The critic score is surfaced to the user (confidence indicator in UI).

INTERVIEW: "How do you prevent the critic from always being too harsh or too lenient?"
The scoring rubric is explicit in the prompt. We also cap MAX_AGENT_ITERATIONS=3
so even a harsh critic can't loop forever. After 3 attempts, we synthesize
with whatever we have and flag low confidence to the user.
"""

from __future__ import annotations

import json
import logging
from textwrap import dedent

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.state import PaperMindState
from app.config import get_llm

logger = logging.getLogger(__name__)

CRITIC_SYSTEM_PROMPT = dedent("""
You are a critical evaluator for a RAG (Retrieval-Augmented Generation) system.
Your job is to assess whether the retrieved context and research analysis are sufficient
to answer the user's query accurately and completely.

Evaluate on these dimensions:
1. COMPLETENESS (0-1): Does the retrieved context contain enough information to fully answer?
2. RELEVANCE (0-1): How on-topic are the retrieved chunks? Are they actually about what was asked?
3. CONSISTENCY (0-1): Are the chunks consistent with each other? Contradictions score lower.

Overall score = 0.4 * completeness + 0.4 * relevance + 0.2 * consistency

Return ONLY a valid JSON object:
{
  "completeness": 0.8,
  "relevance": 0.9,
  "consistency": 0.95,
  "overall_score": 0.87,
  "feedback": "Detailed explanation of the scores",
  "should_retry": false,
  "retry_reason": "Only populate if should_retry is true — explain what additional info is needed"
}

Be honest and calibrated. A score above 0.7 means proceed to synthesis.
Below 0.7 means the system should try to retrieve more/better information.
""").strip()


async def critic_node(state: PaperMindState) -> dict:
    """
    LangGraph node: Critic Agent.

    INTERVIEW: "How does the critic integrate with LangGraph's routing?"
    This node returns critic_score and should_retry.
    The orchestrator's conditional edge reads these and routes to either:
    - 'synthesizer' (if score >= MIN_CRITIC_SCORE or max iterations reached)
    - 'retrieval' (if score < threshold and iterations remaining)
    """
    logger.info(
        f"Critic agent | session={state.get('session_id')} | "
        f"iter={state.get('iteration_count', 0)}"
    )

    query = state["query"]
    research_analysis = state.get("research_analysis", "")
    key_points = state.get("key_points", [])
    identified_gaps = state.get("identified_gaps", [])
    retrieved_chunks = state.get("retrieved_chunks", [])
    iteration_count = state.get("iteration_count", 0)

    user_message = dedent(f"""
    QUERY: {query}

    RESEARCH ANALYSIS:
    {research_analysis}

    KEY POINTS IDENTIFIED:
    {chr(10).join(f"- {p}" for p in key_points)}

    IDENTIFIED GAPS:
    {chr(10).join(f"- {g}" for g in identified_gaps)}

    NUMBER OF RETRIEVED CHUNKS: {len(retrieved_chunks)}
    CURRENT ITERATION: {iteration_count}

    Evaluate the quality and completeness of this research and return the JSON assessment.
    """).strip()

    llm = get_llm()
    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    try:
        response = await llm.ainvoke(messages)
        content = response.content.strip()
        result = _parse_critic_response(content)

        critic_score = float(result.get("overall_score", 0.5))
        should_retry = result.get("should_retry", False)

        logger.info(
            f"Critic score: {critic_score:.2f} | "
            f"should_retry={should_retry} | "
            f"completeness={result.get('completeness', 0):.2f}"
        )

        return {
            "critic_score": critic_score,
            "critic_feedback": result.get("feedback", ""),
            "should_retry": should_retry and iteration_count < state.get("max_iterations", 3),
        }

    except Exception as e:
        logger.error(f"Critic agent error: {e}")
        return {
            "critic_score": 0.5,  # Neutral score on error — proceed to synthesis
            "critic_feedback": f"Evaluation failed: {str(e)}. Proceeding with available context.",
            "should_retry": False,
            "error": str(e),
            "error_node": "critic",
        }


def _parse_critic_response(content: str) -> dict:
    """Parse JSON from critic response."""
    import re

    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if json_match:
        content = json_match.group(1)

    obj_match = re.search(r"\{[\s\S]*\}", content)
    if obj_match:
        content = obj_match.group()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Fallback: extract score from text if JSON fails
        score_match = re.search(r"(?:overall_score|score)[:\s]+([0-9.]+)", content)
        score = float(score_match.group(1)) if score_match else 0.6
        return {
            "overall_score": score,
            "feedback": content,
            "should_retry": score < 0.5,
            "completeness": score,
            "relevance": score,
            "consistency": score,
        }