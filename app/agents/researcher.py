"""
INTERVIEW: "What does the researcher agent do?"
The researcher agent:
1. Receives retrieved chunks from the retrieval pipeline
2. Analyzes them relative to the query
3. Identifies key information, gaps, and confidence level
4. Returns structured analysis for the critic to evaluate

INTERVIEW: "Why have a researcher agent instead of directly synthesizing?"
The separation exists to create a reviewable intermediate step.
The critic can evaluate the researcher's analysis BEFORE it's synthesized into
an answer. This catches cases where retrieved context is insufficient,
contradictory, or off-topic — problems that would only become obvious after
a user gets a bad answer.

INTERVIEW: "What prompt engineering techniques do you use?"
1. Role specification: "You are an expert research analyst"
2. Structured output: Request JSON with specific keys
3. Chain of thought: "First analyze relevance, then extract points, then identify gaps"
4. Few-shot implicitly: The schema guides the model toward good outputs
"""

from __future__ import annotations

import json
import logging
from textwrap import dedent

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.state import PaperMindState
from app.config import get_llm, get_settings
from app.models.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

RESEARCHER_SYSTEM_PROMPT = dedent("""
You are an expert research analyst for PaperMind, an AI knowledge assistant.
Your task is to analyze retrieved document chunks and extract structured information.

You will be given:
1. A user query
2. Retrieved chunks from the knowledge base

Your job:
1. Assess how well the retrieved chunks answer the query
2. Extract the most relevant key points
3. Identify any gaps or contradictions in the information
4. Assign a preliminary confidence score (0.0-1.0) based on retrieval quality

Return ONLY a valid JSON object with this exact structure:
{
  "research_analysis": "A comprehensive paragraph analyzing the retrieved content relative to the query",
  "key_points": ["point 1", "point 2", "point 3"],
  "identified_gaps": ["gap 1", "gap 2"],
  "query_refined": "Optional: if the query should be rephrased for better retrieval, suggest it here",
  "preliminary_confidence": 0.85
}

Be specific. Reference actual content from the chunks, not generic statements.
If the chunks are insufficient, say so explicitly in identified_gaps.
""").strip()


async def researcher_node(state: PaperMindState) -> dict:
    """
    LangGraph node: Research Agent.

    INTERVIEW: "How do you prevent the researcher from hallucinating?"
    1. System prompt explicitly says "reference actual content from the chunks"
    2. We pass the chunks verbatim — no pre-processing that might lose info
    3. Critic agent validates the analysis in the next node
    4. Temperature is 0.1 — low temperature = less creative, more factual
    """
    logger.info(f"Researcher agent | session={state.get('session_id')} | iter={state.get('iteration_count', 0)}")

    query = state["query"]
    mode = state.get("mode", "research")
    retrieved_chunks = state.get("retrieved_chunks", [])

    if not retrieved_chunks:
        return {
            "research_analysis": "No relevant documents found for this query.",
            "key_points": [],
            "identified_gaps": [
                "No documents have been ingested for this mode, or the query did not match any content."],
            "query_refined": query,
        }

    # Format context for the LLM
    context = _format_chunks_for_research(retrieved_chunks, mode)

    user_message = dedent(f"""
    QUERY: {query}

    MODE: {mode} ({'Research papers' if mode == 'research' else 'Developer documentation'})

    RETRIEVED CONTEXT ({len(retrieved_chunks)} chunks):
    {context}

    Analyze the above context and return the JSON structure as instructed.
    """).strip()

    llm = get_llm()
    messages = [
        SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    try:
        # Invoke LLM (synchronous in async context — ChatGroq handles this)
        response = await llm.ainvoke(messages)
        content = response.content.strip()

        # Parse JSON response
        result = _parse_json_response(content)

        logger.info(
            f"Researcher complete | "
            f"confidence={result.get('preliminary_confidence', 0):.2f} | "
            f"key_points={len(result.get('key_points', []))} | "
            f"gaps={len(result.get('identified_gaps', []))}"
        )

        return {
            "research_analysis": result.get("research_analysis", ""),
            "key_points": result.get("key_points", []),
            "identified_gaps": result.get("identified_gaps", []),
            "query_refined": result.get("query_refined", query),
        }

    except Exception as e:
        logger.error(f"Researcher agent error: {e}")
        return {
            "research_analysis": f"Research analysis failed: {str(e)}",
            "key_points": [],
            "identified_gaps": ["Research agent encountered an error"],
            "error": str(e),
            "error_node": "researcher",
        }


def _format_chunks_for_research(chunks: list[dict], mode: str) -> str:
    """
    Format retrieved chunks into a context string for the LLM.
    INTERVIEW: "How do you format context for the LLM?"
    We give each chunk a numbered label and include source metadata.
    This lets the LLM reference [Source 1], [Source 2] etc. in its analysis.
    We order chunks by final_score (reranker output) — best first.
    """
    lines: list[str] = []

    for i, chunk in enumerate(chunks, start=1):
        if mode == "research":
            source_info = (
                f"Paper: {chunk.get('metadata', {}).get('title', 'Unknown')} | "
                f"Section: {chunk.get('metadata', {}).get('section', 'N/A')} | "
                f"Page: {chunk.get('metadata', {}).get('page_number', 'N/A')}"
            )
        else:
            source_info = (
                f"Docs: {chunk.get('metadata', {}).get('collection_name', 'Unknown')} | "
                f"URL: {chunk.get('metadata', {}).get('section_url', chunk.get('metadata', {}).get('url', 'N/A'))} | "
                f"Section: {chunk.get('metadata', {}).get('section_heading', 'N/A')}"
            )

        lines.append(f"[Source {i}] {source_info}")
        lines.append(chunk.get("content", ""))
        lines.append("")

    return "\n".join(lines)


def _parse_json_response(content: str) -> dict:
    """
    Safely parse JSON from LLM response.
    INTERVIEW: "How do you handle JSON parsing failures?"
    LLMs sometimes wrap JSON in markdown code blocks (```json ... ```).
    We strip those first. If parsing still fails, return a safe default.
    In production, use structured outputs / tool calling for guaranteed JSON.
    """
    import re

    # Strip markdown code blocks
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if json_match:
        content = json_match.group(1)

    # Try to find JSON object
    obj_match = re.search(r"\{[\s\S]*\}", content)
    if obj_match:
        content = obj_match.group()

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e}. Raw: {content[:200]}")
        return {
            "research_analysis": content,
            "key_points": [],
            "identified_gaps": ["Could not parse structured response"],
            "query_refined": "",
            "preliminary_confidence": 0.5,
        }