"""
INTERVIEW: "How does the synthesizer generate the final answer?"

The synthesizer receives:
1. The original query
2. Retrieved chunks (with full metadata for citations)
3. The researcher's analysis (structured key points)
4. The critic's score (used in the prompt to calibrate hedging)

It generates a formatted answer with:
1. Direct answer to the query
2. Supporting details with [Source N] inline citations
3. Code examples preserved (especially important for docs mode)
4. A "Further Reading" section with full citation details

INTERVIEW: "How do you handle citations?"
Each retrieved chunk has a chunk_id and source metadata.
The synthesizer references [Source N] inline, where N maps to the chunk.
After synthesis, we parse [Source N] markers and build Citation objects
from the corresponding chunk metadata. This gives clickable links in the UI.

INTERVIEW: "What's different between research mode and docs mode synthesis?"
Research mode: more emphasis on "according to [paper title] (year)"
Docs mode: more emphasis on code examples and direct page links
The system prompt adapts based on the mode.
"""

from __future__ import annotations

import json
import logging
import re
from textwrap import dedent
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.state import PaperMindState
from app.config import get_llm_streaming, get_settings

logger = logging.getLogger(__name__)


def _get_synthesis_prompt(mode: str, lightweight: bool = False) -> str:
    """
    Mode-specific synthesis prompt.
    INTERVIEW: "Why different prompts for research vs docs?"
    Users in research mode want academic-style answers with paper citations.
    Users in docs mode want concise, actionable answers with code snippets.
    Same pipeline, different output formatting — single responsibility.

    When lightweight=True, the prompt also includes instructions for analysis
    that would normally come from the researcher agent, since the synthesizer
    is the only LLM call in the pipeline.
    """
    base = dedent("""
    You are PaperMind, an expert knowledge synthesizer.
    Generate a comprehensive, accurate answer to the user's query based on the provided context.

    STRICT RULES:
    1. ONLY use information from the provided context. Do NOT add outside knowledge.
    2. Cite sources inline using [Source N] format where N is the source number.
    3. If the context doesn't fully answer the query, explicitly state what's missing.
    4. Format your response in Markdown.
    5. End with a "## Sources" section listing all cited sources.
    """).strip()

    if lightweight:
        # In lightweight mode, the synthesizer also performs the researcher's analysis
        base += dedent("""

    ANALYSIS INSTRUCTIONS (since you are doing analysis + synthesis in one step):
    - First internally assess how well the context answers the query
    - Identify the most relevant key points from the context
    - Note any gaps or contradictions in the information
    - Then synthesize a well-structured answer based on your analysis
    """).strip()

    if mode == "research":
        return base + dedent("""

        RESEARCH MODE ADDITIONAL RULES:
        - Use academic tone: "According to [Paper Title] ([Source N])..."
        - Highlight consensus vs. disagreement between papers
        - Mention key findings and experimental results where present
        - Note if findings are from a specific section (e.g., "The ablation study in [Source 2] shows...")
        """).strip()
    else:  # docs
        # Strong, very explicit brevity constraints for docs mode so answers stay focused
        return base + dedent("""
+
+        DOCS MODE ADDITIONAL RULES (CONCISE):
+        - PRIORITIZE BREVITY: Start with a short TL;DR (1-2 sentences) that directly answers the query.
+        - STRUCTURE: After the TL;DR, provide up to 3 concise bullet steps or a single short code snippet if applicable.
+        - LENGTH LIMIT: The entire answer (excluding the ## Sources section) must be <= 150 words.
+        - Preserve code examples exactly as they appear in the context. If no code is present, do not invent code.
+        - When multiple approaches exist, recommend one (single line) and optionally list up to 2 alternatives as bullets.
+        - NO LONG INTRODUCTIONS: Do not include greetings, background sections, or elongated explanations.
+        - Use plain practical tone: "To achieve X, do A; then B." Keep sentences short.
+        """).strip()


async def synthesizer_node(state: PaperMindState) -> dict:
    """
    LangGraph node: Synthesizer Agent.

    INTERVIEW: "How do you prevent the synthesizer from adding information not in context?"
    1. Explicit instruction: "ONLY use information from the provided context"
    2. Low temperature (0.1): reduces creative deviation
    3. Supervisor agent validates citations afterward — catches hallucinations

    In LIGHTWEIGHT mode, this is the ONLY LLM call in the pipeline.
    The prompt is enhanced to also cover analysis that the researcher agent
    would normally perform. This keeps API calls to exactly 1 per question.
    """
    logger.info(f"Synthesizer agent | session={state.get('session_id')}")

    settings = get_settings()
    query = state["query"]
    mode = state.get("mode", "research")
    retrieved_chunks = state.get("retrieved_chunks", [])
    research_analysis = state.get("research_analysis", "")
    key_points = state.get("key_points", [])
    critic_score = state.get("critic_score", 0.7)
    lightweight = settings.LIGHTWEIGHT_PIPELINE

    if not retrieved_chunks:
        return {
            "final_answer": "I don't have enough information in the knowledge base to answer this question. Please ingest relevant documents first.",
            "citations": [],
            "answer_format": "markdown",
        }

    # Format chunks with numbered sources
    context_str, source_metadata = _format_context_with_sources(retrieved_chunks, mode)

    hedging_instruction = ""
    # In lightweight mode, critic_score won't be set (stays at default 0.7),
    # so hedging only applies in full pipeline mode
    if not lightweight and critic_score < 0.7:
        hedging_instruction = (
            f"\nNOTE: The retrieved context has a quality score of {critic_score:.2f} (below optimal). "
            "Be explicit about any uncertainties and recommend the user verify key claims."
        )

    # For docs mode encourage the concise answer style explicitly in the user message
    if mode == "docs":
        user_instruction = "AnswerStyle: concise; start with TL;DR (1-2 sentences), then up to 3 bullets or a short code example. Max 150 words."
    else:
        user_instruction = ""

    # Build user message — in lightweight mode, skip researcher's key points section
    if lightweight or not key_points:
        key_points_section = ""
    else:
        key_points_section = f"""
    RESEARCHER'S KEY POINTS:
    {chr(10).join(f"• {p}" for p in key_points)}
    """

    user_message = dedent(f"""
    QUERY: {query}
    {key_points_section}
    RETRIEVED CONTEXT:
    {context_str}
    {hedging_instruction}

    {user_instruction}

    Generate a Markdown answer with inline [Source N] citations. End with a ## Sources section.
    """).strip()

    # Use streaming LLM for token-by-token SSE output
    llm = get_llm_streaming()
    messages = [
        SystemMessage(content=_get_synthesis_prompt(mode, lightweight=lightweight)),
        HumanMessage(content=user_message),
    ]

    try:
        response = await llm.ainvoke(messages)
        answer = response.content.strip()

        # In lightweight mode, we do NOT make a second LLM call for brevity rewrite.
        # The prompt already enforces the 150-word limit for docs mode.
        # In full pipeline mode, we also skip the rewrite to save API calls —
        # the prompt is strong enough to produce concise answers.

        # Extract citations from [Source N] markers in the answer
        citations = _extract_citations(answer, source_metadata, retrieved_chunks)

        logger.info(
            f"Synthesizer complete | "
            f"answer_length={len(answer)} | "
            f"citations={len(citations)} | "
            f"lightweight={lightweight}"
        )

        return {
            "final_answer": answer,
            "citations": [c for c in citations],
            "answer_format": "markdown",
        }

    except Exception as e:
        logger.error(f"Synthesizer agent error: {e}")
        return {
            "final_answer": f"Answer synthesis failed: {str(e)}. Please try again.",
            "citations": [],
            "error": str(e),
            "error_node": "synthesizer",
        }


def _format_context_with_sources(
         chunks: list[dict], mode: str
 ) -> tuple[str, dict[int, dict]]:
    """
    Format chunks as numbered sources.
    Returns (formatted_string, {source_num: chunk_metadata})
    """
    lines: list[str] = []
    source_metadata: dict[int, dict] = {}

    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        content = chunk.get("content", "") or ""

        # In docs mode, provide a short snippet per source to keep the prompt compact
        if mode == "docs":
            snippet = content.strip().replace("\n", " ")[:400]  # first ~400 chars
        else:
            snippet = content

        source_metadata[i] = {**meta, "content": content}

        if mode == "research":
            source_header = (
                f"[Source {i}] {meta.get('title', 'Unknown Paper')} "
                f"| Section: {meta.get('section', 'N/A')} "
                f"| Page: {meta.get('page_number', 'N/A')}"
            )
        else:
            source_header = (
                f"[Source {i}] {meta.get('collection_name', 'Docs')} "
                f"| {meta.get('section_heading', meta.get('page_title', 'N/A'))} "
                f"| URL: {meta.get('section_url', meta.get('url', 'N/A'))}"
            )

        lines.append(f"\n{source_header}")
        lines.append(snippet)

    return "\n".join(lines), source_metadata


def _extract_citations(
        answer: str,
        source_metadata: dict[int, dict],
        chunks: list[dict],
) -> list[dict[str, Any]]:
    """
    Parse [Source N] markers in the answer and build Citation objects.
    INTERVIEW: "How do you link inline citations to actual sources?"
    We regex-scan the answer for [Source N] patterns.
    N maps to the numbered source list we gave the LLM.
    We then pull metadata for that chunk to build the citation.
    """
    cited_sources: set[int] = set()
    pattern = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)

    for match in pattern.finditer(answer):
        source_num = int(match.group(1))
        if 1 <= source_num <= len(chunks):
            cited_sources.add(source_num)

    citations: list[dict[str, Any]] = []
    for source_num in sorted(cited_sources):
        meta = source_metadata.get(source_num, {})
        chunk = chunks[source_num - 1] if source_num <= len(chunks) else {}

        citation = {
            "doc_id": meta.get("doc_id", ""),
            "title": meta.get("title") or meta.get("page_title") or meta.get("collection_name", "Unknown"),
            "source_url": meta.get("section_url") or meta.get("url"),
            "page_number": meta.get("page_number"),
            "section": meta.get("section") or meta.get("section_heading"),
            "chunk_index": chunk.get("chunk_index", 0),
            "relevance_score": round(chunk.get("final_score", 0.0), 3),
            "snippet": meta.get("content", "")[:200] + "..." if len(meta.get("content", "")) > 200 else meta.get(
                "content", ""),
        }
        citations.append(citation)

    return citations
