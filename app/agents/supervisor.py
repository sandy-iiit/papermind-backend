"""
INTERVIEW: "What does the supervisor agent do that the synthesizer doesn't?"

The supervisor performs post-processing WITHOUT another LLM call (mostly).
It's a deterministic quality gate, not a generative step.

Supervisor checks:
1. Citation validation: Are all [Source N] markers valid? Remove dangling ones.
2. Length sanity: Answer not too short (< 100 chars = likely failed) or too long (> 10k chars)
3. Confidence scoring: Combines critic_score + citation density + answer length
4. Format cleanup: Ensure proper markdown formatting

INTERVIEW: "Why a supervisor instead of just trusting the synthesizer?"
The synthesizer is LLM-based and can have off days. The supervisor is deterministic.
It catches edge cases like: synthesizer references [Source 7] but we only have 5 chunks.
This is a reliability pattern — LLM outputs validated by deterministic rules.

INTERVIEW: "How do you compute confidence score?"
confidence = 0.5 * critic_score + 0.3 * (citations / max_expected_citations) + 0.2 * length_score
This rewards: high critic score, well-cited answers, appropriate length.
"""

from __future__ import annotations

import logging
import re

from app.agents.state import PaperMindState

logger = logging.getLogger(__name__)

MAX_ANSWER_LENGTH = 8000  # Characters
MIN_ANSWER_LENGTH = 100


async def supervisor_node(state: PaperMindState) -> dict:
    """
    LangGraph node: Supervisor Agent.
    Validates and post-processes the synthesizer's output.
    Deterministic — no LLM call needed.
    """
    logger.info(f"Supervisor agent | session={state.get('session_id')}")

    answer = state.get("final_answer", "")
    citations = state.get("citations", [])
    critic_score = state.get("critic_score", 0.5)
    retrieved_chunks = state.get("retrieved_chunks", [])
    iteration_count = state.get("iteration_count", 0)
    error = state.get("error")

    # ── Check 1: Error propagation ────────────────────────────────────────
    if error or not answer:
        return {
            "confidence_score": 0.0,
            "validated": False,
            "post_processing_notes": f"Pipeline error: {error or 'empty answer'}",
            "final_answer": answer or "An error occurred. Please try again.",
        }

    notes: list[str] = []

    # ── Check 2: Length validation ────────────────────────────────────────
    if len(answer) < MIN_ANSWER_LENGTH:
        notes.append(f"Answer suspiciously short ({len(answer)} chars)")

    if len(answer) > MAX_ANSWER_LENGTH:
        answer = answer[:MAX_ANSWER_LENGTH] + "\n\n*[Answer truncated for length]*"
        notes.append("Answer truncated")

    # ── Check 3: Citation validation ──────────────────────────────────────
    cited_in_answer = set(
        int(m) for m in re.findall(r"\[Source\s+(\d+)\]", answer, re.IGNORECASE)
    )
    valid_range = set(range(1, len(retrieved_chunks) + 1))
    invalid_citations = cited_in_answer - valid_range

    if invalid_citations:
        # Remove invalid [Source N] references from the answer
        for invalid_num in invalid_citations:
            answer = re.sub(
                rf"\[Source\s+{invalid_num}\]", "[Source removed]", answer
            )
        notes.append(f"Removed invalid citations: {invalid_citations}")

    # ── Check 4: Ensure ## Sources section exists ─────────────────────────
    if citations and "## Sources" not in answer and "## Source" not in answer:
        sources_section = _build_sources_section(citations, state.get("mode", "research"))
        answer = answer + "\n\n" + sources_section
        notes.append("Added missing Sources section")

    # ── Check 5: Compute confidence score ─────────────────────────────────
    citation_density = min(len(citations) / max(len(retrieved_chunks), 1), 1.0)
    length_score = _compute_length_score(len(answer))
    retry_penalty = max(0.0, 1.0 - (iteration_count - 1) * 0.1)  # Small penalty per retry

    confidence_score = (
            0.5 * critic_score
            + 0.3 * citation_density
            + 0.15 * length_score
            + 0.05 * retry_penalty
    )
    confidence_score = max(0.0, min(1.0, confidence_score))

    logger.info(
        f"Supervisor complete | "
        f"confidence={confidence_score:.2f} | "
        f"citations={len(citations)} | "
        f"answer_len={len(answer)} | "
        f"notes={notes}"
    )

    return {
        "final_answer": answer,
        "citations": citations,
        "confidence_score": round(confidence_score, 3),
        "validated": True,
        "post_processing_notes": "; ".join(notes) if notes else "All checks passed",
    }


def _compute_length_score(char_count: int) -> float:
    """
    Score answer length. Optimal range: 500-3000 chars.
    Too short → incomplete. Too long → verbose/unfocused.
    """
    if char_count < MIN_ANSWER_LENGTH:
        return 0.2
    elif char_count < 500:
        return 0.6
    elif char_count <= 3000:
        return 1.0
    elif char_count <= 6000:
        return 0.8
    else:
        return 0.5


def _build_sources_section(citations: list[dict], mode: str) -> str:
    """Build a formatted ## Sources section for the answer."""
    lines = ["## Sources\n"]

    for i, citation in enumerate(citations, start=1):
        title = citation.get("title", "Unknown")

        if mode == "research":
            page = citation.get("page_number")
            section = citation.get("section")
            line = f"{i}. **{title}**"
            if section:
                line += f" — {section}"
            if page:
                line += f" (p. {page})"
        else:
            url = citation.get("source_url")
            section = citation.get("section")
            line = f"{i}. **{title}**"
            if section:
                line += f" — {section}"
            if url and url != "null":
                line += f" — [{url}]({url})"

        lines.append(line)

    return "\n".join(lines)