"""
INTERVIEW: "How do you define state in LangGraph?"
LangGraph uses TypedDict as the state schema.
Each node receives the full state, processes it, and returns a dict
of partial updates — LangGraph merges the update into the current state.

INTERVIEW: "What's in your agent state?"
We track:
1. Input: query, mode, session_id
2. Retrieval: the retrieved chunks (input to agents)
3. Agent outputs: researcher analysis, critic score/feedback
4. Iteration control: prevents infinite retry loops
5. Final output: answer, citations, confidence
6. Error: allows graceful error propagation through the graph

DESIGN CHOICE: Flat TypedDict over nested dataclass.
LangGraph's merge is simpler with flat structures.
Nested objects require custom reducers.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict


class PaperMindState(TypedDict, total=False):
    """
    Shared state flowing through all LangGraph agent nodes.
    `total=False` makes all keys optional — nodes only return their updates.
    """
    # ── Input ──────────────────────────────────────────────────────────────
    query: str
    mode: str  # "research" | "docs"
    session_id: str
    retrieval_top_k: int
    rerank_top_n: int

    # ── Retrieval ──────────────────────────────────────────────────────────
    retrieved_chunks: list[dict[str, Any]]  # Serialized RetrievedChunk dicts
    formatted_context: str  # Context string for LLM prompt

    # ── Researcher Agent Output ────────────────────────────────────────────
    research_analysis: str  # Structured analysis of retrieved chunks
    key_points: list[str]  # Bullet points of key findings
    identified_gaps: list[str]  # What's missing in the retrieved context
    query_refined: str  # Researcher may refine query for retry

    # ── Critic Agent Output ────────────────────────────────────────────────
    critic_score: float  # 0.0 - 1.0 quality score
    critic_feedback: str  # Textual feedback explaining the score
    should_retry: bool  # Critic's recommendation

    # ── Iteration Control ──────────────────────────────────────────────────
    iteration_count: int  # Incremented on each retrieval cycle
    max_iterations: int  # Hard cap from settings

    # ── Synthesizer Agent Output ───────────────────────────────────────────
    final_answer: str  # The formatted answer to return
    citations: list[dict[str, Any]]  # Serialized Citation objects
    answer_format: str  # "markdown" | "plain"

    # ── Supervisor Agent Output ────────────────────────────────────────────
    confidence_score: float  # 0.0 - 1.0 overall confidence
    validated: bool  # Passed supervisor checks
    post_processing_notes: str  # Any modifications made by supervisor

    # ── Error Handling ─────────────────────────────────────────────────────
    error: Optional[str]
    error_node: Optional[str]  # Which node raised the error