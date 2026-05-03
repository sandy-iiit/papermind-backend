"""
INTERVIEW: "Explain your LangGraph architecture."

The orchestrator defines the agent graph — think of it like a flowchart
where nodes are agents and edges define the flow between them.

Graph:
  START → [retrieval_node] → [researcher_node] → [critic_node]
      ┌─────────────────────────────────────────────────┘
      │  Conditional edge based on critic_score + iteration_count
      │
      ├── score >= MIN_CRITIC_SCORE → [synthesizer_node]
      ├── score < MIN_CRITIC_SCORE AND iterations < MAX → [retrieval_node] (RETRY)
      └── iterations >= MAX → [synthesizer_node] (force proceed)

  [synthesizer_node] → [supervisor_node] → END

INTERVIEW: "What are the advantages of LangGraph over a simple for loop?"
1. Visualizable: the graph structure is explicit and inspectable
2. Resumable: state can be checkpointed and resumed (for long-running tasks)
3. Parallel branches: multiple agents can run concurrently with fan-out/fan-in
4. Type-safe: TypedDict state catches schema violations at graph compile time
5. Streaming: LangGraph can stream intermediate states to the client

INTERVIEW: "How would you add a new agent?"
1. Write the agent function (takes PaperMindState, returns dict)
2. graph.add_node("new_agent", new_agent_function)
3. Add edges connecting it to the appropriate nodes
4. No other changes needed — the rest of the system is unaffected.
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, StateGraph

from app.agents.critic import critic_node
from app.agents.researcher import researcher_node
from app.agents.state import PaperMindState
from app.agents.supervisor import supervisor_node
from app.agents.synthesizer import synthesizer_node
from app.config import get_settings
from app.models.schemas import RetrievedChunk
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


def create_retrieval_node(
        retriever: HybridRetriever, reranker: CrossEncoderReranker
):
    """
    Factory function that creates the retrieval node with injected dependencies.
    INTERVIEW: "Why a factory function instead of a class method?"
    LangGraph nodes must be plain callables (state → dict).
    Dependency injection via closure keeps the node signature clean.
    """

    async def retrieval_node(state: PaperMindState) -> dict:
        """
        LangGraph node: Retrieval.
        Runs hybrid retrieval + cross-encoder reranking.
        """
        logger.info(
            f"Retrieval node | session={state.get('session_id')} | "
            f"iter={state.get('iteration_count', 0) + 1}"
        )

        # Use refined query from researcher on retry iterations
        iteration = state.get("iteration_count", 0)
        if iteration > 0 and state.get("query_refined"):
            query = state["query_refined"]
            logger.info(f"Using refined query: {query}")
        else:
            query = state["query"]

        mode = state["mode"]
        top_k = state.get("retrieval_top_k") or get_settings().RETRIEVAL_TOP_K
        rerank_top_n = state.get("rerank_top_n") or get_settings().RERANK_TOP_N

        # Run hybrid retrieval
        chunks = await retriever.retrieve(
            query=query,
            mode=mode,
            top_k=top_k,
            rerank_top_n=None,  # Get all candidates for reranker
        )

        if not chunks:
            return {
                "retrieved_chunks": [],
                "formatted_context": "",
                "iteration_count": iteration + 1,
            }

        # Rerank candidates
        reranked = await reranker.rerank(
            query=query,
            chunks=chunks,
            top_n=rerank_top_n,
        )

        # Serialize chunks to dicts (TypedDict state must be JSON-serializable)
        serialized = [_serialize_chunk(c) for c in reranked]

        logger.info(
            f"Retrieval complete | "
            f"{len(chunks)} retrieved → {len(reranked)} reranked | "
            f"top score: {reranked[0].final_score:.3f}" if reranked else ""
        )

        return {
            "retrieved_chunks": serialized,
            "iteration_count": iteration + 1,
        }

    return retrieval_node


def route_after_critic(
        state: PaperMindState,
) -> Literal["synthesizer", "retrieval"]:
    """
    Conditional edge function after the critic node.
    INTERVIEW: "How does LangGraph routing work?"
    This function reads state and returns a string key.
    The conditional edge maps that key to the next node.
    It's essentially a router/dispatcher pattern.
    """
    settings = get_settings()
    critic_score = state.get("critic_score", 0.5)
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", settings.MAX_AGENT_ITERATIONS)
    should_retry = state.get("should_retry", False)
    error = state.get("error")

    # Always proceed if there was an error (don't retry on errors)
    if error:
        logger.info(f"Routing to synthesizer (error encountered)")
        return "synthesizer"

    # Check if quality is acceptable
    if critic_score >= settings.MIN_CRITIC_SCORE:
        logger.info(f"Routing to synthesizer (critic_score={critic_score:.2f} >= {settings.MIN_CRITIC_SCORE})")
        return "synthesizer"

    # Check if we should retry
    if should_retry and iteration_count < max_iterations:
        logger.info(
            f"Routing to retrieval (retry) | "
            f"critic_score={critic_score:.2f} | "
            f"iter={iteration_count}/{max_iterations}"
        )
        return "retrieval"

    # Max iterations reached — proceed with what we have
    logger.info(
        f"Routing to synthesizer (max iterations reached | "
        f"critic_score={critic_score:.2f})"
    )
    return "synthesizer"


def build_papermind_graph(
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
) -> any:
    """
    Build and compile the PaperMind LangGraph.
    Returns a compiled graph ready to invoke.

    INTERVIEW: "What does graph.compile() do?"
    Compile validates the graph structure (checks for unreachable nodes,
    missing edges, type mismatches in state), and returns a runnable
    object with .invoke() and .astream() methods.
    """
    graph = StateGraph(PaperMindState)

    # Create retrieval node with dependencies injected
    retrieval_node = create_retrieval_node(retriever, reranker)

    # ── Add Nodes ──────────────────────────────────────────────────────────
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("critic", critic_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("supervisor", supervisor_node)

    # ── Add Edges ──────────────────────────────────────────────────────────
    # Entry point
    graph.set_entry_point("retrieval")

    # Fixed edges (always happen)
    graph.add_edge("retrieval", "researcher")
    graph.add_edge("researcher", "critic")

    # Conditional edge after critic — core of the retry loop
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "synthesizer": "synthesizer",
            "retrieval": "retrieval",  # Back-edge creates the retry loop
        },
    )

    graph.add_edge("synthesizer", "supervisor")
    graph.add_edge("supervisor", END)

    compiled = graph.compile()
    logger.info("PaperMind LangGraph compiled successfully")
    return compiled


def _serialize_chunk(chunk: RetrievedChunk) -> dict:
    """Convert RetrievedChunk to dict for state storage."""
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "content": chunk.content,
        "metadata": chunk.metadata,
        "dense_score": chunk.dense_score,
        "bm25_score": chunk.bm25_score,
        "rrf_score": chunk.rrf_score,
        "reranker_score": chunk.reranker_score,
        "final_score": chunk.final_score,
        "chunk_index": getattr(chunk, "chunk_index", 0),
    }