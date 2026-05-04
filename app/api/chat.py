"""
INTERVIEW: "How do you implement streaming in FastAPI?"

We use Server-Sent Events (SSE) via the sse-starlette library.
The client connects with:
  EventSource('/api/chat/stream', {method: 'POST', body: JSON.stringify(query)})

SSE advantages over WebSockets for this use case:
1. Unidirectional: we only need server → client streaming
2. HTTP/1.1 compatible: no upgrade needed
3. Auto-reconnect: browser EventSource reconnects automatically
4. Simpler: no ws:// URL scheme, works through standard CORS

INTERVIEW: "What events does the stream emit?"
1. {event: "agent_step", data: {step: "retrieval", message: "Searching knowledge base..."}}
2. {event: "agent_step", data: {step: "research", message: "Analyzing retrieved content..."}}
3. {event: "agent_step", data: {step: "critic", message: "Evaluating quality..."}}
4. {event: "token", data: {token: "The "}} — streamed answer tokens
5. {event: "done", data: {answer, citations, confidence_score, ...}}
6. {event: "error", data: {message: "..."}}

INTERVIEW: "How do you stream LangGraph agent state to the client?"
astream_events() on LangGraph gives us an async iterator of events.
Each node completion emits an event we can forward to the client via SSE.
For token streaming, we intercept the LLM's astream() response.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sse_starlette.sse import EventSourceResponse

from app.agents.orchestrator import build_papermind_graph
from app.agents.state import PaperMindState
from app.config import get_settings
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
)
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


def get_retriever(request: Request) -> HybridRetriever:
    """FastAPI dependency — retriever is stored in app state."""
    return request.app.state.retriever


def get_reranker(request: Request) -> CrossEncoderReranker:
    """FastAPI dependency — reranker is stored in app state."""
    return request.app.state.reranker


@router.post("/query", response_model=QueryResponse)
async def query_non_streaming(
        query_req: QueryRequest,
        retriever: HybridRetriever = Depends(get_retriever),
        reranker: CrossEncoderReranker = Depends(get_reranker),
):
    """
    Non-streaming query endpoint.
    Use this for programmatic access / testing.
    Returns complete response as JSON.

    INTERVIEW: "When would you use non-streaming vs streaming?"
    Non-streaming: API clients, batch processing, testing
    Streaming: User-facing chat UI where perceived latency matters
    Streaming feels faster even if total time is the same —
    user sees tokens appearing immediately vs blank screen for 3 seconds.
    """
    settings = get_settings()
    start_time = time.time()

    graph = build_papermind_graph(retriever, reranker)

    initial_state: PaperMindState = {
        "query": query_req.query,
        "mode": query_req.mode.value,
        "session_id": query_req.session_id or "anonymous",
        "retrieval_top_k": query_req.retrieval_top_k or settings.RETRIEVAL_TOP_K,
        "rerank_top_n": query_req.rerank_top_n or settings.RERANK_TOP_N,
        "iteration_count": 0,
        "max_iterations": settings.MAX_AGENT_ITERATIONS,
        "retrieved_chunks": [],
        "final_answer": "",
        "citations": [],
        "critic_score": 0.0,
        "confidence_score": 0.0,
    }

    try:
        final_state = await graph.ainvoke(initial_state)
    except Exception as e:
        logger.error(f"Graph execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    elapsed_ms = (time.time() - start_time) * 1000

    return QueryResponse(
        session_id=initial_state["session_id"],
        query=query_req.query,
        mode=query_req.mode,
        answer=final_state.get("final_answer", ""),
        citations=final_state.get("citations", []),
        confidence_score=final_state.get("confidence_score", 0.0),
        critic_score=final_state.get("critic_score", 0.0),
        iteration_count=final_state.get("iteration_count", 0),
        processing_time_ms=elapsed_ms,
        model_used=settings.GROQ_MODEL_NAME,
    )


@router.api_route("/stream", methods=["GET", "POST"])
async def query_streaming(
        request: Request,
        # keep dependencies the same
        retriever: HybridRetriever = Depends(get_retriever),
        reranker: CrossEncoderReranker = Depends(get_reranker),
):
    """
    SSE streaming query endpoint.
    Returns an EventSourceResponse that streams agent progress + tokens.

    INTERVIEW: "How does the client know when streaming is done?"
    We emit a 'done' event with the complete response payload.
    The client can close the EventSource connection after receiving 'done'.

    INTERVIEW: "What happens if the client disconnects?"
    sse-starlette detects the disconnect via the request.is_disconnected() check.
    The async generator yields events — when the client disconnects,
    the generator is garbage collected and the LangGraph task is cancelled.
    """
    settings = get_settings()

    # Log method for easier debugging in server logs
    logger.info(f"/api/chat/stream called with method={request.method} origin={request.headers.get('origin')}")

    async def event_generator() -> AsyncIterator[dict]:
        start_time = time.time()
        graph = build_papermind_graph(retriever, reranker)

        # Support both GET (EventSource) and POST (fetch with JSON body).
        if request.method == "GET":
            # Support two GET styles:
            # 1) ?query=...&mode=... (EventSource typical)
            # 2) ?payload=<json-encoded-body> (some frontends encode the whole body)
            payload = request.query_params.get("payload")
            if payload:
                try:
                    body = json.loads(payload)
                except Exception:
                    # malformed payload — emit error and stop
                    yield {
                        "event": "error",
                        "data": json.dumps({"message": "Malformed payload query param (invalid JSON)"}),
                    }
                    return
                q = body.get("query")
                mode = body.get("mode", "docs")
                session_id = body.get("session_id")
                retrieval_top_k = body.get("retrieval_top_k")
                rerank_top_n = body.get("rerank_top_n")
            else:
                # Parse individual query params
                q = request.query_params.get("query")
                mode = request.query_params.get("mode", "docs")
                session_id = request.query_params.get("session_id")
                retrieval_top_k = request.query_params.get("retrieval_top_k")
                rerank_top_n = request.query_params.get("rerank_top_n")
        else:
            # POST — parse JSON body (fast path)
            body = await request.json()
            q = body.get("query")
            mode = body.get("mode", "docs")
            session_id = body.get("session_id")
            retrieval_top_k = body.get("retrieval_top_k")
            rerank_top_n = body.get("rerank_top_n")

        # Validate essential inputs early and return a clear error SSE event if missing
        if not q:
            yield {
                "event": "error",
                "data": json.dumps({"message": "Missing 'query' parameter in request"}),
            }
            return

        initial_state: PaperMindState = {
            "query": q,
            "mode": mode,
            "session_id": session_id or "anonymous",
            "retrieval_top_k": int(retrieval_top_k) if retrieval_top_k else settings.RETRIEVAL_TOP_K,
            "rerank_top_n": int(rerank_top_n) if rerank_top_n else settings.RERANK_TOP_N,
            "iteration_count": 0,
            "max_iterations": settings.MAX_AGENT_ITERATIONS,
            "retrieved_chunks": [],
            "final_answer": "",
            "citations": [],
            "critic_score": 0.0,
            "confidence_score": 0.0,
        }

        # Track which agents have been announced to avoid duplicate events
        announced_steps: set[str] = set()

        # Capture final state from streaming events instead of re-invoking
        # This prevents the double-execution bug that was doubling all API calls
        final_state: dict = {}

        try:
            # astream_events streams events as they happen in the graph
            # INTERVIEW: "What events does LangGraph stream?"
            # - on_chain_start / on_chain_end: node lifecycle
            # - on_llm_start / on_llm_stream / on_llm_end: LLM token streaming
            # - on_tool_start / on_tool_end: tool calls (if any)

            async for event in graph.astream_events(initial_state, version="v1"):
                event_name = event.get("event", "")
                event_data = event.get("data", {})
                node_name = event.get("name", "")

                # ── Agent Step Events ────────────────────────────────────
                if event_name == "on_chain_start" and node_name in {
                    "retrieval", "researcher", "critic", "synthesizer", "supervisor"
                }:
                    if node_name not in announced_steps:
                        announced_steps.add(node_name)
                        step_messages = {
                            "retrieval": "🔍 Searching knowledge base...",
                            "researcher": "🧠 Analyzing retrieved content...",
                            "critic": "⚖️ Evaluating answer quality...",
                            "synthesizer": "✍️ Composing answer...",
                            "supervisor": "✅ Finalizing response...",
                        }
                        yield {
                            "event": "agent_step",
                            "data": json.dumps({
                                "step": node_name,
                                "message": step_messages.get(node_name, f"Running {node_name}..."),
                                "iteration": initial_state.get("iteration_count", 0),
                            }),
                        }

                # ── Capture final state from node outputs ────────────────
                # Track on_chain_end events to accumulate state from each node
                # This replaces the second graph.ainvoke() call that was
                # doubling all API calls
                elif event_name == "on_chain_end" and node_name in {
                    "retrieval", "researcher", "critic", "synthesizer", "supervisor"
                }:
                    output = event_data.get("output")
                    if isinstance(output, dict):
                        final_state.update(output)

                # ── LLM Token Streaming ──────────────────────────────────
                # INTERVIEW: "How do you stream individual tokens from the LLM?"
                # LangChain's streaming=True + astream_events gives on_llm_stream events.
                # Each event contains a chunk of text (usually 1-5 tokens).
                elif event_name == "on_llm_stream" and "synthesizer" in (node_name or ""):
                    # event_data may be None, a dict, or an object; chunk may be in different shapes
                    try:
                        evdata = event.get("data") if isinstance(event, dict) else None
                    except Exception:
                        evdata = None

                    # normalize chunk extraction
                    chunk = None
                    if isinstance(evdata, dict):
                        chunk = evdata.get("chunk")
                    else:
                        # try attribute access
                        chunk = getattr(evdata, "chunk", None) if evdata is not None else None

                    token_text = None
                    if isinstance(chunk, dict):
                        token_text = chunk.get("content") or chunk.get("text") or chunk.get("token")
                    else:
                        token_text = getattr(chunk, "content", None) if chunk is not None else None
                        if not token_text:
                            token_text = getattr(chunk, "text", None) if chunk is not None else None

                    if token_text:
                        yield {
                            "event": "token",
                            "data": json.dumps({"token": token_text}),
                        }

                # ── Retry Notification ───────────────────────────────────
                elif event_name == "on_chain_start" and node_name == "retrieval":
                    iteration = initial_state.get("iteration_count", 0)
                    if iteration > 0:
                        yield {
                            "event": "agent_step",
                            "data": json.dumps({
                                "step": "retrieval",
                                "message": f"🔄 Retry #{iteration}: Refining search...",
                                "iteration": iteration,
                            }),
                        }

            # ── Emit Final State ─────────────────────────────────────────
            # Final state was captured from on_chain_end events above —
            # no need for a second graph.ainvoke() call
            elapsed_ms = (time.time() - start_time) * 1000

            # Emit done event
            yield {
                "event": "done",
                "data": json.dumps({
                    "answer": final_state.get("final_answer", ""),
                    "citations": final_state.get("citations", []),
                    "confidence_score": final_state.get("confidence_score", 0.0),
                    "critic_score": final_state.get("critic_score", 0.0),
                    "processing_time_ms": elapsed_ms,
                    "model_used": settings.GROQ_MODEL_NAME,
                    "iteration_count": final_state.get("iteration_count", 0),
                }),
            }

        except asyncio.CancelledError:
            logger.info("SSE stream cancelled by client")
            return
        except Exception:
            # Log full stack trace for easier debugging
            logger.exception("Stream error")
            yield {
                "event": "error",
                "data": json.dumps({"message": "Internal server error during stream — check server logs"}),
            }

    return EventSourceResponse(event_generator())


@router.options("/stream")
async def stream_options(request: Request):
    """Explicit preflight handler to avoid FastAPI trying to validate or run dependencies on OPTIONS.

    Returns 200 so CORS preflight succeeds. The global CORSMiddleware will attach the correct
    Access-Control-Allow-* headers to the response.
    """
    origin = request.headers.get("origin", "*")
    acrh = request.headers.get("access-control-request-headers", "*")
    logger.info(f"Handling preflight OPTIONS for /api/chat/stream from origin={origin} headers={acrh}")
    return Response(status_code=200, headers={
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        "Access-Control-Allow-Headers": acrh,
        "Access-Control-Allow-Credentials": "true",
    })
