"""
Python client examples for interacting with the PaperMind FastAPI chat endpoints.

Provides:
- run_non_streaming_query(...) -> performs POST /api/chat/query
- run_streaming_query_sse(...) -> performs GET /api/chat/stream?payload=... and parses SSE events
- fetch_history(...) -> GET /api/chat/history

Requires: httpx (already in requirements.txt). Run with Python 3.10+.

Usage examples are included in `main()` at bottom.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional

import httpx


async def run_non_streaming_query(
    base_url: str,
    query: str,
    session_id: Optional[str] = None,
    mode: str = "docs",
) -> Dict[str, Any]:
    """Send a non-streaming query to /api/chat/query and return parsed JSON response.

    base_url: e.g. 'http://localhost:8000'
    """
    url = f"{base_url.rstrip('/')}/api/chat/query"
    body = {
        "query": query,
        "mode": mode,
    }
    if session_id:
        body["session_id"] = session_id
    # Ask for non-streaming response
    body["stream"] = False

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, json=body)
        resp.raise_for_status()
        return resp.json()


async def fetch_history(base_url: str, session_id: str, limit: int = 50):
    """Fetch chat history for session_id from /api/chat/history."""
    url = f"{base_url.rstrip('/')}/api/chat/history"
    params = {"session_id": session_id, "limit": str(limit)}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


# Simple SSE parser adapted for streaming EventSource text over an httpx response.
# It yields tuples (event_type, data_str) for each event.
async def _sse_iter_text_lines(aiter):
    """Yield lines from an async iterator of text chunks, preserving line breaks."""
    buf = ""
    async for chunk in aiter:
        if not chunk:
            continue
        buf += chunk
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            yield line
    if buf:
        # leftover
        yield buf


async def run_streaming_query_sse(
    base_url: str,
    query: str,
    session_id: Optional[str] = None,
    mode: str = "docs",
):
    """Run a streaming query via SSE to /api/chat/stream and print events as they're received.

    This function uses HTTP GET with the `payload` query parameter (EventSource-friendly).
    It prints 'agent_step', 'token', 'done', and 'error' events.
    """
    payload = {"query": query, "mode": mode}
    if session_id:
        payload["session_id"] = session_id

    url = f"{base_url.rstrip('/')}/api/chat/stream"
    params = {"payload": json.dumps(payload)}

    async with httpx.AsyncClient(timeout=None) as client:
        # We set timeout=None for long-lived streaming
        async with client.stream("GET", url, params=params) as resp:
            resp.raise_for_status()
            # httpx gives access to an async iterator of bytes; use aiter_text for decoded strings
            aiter = resp.aiter_text()

            # We'll parse SSE "events". Very small parser that handles lines starting with 'event:' and 'data:'
            event_type = None
            data_lines = []

            async for raw_line in _sse_iter_text_lines(aiter):
                line = raw_line.strip()
                # ignore comments or colon-only lines
                if not line:
                    # blank line → dispatch event
                    if event_type is None and not data_lines:
                        continue
                    data_str = "\n".join(data_lines)
                    # yield or handle event
                    yield (event_type or "message", data_str)
                    # reset
                    event_type = None
                    data_lines = []
                    continue

                if line.startswith(':'):
                    # comment
                    continue
                if line.startswith('event:'):
                    event_type = line[len('event:'):].strip()
                elif line.startswith('data:'):
                    data_lines.append(line[len('data:'):].lstrip())
                else:
                    # some servers send bare data lines
                    data_lines.append(line)

            # EOF reached — dispatch any remaining event
            if data_lines:
                data_str = "\n".join(data_lines)
                yield (event_type or 'message', data_str)


# Example quick-run harness
async def main():
    base = "http://localhost:8000"

    # Ensure you have a session id (you can generate on the frontend too)
    session_id = "py-client-1"

    print("--- Non-streaming query example ---")
    res = await run_non_streaming_query(base, "What is RAG?", session_id=session_id, mode="docs")
    print("Response:", json.dumps(res, indent=2))

    print('\n--- Fetch history (after non-streaming) ---')
    h = await fetch_history(base, session_id=session_id, limit=20)
    print(json.dumps(h, indent=2))

    # Streaming example (SSE). run_streaming_query_sse yields events; print them live.
    print('\n--- Streaming (SSE) example ---')
    async for ev, data in run_streaming_query_sse(base, "Explain retrieval-augmented generation (short).", session_id=session_id, mode="docs"):
        try:
            parsed = json.loads(data)
        except Exception:
            parsed = data
        print('EVENT:', ev, parsed)
        if ev == 'done':
            # When done, you may want to fetch history (server appends it when done)
            print('\nFetching history after stream finished...')
            h2 = await fetch_history(base, session_id=session_id, limit=20)
            print(json.dumps(h2, indent=2))
            break


if __name__ == '__main__':
    asyncio.run(main())
