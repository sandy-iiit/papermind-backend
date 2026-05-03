"""
INTERVIEW: "How do you structure a production FastAPI application?"
Key patterns we use:
1. Lifespan context manager: startup/shutdown logic in one place
2. App state: shared objects (retriever, reranker) initialized once, injected via Depends
3. Middleware: CORS, logging, request timing
4. Router separation: chat, ingestion, evaluation are separate routers with prefixes
5. Health check: monitoring systems ping /health to know if the app is up

INTERVIEW: "Why initialize heavy models in lifespan instead of globally?"
Global initialization runs at module import time — before the event loop starts.
This blocks startup and can't be async. Lifespan runs after startup, in the
async event loop, and can use await.
Also: lifespan cleanup (shutdown) ensures graceful termination.

INTERVIEW: "How does dependency injection work in FastAPI?"
Depends() creates a dependency injection chain.
When a route function declares `retriever: HybridRetriever = Depends(get_retriever)`,
FastAPI calls get_retriever(request) → fetches from app.state → injects.
No global variables needed. Easy to mock in tests.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import structlog  # structlog is intentionally imported for structured logging in production; keep import
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

from app import __app_name__, __version__
from app.api import chat, evaluation, ingestion
from app.config import get_settings
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_store import VectorStore

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    INTERVIEW: "What's the order of initialization?"
    1. ChromaDB → must exist before retriever queries it
    2. VectorStore wrapper → wraps ChromaDB client
    3. HybridRetriever → loads sentence-transformers embedding model (~1-2s)
    4. CrossEncoderReranker → loads cross-encoder model (~0.5s)
    5. Build BM25 indexes from existing ChromaDB data (if any)

    INTERVIEW: "What happens if ChromaDB fails to initialize?"
    The exception propagates out of lifespan → FastAPI aborts startup.
    Fail-fast: better to crash at boot than fail silently on first request.
    """
    settings = get_settings()
    logger.info(f"Starting {__app_name__} v{__version__}")
    logger.info(f"Model: {settings.GROQ_MODEL_NAME} | Embedding: {settings.EMBEDDING_MODEL}")

    # ── Initialize ChromaDB ────────────────────────────────────────────────
    vector_store = VectorStore()
    vector_store.initialize()
    app.state.vector_store = vector_store

    # ── Initialize Retriever (loads embedding model) ───────────────────────
    retriever = HybridRetriever(vector_store=vector_store)
    app.state.retriever = retriever

    # ── Initialize Reranker (loads cross-encoder) ──────────────────────────
    reranker = CrossEncoderReranker()
    app.state.reranker = reranker

    # ── Build BM25 indexes from existing data ──────────────────────────────
    for mode in ("research", "docs"):
        if vector_store.count(mode) > 0:
            logger.info(f"Building BM25 index for {mode}...")
            await retriever.build_bm25_index(mode)

    logger.info(f"{__app_name__} startup complete — ready to serve requests")

    yield  # ← App runs here

    # ── Shutdown ───────────────────────────────────────────────────────────
    logger.info(f"{__app_name__} shutting down...")
    # ChromaDB PersistentClient flushes to disk on garbage collection
    # No explicit close needed for embedded mode


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=__app_name__,
        description=(
            "PaperMind — AI-powered research paper and documentation Q&A. "
            "Features: hybrid retrieval (BM25 + dense), cross-encoder reranking, "
            "multi-agent LangGraph pipeline, RAGAS evaluation dashboard."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",  # Swagger UI
        redoc_url="/redoc",  # ReDoc
        openapi_url="/openapi.json",
    )

    # Early preflight middleware to catch OPTIONS requests for SSE and avoid 400s
    class PreflightMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            # Intercept OPTIONS preflight to /api/chat/stream and respond immediately
            if request.method == "OPTIONS" and request.url.path == "/api/chat/stream":
                origin = request.headers.get("origin", "*")
                acrh = request.headers.get("access-control-request-headers", "*")
                logger.info(f"PreflightMiddleware responding to OPTIONS {request.url.path} origin={origin} headers={acrh}")
                headers = {
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
                    "Access-Control-Allow-Headers": acrh,
                    "Access-Control-Allow-Credentials": "true",
                }
                return StarletteResponse(status_code=200, headers=headers)
            return await call_next(request)

    # Install preflight middleware before other middlewares/routers
    app.add_middleware(PreflightMiddleware)

    # ── CORS Middleware ────────────────────────────────────────────────────
    # INTERVIEW: "Why do you need CORS?"
    # Browser security policy blocks cross-origin requests.
    # Our React frontend (localhost:3000) calls backend (localhost:8000).
    # Different ports = different origins = CORS needed.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Processing-Time"],
    )

    # ── GZip Middleware ────────────────────────────────────────────────────
    # Compress large responses (evaluation results, long answers)
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # ── Request Timing Middleware ──────────────────────────────────────────
    @app.middleware("http")
    async def add_timing_header(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        elapsed = (time.time() - start) * 1000
        response.headers["X-Processing-Time"] = f"{elapsed:.2f}ms"
        return response

    # Add application-level OPTIONS handler for the SSE endpoint to ensure preflight is handled
    @app.options("/api/chat/stream")
    async def api_chat_stream_options(request: Request):
        logger.debug(f"App-level preflight for /api/chat/stream origin={request.headers.get('origin')}")
        return Response(status_code=200)

    # Fallback wildcard OPTIONS handler for any /api/* preflight to help development
    @app.options("/api/{path:path}")
    async def api_wildcard_options(request: Request, path: str):
        """Respond to CORS preflight requests for any /api/* path during development.

        This echoes the Origin header and requested headers so browsers get the Access-Control-Allow-* headers
        they expect. In production, prefer configuring exact origins via env and relying on CORSMiddleware.
        """
        origin = request.headers.get("origin", "*")
        acrh = request.headers.get("access-control-request-headers", "*")
        return Response(status_code=200, headers={
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
            "Access-Control-Allow-Headers": acrh,
            "Access-Control-Allow-Credentials": "true",
        })

    # ── Global Exception Handler ───────────────────────────────────────────
    # INTERVIEW: "How do you handle unexpected errors in production?"
    # Catch all unhandled exceptions, log them, return structured error response.
    # NEVER expose stack traces to clients (security risk).
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            f"Unhandled exception: {exc}",
            exc_info=True,
            extra={"path": request.url.path, "method": request.method},
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if settings.DEBUG else "An unexpected error occurred",
                "path": request.url.path,
            },
        )

    # ── Include Routers ────────────────────────────────────────────────────
    app.include_router(chat.router, prefix="/api")
    app.include_router(ingestion.router, prefix="/api")
    app.include_router(evaluation.router, prefix="/api")

    # ── Health Check ───────────────────────────────────────────────────────
    @app.get("/health", tags=["System"])
    async def health_check(request: Request):
        """
        INTERVIEW: "What should a health check return?"
        At minimum: 200 OK = app is running.
        Better: check dependent services (ChromaDB, LLM API reachability).
        We return document counts — if ChromaDB crashed, count would fail.
        Monitoring systems use this to decide if the app needs to be restarted.
        """
        vs: VectorStore = request.app.state.vector_store
        settings = get_settings()
        return {
            "status": "healthy",
            "version": __version__,
            "model": settings.GROQ_MODEL_NAME,
            "embedding_model": settings.EMBEDDING_MODEL,
            "chroma_research_docs": vs.count("research"),
            "chroma_docs_docs": vs.count("docs"),
        }

    @app.get("/", tags=["System"])
    async def root():
        return {
            "app": __app_name__,
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
        }

    return app


# Entry point for uvicorn
app = create_app()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info",
    )