"""
INTERVIEW: "Why define Pydantic schemas separately from the DB models?"
1. Request/response schemas can differ from internal models (don't expose DB IDs)
2. FastAPI auto-generates OpenAPI docs from these — free API documentation
3. Pydantic validates input before it touches your business logic — fail fast

INTERVIEW: "What's the difference between BaseModel and TypedDict here?"
BaseModel: request/response validation with serialization
TypedDict: agent state (no validation needed — internal use only)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl


# ── Enums ───────────────────────────────────────────────────────────────────

class Mode(str, Enum):
    RESEARCH = "research"
    DOCS = "docs"


class IngestionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentStep(str, Enum):
    RETRIEVAL = "retrieval"
    RESEARCH = "research"
    CRITIC = "critic"
    SYNTHESIS = "synthesis"
    SUPERVISOR = "supervisor"
    COMPLETE = "complete"
    ERROR = "error"


# ── Citation Models ──────────────────────────────────────────────────────────

class Citation(BaseModel):
    """
    INTERVIEW: "What metadata do you store per chunk?"
    Papers: doc_id, title, authors, year, section, page_number, chunk_index
    Docs: doc_id, url, page_title, section_header, chunk_index
    This metadata is stored in ChromaDB alongside the vector and used for citation generation.
    """
    doc_id: str
    title: str
    source_url: Optional[str] = None
    page_number: Optional[int] = None
    section: Optional[str] = None
    chunk_index: int
    relevance_score: float = Field(ge=0.0, le=1.0)
    snippet: str = Field(max_length=300)


# ── Chat / Query Models ──────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(
        min_length=3,
        max_length=1000,
        description="The question to answer from ingested documents",
    )
    mode: Mode = Field(description="'research' for papers, 'docs' for dev documentation")
    session_id: Optional[str] = Field(
        default_factory=lambda: str(uuid4()),
        description="Client session ID for conversation continuity",
    )
    stream: bool = Field(default=True, description="Stream response via SSE")
    retrieval_top_k: Optional[int] = Field(default=None, ge=5, le=50)
    rerank_top_n: Optional[int] = Field(default=None, ge=1, le=10)


class QueryResponse(BaseModel):
    session_id: str
    query: str
    mode: Mode
    answer: str
    citations: list[Citation]
    confidence_score: float = Field(ge=0.0, le=1.0)
    critic_score: float = Field(ge=0.0, le=1.0)
    iteration_count: int
    processing_time_ms: float
    model_used: str


# ── SSE Event Models ─────────────────────────────────────────────────────────

class SSEEvent(BaseModel):
    """
    INTERVIEW: "What does your SSE event structure look like?"
    Each event has a type so the frontend knows how to render it.
    'agent_step': show which agent is thinking (UX transparency)
    'token': stream tokens into the answer box
    'citation': append a citation card
    'done': finalize and show metrics
    'error': show error toast
    """
    event: str  # agent_step | token | citation | done | error
    data: Any
    session_id: str


class AgentStepEvent(BaseModel):
    step: AgentStep
    message: str
    iteration: Optional[int] = None


class DoneEvent(BaseModel):
    answer: str
    citations: list[Citation]
    confidence_score: float
    critic_score: float
    processing_time_ms: float
    model_used: str


# ── Ingestion Models ─────────────────────────────────────────────────────────

class PDFIngestionRequest(BaseModel):
    """For ingesting a PDF already saved to disk (e.g. via upload endpoint)"""
    file_path: str
    doc_id: Optional[str] = Field(default_factory=lambda: str(uuid4()))
    title: Optional[str] = None
    authors: Optional[list[str]] = None
    year: Optional[int] = None
    arxiv_id: Optional[str] = None


class DocsIngestionRequest(BaseModel):
    """
    INTERVIEW: "How do you ingest documentation sites?"
    We use the sitemap.xml if available (structured URLs) then fall back to crawling.
    We scope crawling by base_url to avoid scraping the entire internet.
    max_pages prevents runaway scraping of huge docs sites.
    """
    base_url: str = Field(description="Root URL of docs site, e.g. https://fastapi.tiangolo.com")
    doc_id: Optional[str] = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(description="Human-readable name, e.g. 'FastAPI Docs'")
    max_pages: int = Field(default=50, ge=1, le=200)
    use_sitemap: bool = Field(default=True)
    allowed_path_prefix: Optional[str] = Field(
        default=None,
        description="Only crawl URLs containing this prefix, e.g. '/docs/'",
    )


class ArxivIngestionRequest(BaseModel):
    arxiv_id: str = Field(description="ArXiv ID, e.g. '1706.03762' for Attention paper")
    doc_id: Optional[str] = Field(default_factory=lambda: str(uuid4()))


class IngestionResponse(BaseModel):
    doc_id: str
    status: IngestionStatus
    message: str
    chunks_created: int = 0
    processing_time_ms: float = 0.0
    title: Optional[str] = None


class IngestionStatusResponse(BaseModel):
    doc_id: str
    status: IngestionStatus
    progress_pct: float = Field(ge=0.0, le=100.0)
    message: str


# ── Retrieval Models (internal, but useful for API transparency) ──────────────

class RetrievedChunk(BaseModel):
    """
    INTERVIEW: "What does a retrieved chunk look like in your pipeline?"
    The chunk carries both content and full metadata. After reranking,
    it also has a final_score that combines dense + sparse + reranker scores.
    """
    chunk_id: str
    doc_id: str
    content: str
    metadata: dict[str, Any]
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0  # Reciprocal Rank Fusion
    reranker_score: float = 0.0
    final_score: float = 0.0


# ── Evaluation Models ────────────────────────────────────────────────────────

class EvalRequest(BaseModel):
    """
    INTERVIEW: "How do you evaluate a RAG system?"
    We use RAGAS — 4 metrics:
    1. faithfulness: Is the answer grounded in the context? (hallucination check)
    2. answer_relevancy: Does the answer address the question?
    3. context_recall: Does the retrieved context contain the answer? (retrieval quality)
    4. context_precision: Is every retrieved chunk relevant? (precision vs recall tradeoff)
    """
    mode: Mode
    num_samples: int = Field(default=10, ge=3, le=50)
    generate_test_set: bool = Field(default=True)


class EvalMetrics(BaseModel):
    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevancy: float = Field(ge=0.0, le=1.0)
    context_recall: float = Field(ge=0.0, le=1.0)
    context_precision: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)


class EvalResponse(BaseModel):
    mode: Mode
    metrics: EvalMetrics
    num_samples: int
    evaluation_time_ms: float
    model_used: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Health Check ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    chroma_research_docs: int
    chroma_docs_docs: int
    embedding_model: str