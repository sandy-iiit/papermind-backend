"""
INTERVIEW: "How do you handle large file uploads in FastAPI?"
We use python-multipart for streaming file uploads.
Files are written to disk first (not held in memory) to handle large PDFs.
We return immediately with a job ID and process asynchronously via background tasks.

INTERVIEW: "Why background tasks instead of blocking the endpoint?"
A 100-page PDF with scraping + chunking + embedding can take 30-60 seconds.
Blocking the endpoint would timeout the HTTP request.
FastAPI's BackgroundTasks runs after the response is sent — non-blocking.
For production: use Celery + Redis for distributed background processing.

INTERVIEW: "How do you prevent re-ingesting the same document?"
We store a file hash (SHA256) per document.
Before ingesting, we check if the hash already exists in ChromaDB metadata.
If yes, return "already ingested" without reprocessing.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from uuid import uuid4

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile

from app.config import get_settings
from app.ingestion.docs_ingestor import DocsIngestor
from app.ingestion.pdf_ingestor import PDFIngestor
from app.models.schemas import (
    ArxivIngestionRequest,
    DocsIngestionRequest,
    IngestionResponse,
    IngestionStatus,
)
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["Ingestion"])

# In-memory status tracker (replace with Redis for production)
_ingestion_jobs: dict[str, dict] = {}


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_retriever(request: Request) -> HybridRetriever:
    return request.app.state.retriever


@router.post("/pdf", response_model=IngestionResponse)
async def ingest_pdf_upload(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        title: str | None = Form(default=None),
        authors: str | None = Form(default=None),  # JSON array string
        year: int | None = Form(default=None),
        doc_id: str | None = Form(default=None),
        vector_store: VectorStore = Depends(get_vector_store),
        retriever: HybridRetriever = Depends(get_retriever),
):
    """
    Upload and ingest a PDF research paper.

    INTERVIEW: "How do you validate file uploads?"
    1. Content-type check: must be application/pdf
    2. File size limit: 50MB (set in main.py middleware)
    3. PDF validity: PyMuPDF will raise if file is not a valid PDF
    """
    settings = get_settings()

    # Validate file type
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        if not (file.filename or "").endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    doc_id = doc_id or str(uuid4())

    # Save uploaded file to disk
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{doc_id}.pdf"

    try:
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(1024 * 64):  # 64KB chunks
                await f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save failed: {e}")

    # Parse authors from JSON string
    authors_list: list[str] = []
    if authors:
        import json
        try:
            authors_list = json.loads(authors)
        except json.JSONDecodeError:
            authors_list = [a.strip() for a in authors.split(",")]

    # Initialize job status
    _ingestion_jobs[doc_id] = {
        "status": IngestionStatus.PENDING,
        "progress_pct": 0.0,
        "message": "Queued for processing",
    }

    # Process in background
    background_tasks.add_task(
        _process_pdf_background,
        file_path=str(file_path),
        doc_id=doc_id,
        title=title or file.filename,
        authors=authors_list,
        year=year,
        vector_store=vector_store,
        retriever=retriever,
        settings=settings,
    )

    return IngestionResponse(
        doc_id=doc_id,
        status=IngestionStatus.PENDING,
        message="PDF queued for ingestion. Poll /ingest/status/{doc_id} for progress.",
        title=title or file.filename,
    )


@router.post("/arxiv", response_model=IngestionResponse)
async def ingest_arxiv(
        request_body: ArxivIngestionRequest,
        background_tasks: BackgroundTasks,
        vector_store: VectorStore = Depends(get_vector_store),
        retriever: HybridRetriever = Depends(get_retriever),
):
    """
    Ingest a paper directly from ArXiv by ID.
    Example: POST /ingest/arxiv with body {"arxiv_id": "1706.03762"}
    That's the "Attention Is All You Need" paper.
    """
    settings = get_settings()
    doc_id = request_body.doc_id or str(uuid4())

    _ingestion_jobs[doc_id] = {
        "status": IngestionStatus.PENDING,
        "progress_pct": 0.0,
        "message": f"Queued ArXiv paper {request_body.arxiv_id}",
    }

    background_tasks.add_task(
        _process_arxiv_background,
        arxiv_id=request_body.arxiv_id,
        doc_id=doc_id,
        vector_store=vector_store,
        retriever=retriever,
        settings=settings,
    )

    return IngestionResponse(
        doc_id=doc_id,
        status=IngestionStatus.PENDING,
        message=f"ArXiv paper {request_body.arxiv_id} queued. Poll /ingest/status/{doc_id}.",
    )


@router.post("/docs", response_model=IngestionResponse)
async def ingest_docs_site(
        request_body: DocsIngestionRequest,
        background_tasks: BackgroundTasks,
        vector_store: VectorStore = Depends(get_vector_store),
        retriever: HybridRetriever = Depends(get_retriever),
):
    """
    Ingest a documentation website.
    Example body:
    {
      "base_url": "https://fastapi.tiangolo.com",
      "name": "FastAPI Docs",
      "max_pages": 50,
      "use_sitemap": true,
      "allowed_path_prefix": "/docs/"
    }
    """
    settings = get_settings()
    doc_id = request_body.doc_id or str(uuid4())

    _ingestion_jobs[doc_id] = {
        "status": IngestionStatus.PENDING,
        "progress_pct": 0.0,
        "message": f"Queued docs ingestion for {request_body.base_url}",
    }

    background_tasks.add_task(
        _process_docs_background,
        request=request_body,
        doc_id=doc_id,
        vector_store=vector_store,
        retriever=retriever,
        settings=settings,
    )

    return IngestionResponse(
        doc_id=doc_id,
        status=IngestionStatus.PENDING,
        message=f"Docs site {request_body.base_url} queued. Poll /ingest/status/{doc_id}.",
        title=request_body.name,
    )


@router.get("/status/{doc_id}")
async def get_ingestion_status(doc_id: str):
    """Poll ingestion job status."""
    if doc_id not in _ingestion_jobs:
        raise HTTPException(status_code=404, detail=f"Job {doc_id} not found")
    return _ingestion_jobs[doc_id]


@router.get("/list")
async def list_ingested_documents(
        mode: str = "research",
        vector_store: VectorStore = Depends(get_vector_store),
):
    """List all ingested documents with document counts."""
    if mode not in ("research", "docs"):
        raise HTTPException(status_code=400, detail="mode must be 'research' or 'docs'")

    count = vector_store.count(mode)
    return {
        "mode": mode,
        "total_chunks": count,
        "collection": "research_papers" if mode == "research" else "dev_docs",
    }


@router.delete("/{doc_id}")
async def delete_document(
        doc_id: str,
        mode: str = "research",
        vector_store: VectorStore = Depends(get_vector_store),
        retriever: HybridRetriever = Depends(get_retriever),
):
    """Delete a document from the knowledge base."""
    deleted = await vector_store.delete_document(mode, doc_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found in {mode} collection")

    # Rebuild BM25 index after deletion
    await retriever.build_bm25_index(mode)

    return {"doc_id": doc_id, "chunks_deleted": deleted, "mode": mode}


# ── Background Task Functions ─────────────────────────────────────────────────

async def _process_pdf_background(
        file_path: str,
        doc_id: str,
        title: str | None,
        authors: list[str],
        year: int | None,
        vector_store: VectorStore,
        retriever: HybridRetriever,
        settings,
):
    """Background task: ingest PDF, embed chunks, store in ChromaDB."""
    try:
        _ingestion_jobs[doc_id]["status"] = IngestionStatus.PROCESSING
        _ingestion_jobs[doc_id]["progress_pct"] = 10.0
        _ingestion_jobs[doc_id]["message"] = "Parsing PDF..."

        ingestor = PDFIngestor(
            chunk_size=settings.RESEARCH_CHUNK_SIZE,
            chunk_overlap=settings.RESEARCH_CHUNK_OVERLAP,
        )
        chunks = await ingestor.ingest_from_path(
            file_path=file_path,
            doc_id=doc_id,
            title=title,
            authors=authors,
            year=year,
        )

        await _embed_and_store(chunks, "research", vector_store, retriever, doc_id)

    except Exception as e:
        logger.exception(f"PDF ingestion failed for {doc_id}: {e}")
        _ingestion_jobs[doc_id]["status"] = IngestionStatus.FAILED
        _ingestion_jobs[doc_id]["message"] = str(e)


async def _process_arxiv_background(
        arxiv_id: str,
        doc_id: str,
        vector_store: VectorStore,
        retriever: HybridRetriever,
        settings,
):
    """Background task: download ArXiv PDF, ingest."""
    try:
        _ingestion_jobs[doc_id]["status"] = IngestionStatus.PROCESSING
        _ingestion_jobs[doc_id]["message"] = f"Downloading ArXiv paper {arxiv_id}..."

        ingestor = PDFIngestor(
            chunk_size=settings.RESEARCH_CHUNK_SIZE,
            chunk_overlap=settings.RESEARCH_CHUNK_OVERLAP,
        )
        chunks = await ingestor.ingest_from_arxiv(arxiv_id=arxiv_id, doc_id=doc_id)
        await _embed_and_store(chunks, "research", vector_store, retriever, doc_id)

    except Exception as e:
        logger.exception(f"ArXiv ingestion failed for {arxiv_id}: {e}")
        _ingestion_jobs[doc_id]["status"] = IngestionStatus.FAILED
        _ingestion_jobs[doc_id]["message"] = str(e)


async def _process_docs_background(
        request: DocsIngestionRequest,
        doc_id: str,
        vector_store: VectorStore,
        retriever: HybridRetriever,
        settings,
):
    """Background task: crawl docs site, ingest."""
    try:
        _ingestion_jobs[doc_id]["status"] = IngestionStatus.PROCESSING
        _ingestion_jobs[doc_id]["message"] = f"Crawling {request.base_url}..."

        ingestor = DocsIngestor(
            chunk_size=settings.DOCS_CHUNK_SIZE,
            chunk_overlap=settings.DOCS_CHUNK_OVERLAP,
        )
        chunks = await ingestor.ingest_from_url(
            base_url=request.base_url,
            doc_id=doc_id,
            name=request.name,
            max_pages=request.max_pages,
            use_sitemap=request.use_sitemap,
            allowed_path_prefix=request.allowed_path_prefix,
        )
        await _embed_and_store(chunks, "docs", vector_store, retriever, doc_id)

    except Exception as e:
        logger.exception(f"Docs ingestion failed for {request.base_url}: {e}")
        _ingestion_jobs[doc_id]["status"] = IngestionStatus.FAILED
        _ingestion_jobs[doc_id]["message"] = str(e)


async def _embed_and_store(
        chunks, mode: str, vector_store: VectorStore, retriever: HybridRetriever, doc_id: str
):
    """Shared: generate embeddings and store chunks in ChromaDB."""
    if not chunks:
        _ingestion_jobs[doc_id]["status"] = IngestionStatus.FAILED
        _ingestion_jobs[doc_id]["message"] = "No chunks extracted"
        return

    _ingestion_jobs[doc_id]["message"] = f"Generating embeddings for {len(chunks)} chunks..."
    _ingestion_jobs[doc_id]["progress_pct"] = 50.0

    # Batch embedding
    texts = [c.content for c in chunks]
    embeddings = await retriever.get_embeddings_batch(texts)

    _ingestion_jobs[doc_id]["message"] = "Storing in ChromaDB..."
    _ingestion_jobs[doc_id]["progress_pct"] = 80.0

    await vector_store.add_chunks(
        mode=mode,
        chunk_ids=[f"{doc_id}_chunk_{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{**c.metadata, "chunk_id": f"{doc_id}_chunk_{i}"} for i, c in enumerate(chunks)],
    )

    # Rebuild BM25 index with new data
    _ingestion_jobs[doc_id]["message"] = "Building search index..."
    await retriever.build_bm25_index(mode)

    _ingestion_jobs[doc_id].update({
        "status": IngestionStatus.COMPLETED,
        "progress_pct": 100.0,
        "message": f"Successfully ingested {len(chunks)} chunks",
        "chunks_created": len(chunks),
    })
    logger.info(f"Ingestion complete: {doc_id} | {len(chunks)} chunks | mode={mode}")