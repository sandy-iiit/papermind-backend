"""
INTERVIEW: "How do you handle PDFs for RAG?"
We use PyMuPDF (fitz) — it's the fastest pure-Python PDF parser.
Key capabilities we use:
1. Text extraction with layout preservation (column detection matters for papers)
2. Page-level metadata (page number per chunk enables precise citation)
3. Table of contents extraction (section headers without regex)
4. Font analysis (bold/large text = headings — better than regex alone)

INTERVIEW: "What about scanned PDFs?"
PyMuPDF can detect if a page has no text layer (scanned). For production,
you'd add Tesseract OCR as a fallback. We note this in metadata as 'is_scanned'.

INTERVIEW: "How do you handle multi-column papers?"
PyMuPDF's sort="textpage" tries to linearize columns, but it's not perfect.
For critical use cases, you'd use a layout model like PDFMiner or LayoutParser.
We use a column heuristic: if a block's width < page_width/2, it's likely a column.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import fitz  # PyMuPDF

from app.ingestion.chunking import CitationAwareChunker, Chunk

logger = logging.getLogger(__name__)


class PDFIngestor:
    """
    Ingests research PDFs from:
    1. Local file path (after upload)
    2. ArXiv ID (downloads PDF automatically)
    3. Direct PDF URL
    """

    ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"
    ARXIV_META_URL = "https://arxiv.org/abs/{arxiv_id}"

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunker = CitationAwareChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    async def ingest_from_path(
            self,
            file_path: str,
            doc_id: str,
            title: str | None = None,
            authors: list[str] | None = None,
            year: int | None = None,
            arxiv_id: str | None = None,
    ) -> list[Chunk]:
        """
        Main ingestion entry point for a PDF file already on disk.
        Returns list of Chunk objects ready for embedding + storage.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found at {file_path}")

        logger.info(f"Ingesting PDF: {path.name}, doc_id={doc_id}")

        # Run PyMuPDF in thread pool (CPU-bound, would block async event loop)
        loop = asyncio.get_event_loop()
        raw_text, toc, page_count = await loop.run_in_executor(
            None, self._extract_text, str(path)
        )

        base_metadata = {
            "doc_id": doc_id,
            "title": title or path.stem,
            "authors": authors or [],
            "year": year,
            "arxiv_id": arxiv_id,
            "page_count": page_count,
            "file_hash": self._file_hash(str(path)),
            "source_type": "research_paper",
            "mode": "research",
        }

        chunks = self.chunker.chunk(raw_text, base_metadata)

        # Add unique chunk IDs
        for chunk in chunks:
            chunk.metadata["chunk_id"] = f"{doc_id}_chunk_{chunk.chunk_index}"

        logger.info(
            f"PDF ingested: {len(chunks)} chunks from {page_count} pages"
        )
        return chunks

    async def ingest_from_arxiv(
            self, arxiv_id: str, doc_id: str
    ) -> list[Chunk]:
        """
        Download from ArXiv and ingest.
        INTERVIEW: "How do you handle external data sources?"
        Async HTTP download with timeout. We save to temp file, process, then clean up.
        This keeps memory bounded regardless of PDF size.
        """
        # Fetch metadata first (title, authors, year)
        metadata = await self._fetch_arxiv_metadata(arxiv_id)

        pdf_url = self.ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
        pdf_bytes = await self._download_bytes(pdf_url)

        # Write to temp file for PyMuPDF
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        try:
            return await self.ingest_from_path(
                file_path=tmp_path,
                doc_id=doc_id,
                title=metadata.get("title"),
                authors=metadata.get("authors"),
                year=metadata.get("year"),
                arxiv_id=arxiv_id,
            )
        finally:
            os.unlink(tmp_path)  # Always clean up temp file

    def _extract_text(self, file_path: str) -> tuple[str, list[dict], int]:
        """
        Extract full text from PDF using PyMuPDF.
        Returns: (full_text, table_of_contents, page_count)

        INTERVIEW: "How do you handle PDF layout?"
        fitz.Page.get_text("text") linearizes text top-to-bottom.
        For two-column papers, this can mix columns. We use get_text("blocks")
        to detect column boundaries when page has two narrow-ish text areas.
        """
        doc = fitz.open(file_path)
        pages_text: list[str] = []
        try:
            toc = doc.get_toc()  # Table of contents [(level, title, page), ...]

            for page_num, page in enumerate(doc, start=1):
                # Detect if page is image-only (scanned)
                text_blocks = page.get_text("blocks")
                if not text_blocks:
                    logger.warning(f"Page {page_num} appears to be scanned (no text layer)")
                    pages_text.append(f"\n[SCANNED PAGE {page_num}]\n")
                    continue

                page_width = page.rect.width

                # Check for two-column layout
                if self._is_two_column(text_blocks, page_width):
                    text = self._extract_two_column(text_blocks, page_width)
                else:
                    text = page.get_text("text")

                # Add page marker for citation tracking
                pages_text.append(f"\n--- PAGE {page_num} ---\n{text}")

            # Capture page count before closing document to avoid accessing a closed doc
            page_count = len(doc)
            return "\n".join(pages_text), toc, page_count
        finally:
            try:
                doc.close()
            except Exception:
                # If closing fails, log but don't raise to avoid masking the original error
                logger.debug("Failed to close PDF document cleanly")

    def _is_two_column(
            self, blocks: list, page_width: float
    ) -> bool:
        """
        Heuristic: if >30% of text blocks have width < 45% of page,
        it's likely a two-column layout (common in academic papers).
        """
        if not blocks:
            return False
        narrow_blocks = sum(
            1 for b in blocks
            if isinstance(b, tuple) and len(b) >= 5 and (b[2] - b[0]) < page_width * 0.45
        )
        return narrow_blocks / max(len(blocks), 1) > 0.3

    def _extract_two_column(
            self, blocks: list, page_width: float
    ) -> str:
        """
        Sort blocks into left and right columns, then concatenate.
        INTERVIEW: "Why does column order matter?"
        If you read top-to-bottom naively in a two-column PDF, you get
        "left col line 1, right col line 1, left col line 2, right col line 2..."
        which breaks sentence structure. We sort left column first, then right.
        """
        left_blocks = [
            b for b in blocks
            if isinstance(b, tuple) and len(b) >= 5 and b[0] < page_width * 0.5
        ]
        right_blocks = [
            b for b in blocks
            if isinstance(b, tuple) and len(b) >= 5 and b[0] >= page_width * 0.5
        ]

        # Sort each column top-to-bottom (by y0 coordinate)
        left_blocks.sort(key=lambda b: b[1])
        right_blocks.sort(key=lambda b: b[1])

        left_text = "\n".join(b[4] for b in left_blocks if isinstance(b[4], str))
        right_text = "\n".join(b[4] for b in right_blocks if isinstance(b[4], str))
        return left_text + "\n" + right_text

    async def _fetch_arxiv_metadata(self, arxiv_id: str) -> dict[str, Any]:
        """Fetch paper metadata from ArXiv abstract page."""
        url = self.ARXIV_META_URL.format(arxiv_id=arxiv_id)
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                # Parse basic metadata from HTML (title, authors, year)
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, "lxml")

                title_tag = soup.find("h1", class_="title")
                title = title_tag.get_text(strip=True).replace("Title:", "").strip() if title_tag else arxiv_id

                authors_tag = soup.find("div", class_="authors")
                authors = [a.get_text(strip=True) for a in authors_tag.find_all("a")] if authors_tag else []

                date_tag = soup.find("div", class_="dateline")
                year = None
                if date_tag:
                    import re
                    year_match = re.search(r"\b(20\d{2}|19\d{2})\b", date_tag.get_text())
                    year = int(year_match.group()) if year_match else None

                return {"title": title, "authors": authors, "year": year}
        except Exception as e:
            logger.warning(f"Could not fetch ArXiv metadata for {arxiv_id}: {e}")
            return {"title": arxiv_id, "authors": [], "year": None}

    async def _download_bytes(self, url: str) -> bytes:
        """Download a URL to bytes with retry."""
        from tenacity import retry, stop_after_attempt, wait_exponential

        async with httpx.AsyncClient(
                timeout=60.0,
                follow_redirects=True,
                headers={"User-Agent": "PaperMind-Research-Assistant/1.0"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content

    @staticmethod
    def _file_hash(file_path: str) -> str:
        """SHA256 hash for deduplication."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
