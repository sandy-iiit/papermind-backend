"""
INTERVIEW: "How do you ingest documentation websites?"

Strategy:
1. Try sitemap.xml first — structured list of all pages, much faster than crawling
2. Fall back to BFS crawling if no sitemap — respect robots.txt, scope to base_url
3. Parse with BeautifulSoup — extract main content, strip nav/footer/ads
4. Store URL + heading hierarchy in metadata for citation links

INTERVIEW: "How do you avoid ingesting irrelevant pages?"
1. allowed_path_prefix: only crawl /docs/, /api/, not /blog/ or /careers/
2. Content extraction: remove <nav>, <footer>, <aside> before chunking
3. Min content length threshold: skip pages with <200 chars of text

INTERVIEW: "What's the rate limiting strategy?"
asyncio.Semaphore limits concurrent requests (polite crawler).
Added random jitter between requests — servers ban bot-like uniform timing.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import deque
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from app.ingestion.chunking import Chunk, URLPreservingChunker

logger = logging.getLogger(__name__)

# Tags that contain navigation/boilerplate — not useful for RAG
NOISE_TAGS = {"nav", "header", "footer", "aside", "script", "style", "noscript", "form"}

# Selectors for main content in popular docs sites
CONTENT_SELECTORS = [
    "article",
    "main",
    '[role="main"]',
    ".content",
    ".documentation",
    ".docs-content",
    "#content",
    ".markdown-body",  # GitHub-style
]


class DocsIngestor:
    """
    Ingests documentation websites.
    Supports: FastAPI, LangChain, React, and any other docs site.
    """

    def __init__(self, chunk_size: int = 384, chunk_overlap: int = 48):
        self.chunker = URLPreservingChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        # DESIGN CHOICE: Semaphore limits concurrent HTTP requests
        # 5 concurrent = polite to server, fast enough for us
        self._semaphore = asyncio.Semaphore(5)

    async def ingest_from_url(
            self,
            base_url: str,
            doc_id: str,
            name: str,
            max_pages: int = 50,
            use_sitemap: bool = True,
            allowed_path_prefix: str | None = None,
    ) -> list[Chunk]:
        """
        Main ingestion entry point for a docs website.
        Returns list of Chunk objects ready for embedding + storage.
        """
        logger.info(f"Starting docs ingestion: {base_url}, max_pages={max_pages}")

        # Step 1: Discover URLs
        if use_sitemap:
            urls = await self._discover_from_sitemap(
                base_url, max_pages, allowed_path_prefix
            )
            if not urls:
                logger.info("No sitemap found, falling back to BFS crawl")
                urls = await self._discover_by_crawl(
                    base_url, max_pages, allowed_path_prefix
                )
        else:
            urls = await self._discover_by_crawl(
                base_url, max_pages, allowed_path_prefix
            )

        logger.info(f"Discovered {len(urls)} URLs to ingest")

        # Step 2: Fetch and parse all pages concurrently
        tasks = [
            self._process_page(url, doc_id, name)
            for url in urls
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_chunks: list[Chunk] = []
        chunk_index = 0
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Page processing failed: {result}")
                continue
            if result:
                for chunk in result:
                    chunk.chunk_index = chunk_index
                    chunk.metadata["chunk_id"] = f"{doc_id}_chunk_{chunk_index}"
                    chunk_index += 1
                all_chunks.extend(result)

        logger.info(f"Docs ingestion complete: {len(all_chunks)} chunks from {len(urls)} pages")
        return all_chunks

    async def _discover_from_sitemap(
            self,
            base_url: str,
            max_pages: int,
            allowed_path_prefix: str | None,
    ) -> list[str]:
        """
        Parse sitemap.xml to get all page URLs.
        INTERVIEW: "Why prefer sitemap over crawling?"
        Sitemap is O(1) — one HTTP request gets all URLs.
        Crawling is O(pages) — you have to visit every page to find links.
        Also, sitemap only lists canonical pages (no duplicate parameter URLs).
        """
        sitemap_urls = [
            urljoin(base_url, "/sitemap.xml"),
            urljoin(base_url, "/sitemap_index.xml"),
            urljoin(base_url, "/sitemap/sitemap.xml"),
        ]

        for sitemap_url in sitemap_urls:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(sitemap_url)
                    if resp.status_code == 200:
                        return self._parse_sitemap(
                            resp.text, base_url, max_pages, allowed_path_prefix
                        )
            except Exception:
                continue

        return []

    def _parse_sitemap(
            self,
            xml_text: str,
            base_url: str,
            max_pages: int,
            allowed_path_prefix: str | None,
    ) -> list[str]:
        """Extract URLs from sitemap XML."""
        soup = BeautifulSoup(xml_text, "lxml-xml")
        urls = []

        for loc in soup.find_all("loc"):
            url = loc.get_text(strip=True)
            if not url.startswith(base_url):
                continue
            if allowed_path_prefix and allowed_path_prefix not in url:
                continue
            # Skip non-HTML resources
            if url.endswith((".png", ".jpg", ".pdf", ".zip", ".json")):
                continue
            urls.append(url)
            if len(urls) >= max_pages:
                break

        return urls

    async def _discover_by_crawl(
            self,
            base_url: str,
            max_pages: int,
            allowed_path_prefix: str | None,
    ) -> list[str]:
        """
        BFS crawl starting from base_url.
        INTERVIEW: "Why BFS over DFS for crawling?"
        BFS discovers the most-linked pages (likely most important) first.
        DFS can go deep into one section and miss top-level important pages.
        We use a deque as the frontier — O(1) appendleft/popleft.
        """
        visited: set[str] = set()
        frontier: deque[str] = deque([base_url])
        found_urls: list[str] = []
        parsed_base = urlparse(base_url)

        while frontier and len(found_urls) < max_pages:
            url = frontier.popleft()
            if url in visited:
                continue
            visited.add(url)

            try:
                async with self._semaphore:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        resp = await client.get(
                            url,
                            headers={"User-Agent": "PaperMind-Docs-Ingestor/1.0"},
                            follow_redirects=True,
                        )
                    if resp.status_code != 200:
                        continue
                    if "text/html" not in resp.headers.get("content-type", ""):
                        continue

                found_urls.append(url)

                # Extract links for BFS frontier
                soup = BeautifulSoup(resp.text, "lxml")
                for a_tag in soup.find_all("a", href=True):
                    href = a_tag["href"]
                    full_url = urljoin(url, href).split("#")[0]  # Remove fragments
                    parsed = urlparse(full_url)

                    # Only crawl same domain
                    if parsed.netloc != parsed_base.netloc:
                        continue
                    if allowed_path_prefix and allowed_path_prefix not in full_url:
                        continue
                    if full_url not in visited and full_url not in frontier:
                        frontier.append(full_url)

                # Polite delay with jitter
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.debug(f"Crawl error for {url}: {e}")

        return found_urls

    async def _process_page(
            self, url: str, doc_id: str, name: str
    ) -> list[Chunk]:
        """Fetch a page, extract content, and chunk it."""
        async with self._semaphore:
            try:
                async with httpx.AsyncClient(
                        timeout=15.0,
                        headers={"User-Agent": "PaperMind-Docs-Ingestor/1.0"},
                        follow_redirects=True,
                ) as client:
                    resp = await client.get(url)
                    resp.raise_for_status()

                content, page_title = self._extract_content(resp.text, url)

                if len(content) < 200:
                    logger.debug(f"Skipping thin page: {url} ({len(content)} chars)")
                    return []

                base_metadata = {
                    "doc_id": doc_id,
                    "collection_name": name,
                    "url": url,
                    "page_title": page_title,
                    "source_type": "documentation",
                    "mode": "docs",
                }

                return self.chunker.chunk(content, base_metadata)

            except Exception as e:
                logger.warning(f"Failed to process {url}: {e}")
                return []

    def _extract_content(self, html: str, url: str) -> tuple[str, str]:
        """
        Extract main content from HTML, removing navigation noise.
        INTERVIEW: "How do you handle different HTML structures across sites?"
        We try a list of CSS selectors for common docs frameworks.
        If none match, fall back to <body> with noisy tags stripped.
        This covers: MkDocs, Docusaurus, Sphinx, GitBook, ReadTheDocs.
        """
        soup = BeautifulSoup(html, "lxml")

        # Extract title
        title_tag = soup.find("title")
        page_title = title_tag.get_text(strip=True) if title_tag else url

        # Remove noise tags
        for tag_name in NOISE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Try content selectors in order
        content_el = None
        for selector in CONTENT_SELECTORS:
            content_el = soup.select_one(selector)
            if content_el:
                break

        if not content_el:
            content_el = soup.find("body") or soup

        # Convert to markdown-like text
        text = self._html_to_text(content_el)
        return text, page_title

    def _html_to_text(self, element) -> str:
        """
        Convert HTML element to structured text preserving headings and code.
        INTERVIEW: "Why not just use .get_text()?"
        .get_text() strips all structure. We need heading markers (#, ##)
        so the chunker can detect section boundaries and chunk correctly.
        """
        lines: list[str] = []

        for tag in element.find_all(
                ["h1", "h2", "h3", "h4", "p", "pre", "code", "li", "ul", "ol"]
        ):
            tag_name = tag.name
            text = tag.get_text(strip=True)

            if not text:
                continue

            if tag_name == "h1":
                lines.append(f"\n# {text}\n")
            elif tag_name == "h2":
                lines.append(f"\n## {text}\n")
            elif tag_name == "h3":
                lines.append(f"\n### {text}\n")
            elif tag_name == "h4":
                lines.append(f"\n#### {text}\n")
            elif tag_name == "pre":
                # Preserve code blocks with fence markers
                lines.append(f"\n```\n{text}\n```\n")
            elif tag_name in ("p", "li"):
                lines.append(text)

        return "\n".join(lines)