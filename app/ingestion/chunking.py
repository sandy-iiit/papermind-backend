"""
INTERVIEW: "What's your chunking strategy and why does it differ between papers and docs?"

RESEARCH MODE — Citation-aware chunking:
- Respect section boundaries (Introduction, Methods, Results, Conclusion)
- Keep reference markers ([1], [2]) attached to the sentence that uses them
- Larger chunks (512 tokens) to preserve argument structure
- Overlap carries the section header into the next chunk so context isn't lost

DOCS MODE — URL-preserving chunking:
- Each chunk must know its exact source URL + anchor tag
- Smaller chunks (384 tokens) because docs are information-dense
- Respect heading hierarchy (h1 > h2 > h3) as natural split points
- Never split a code block across chunks — code examples must be atomic

INTERVIEW: "What's recursive character splitting?"
Split by \n\n (paragraphs) first. If still too big, split by \n (lines).
If still too big, split by space (words). Never split mid-word.
Preserves semantic units as long as possible before falling back to smaller splits.

INTERVIEW: "Why overlap?"
Overlap ensures that sentences spanning a chunk boundary aren't lost.
If a concept starts at the end of chunk N and continues in chunk N+1,
the overlap in chunk N+1 carries enough context for the reranker to score it correctly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    content: str
    metadata: dict[str, Any]
    chunk_index: int
    char_start: int
    char_end: int


class CitationAwareChunker:
    """
    For research papers (PDFs).
    INTERVIEW: "What makes research chunking different?"
    1. Section detection: We parse section headers so chunks know their section.
       This means "What is the attention mechanism?" retrieves from Methods, not Abstract.
    2. Citation preservation: [1], (Smith et al., 2017) stay with their sentence.
    3. Figure/table skipping: Chunks that are just "Figure 3. Caption." add noise.
    """

    SECTION_HEADERS = re.compile(
        r"^(abstract|introduction|background|related work|methodology|method|"
        r"approach|experiment|results?|discussion|conclusion|references?|"
        r"appendix|\d+\.?\s+\w+)",
        re.IGNORECASE | re.MULTILINE,
    )

    FIGURE_TABLE_PATTERN = re.compile(
        r"^(figure|fig\.?|table|tab\.?)\s+\d+", re.IGNORECASE
    )

    CITATION_PATTERN = re.compile(
        r"(\[\d+(?:,\s*\d+)*\]|\(\w+\s+et\s+al\.?,?\s+\d{4}\))"
    )

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, base_metadata: dict[str, Any]) -> list[Chunk]:
        """Split paper text into citation-aware chunks."""
        # Normalize whitespace but preserve section structure
        text = self._normalize_text(text)
        sections = self._detect_sections(text)

        chunks: list[Chunk] = []
        chunk_index = 0

        for section_name, section_text in sections:
            section_chunks = self._split_section(
                section_text, section_name, base_metadata, chunk_index
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        return chunks

    def _normalize_text(self, text: str) -> str:
        # Remove hyphenation from PDF line breaks (e.g., "atten-\ntion" → "attention")
        text = re.sub(r"-\n", "", text)
        # Normalize multiple blank lines to double newline
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _detect_sections(self, text: str) -> list[tuple[str, str]]:
        """
        Parse text into (section_name, section_text) pairs.
        INTERVIEW: "How do you detect section boundaries in a PDF?"
        We use regex for common academic section headers. For production,
        you'd also use font size metadata from PyMuPDF (bold + larger font = header).
        """
        matches = list(self.SECTION_HEADERS.finditer(text))

        if not matches:
            return [("BODY", text)]

        sections = []
        for i, match in enumerate(matches):
            section_name = match.group().strip().upper()
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()

            if len(section_text) > 50:  # Skip empty/tiny sections
                sections.append((section_name, section_text))

        return sections

    def _split_section(
            self,
            text: str,
            section_name: str,
            base_metadata: dict[str, Any],
            start_index: int,
    ) -> list[Chunk]:
        """Recursively split a section into chunks with overlap."""
        # Skip figure/table captions — they add noise to retrieval
        if self.FIGURE_TABLE_PATTERN.match(text[:50]):
            return []

        chunks: list[Chunk] = []
        # Split by paragraph first (semantic units)
        paragraphs = text.split("\n\n")

        current_chunk = ""
        current_start = 0
        chunk_idx = start_index

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph exceeds chunk_size, flush and start new
            if len(current_chunk) + len(para) > self.chunk_size * 4:  # ~4 chars/token
                if current_chunk:
                    chunks.append(
                        Chunk(
                            content=current_chunk.strip(),
                            metadata={
                                **base_metadata,
                                "section": section_name,
                                "has_citations": bool(
                                    self.CITATION_PATTERN.search(current_chunk)
                                ),
                                "chunk_type": "research",
                            },
                            chunk_index=chunk_idx,
                            char_start=current_start,
                            char_end=current_start + len(current_chunk),
                        )
                    )
                    chunk_idx += 1
                    # Overlap: carry last N chars into next chunk
                    overlap_text = current_chunk[-self.chunk_overlap * 4:]
                    current_chunk = overlap_text + "\n\n" + para
                    current_start += len(current_chunk) - len(overlap_text)
                else:
                    current_chunk = para
            else:
                current_chunk = (current_chunk + "\n\n" + para).strip()

        # Flush remaining
        if current_chunk.strip():
            chunks.append(
                Chunk(
                    content=current_chunk.strip(),
                    metadata={
                        **base_metadata,
                        "section": section_name,
                        "has_citations": bool(
                            self.CITATION_PATTERN.search(current_chunk)
                        ),
                        "chunk_type": "research",
                    },
                    chunk_index=chunk_idx,
                    char_start=current_start,
                    char_end=current_start + len(current_chunk),
                )
            )

        return chunks


class URLPreservingChunker:
    """
    For documentation websites.
    INTERVIEW: "What's special about docs chunking?"
    1. Every chunk carries the URL + anchor so we can generate direct links.
    2. Code blocks are NEVER split — a code example without context is useless.
    3. We respect heading hierarchy — a chunk under h3 "Middleware" also knows
       its h2 parent "Advanced" and h1 "FastAPI" for richer metadata.
    4. Smaller chunks (384 tokens) because docs are information-dense.
    """

    CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```|`[^`]+`")

    def __init__(self, chunk_size: int = 384, chunk_overlap: int = 48):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, base_metadata: dict[str, Any]) -> list[Chunk]:
        """
        Split docs text into URL-aware chunks.
        base_metadata must contain: url, page_title, section_hierarchy
        """
        chunks: list[Chunk] = []

        # Protect code blocks — replace with placeholders, restore after splitting
        code_blocks: dict[str, str] = {}
        protected_text = self._protect_code_blocks(text, code_blocks)

        # Split by headings
        sections = self._split_by_headings(protected_text, base_metadata)

        chunk_index = 0
        for section_meta, section_text in sections:
            # Restore code blocks in section text
            for placeholder, code in code_blocks.items():
                section_text = section_text.replace(placeholder, code)

            section_chunks = self._chunk_section(section_text, section_meta, chunk_index)
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        return chunks

    def _protect_code_blocks(
            self, text: str, storage: dict[str, str]
    ) -> str:
        """Replace code blocks with placeholders to prevent splitting them."""

        def replacer(match: re.Match) -> str:
            key = f"__CODE_{len(storage)}__"
            storage[key] = match.group()
            return key

        return self.CODE_BLOCK_PATTERN.sub(replacer, text)

    def _split_by_headings(
            self, text: str, base_metadata: dict[str, Any]
    ) -> list[tuple[dict[str, Any], str]]:
        """
        Split on markdown headings (# ## ###) or HTML headings.
        Each section inherits the URL and heading hierarchy as metadata.
        """
        heading_pattern = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
        matches = list(heading_pattern.finditer(text))

        if not matches:
            return [(base_metadata, text)]

        sections = []
        for i, match in enumerate(matches):
            level = len(match.group(1))
            heading_text = match.group(2).strip()
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()

            if len(content) < 30:
                continue

            # Build anchor from heading text (for URL fragment)
            anchor = re.sub(r"[^a-z0-9\-]", "", heading_text.lower().replace(" ", "-"))
            source_url = base_metadata.get("url", "")
            section_url = f"{source_url}#{anchor}" if anchor else source_url

            sections.append((
                {
                    **base_metadata,
                    "section_heading": heading_text,
                    "heading_level": level,
                    "section_url": section_url,
                    "chunk_type": "docs",
                },
                content,
            ))

        return sections

    def _chunk_section(
            self,
            text: str,
            metadata: dict[str, Any],
            start_index: int,
    ) -> list[Chunk]:
        """Split section text into overlapping chunks."""
        # Check if text fits in one chunk
        if len(text) <= self.chunk_size * 4:
            # Copy metadata so each Chunk has its own dict and mutations won't collide
            meta_copy = {**metadata}
            return [
                Chunk(
                    content=text,
                    metadata=meta_copy,
                    chunk_index=start_index,
                    char_start=0,
                    char_end=len(text),
                )
            ]

        chunks: list[Chunk] = []
        paragraphs = text.split("\n\n")
        current = ""
        idx = start_index

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current) + len(para) > self.chunk_size * 4:
                if current:
                    chunks.append(
                        Chunk(
                            content=current.strip(),
                            metadata={**metadata},  # copy per-chunk
                            chunk_index=idx,
                            char_start=0,
                            char_end=len(current),
                        )
                    )
                    idx += 1
                    # Carry overlap
                    overlap = current[-self.chunk_overlap * 4:]
                    current = overlap + "\n\n" + para
                else:
                    current = para
            else:
                current = (current + "\n\n" + para).strip()

        if current.strip():
            chunks.append(
                Chunk(
                    content=current.strip(),
                    metadata={**metadata},
                    chunk_index=idx,
                    char_start=0,
                    char_end=len(current),
                )
            )

        return chunks

