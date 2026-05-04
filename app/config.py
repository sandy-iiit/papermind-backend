"""
INTERVIEW: "How do you manage configuration in a production FastAPI app?"
We use pydantic-settings which:
1. Reads from environment variables (12-factor app principle)
2. Type-validates everything at startup — fail fast if GROQ_API_KEY is missing
3. lru_cache makes it a singleton — settings object is created once, reused everywhere
4. Easy to override in tests: clear the cache, set os.environ, get fresh settings

INTERVIEW: "How would you switch from Groq to OpenAI?"
Change GROQ_MODEL_NAME env var + swap langchain_groq.ChatGroq for langchain_openai.ChatOpenAI
in config.get_llm(). All agent code is model-agnostic — they just call the LLM interface.
"""

from __future__ import annotations

from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── LLM ────────────────────────────────────────────────────────────────
    GROQ_API_KEY: str = Field(..., description="From console.groq.com — free")
    GROQ_MODEL_NAME: str = Field(
        default="llama3-70b-8192",
        description=(
            "Switchable without code changes. Options:\n"
            "  llama3-70b-8192   — best quality, 30 RPM free\n"
            "  llama3-8b-8192    — fastest, higher rate limit\n"
            "  mixtral-8x7b-32768 — long context (32k)\n"
            "  gemma-7b-it       — Google's open model"
        ),
    )
    GROQ_MAX_TOKENS: int = Field(default=2048)
    # INTERVIEW: "Why 0.1 temperature for RAG?"
    # High temp = creative but hallucinates. Low temp = factual, stays grounded in context.
    GROQ_TEMPERATURE: float = Field(default=0.1)

    # ── Embeddings ──────────────────────────────────────────────────────────
    # INTERVIEW: "Why sentence-transformers over OpenAI embeddings?"
    # Free: $0 per query. Local: no network round-trip.
    # all-MiniLM-L6-v2 scores within 5% of ada-002 on MTEB benchmark.
    EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION: int = Field(default=384)

    # ── ChromaDB ────────────────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = Field(default="./chroma_data")
    CHROMA_RESEARCH_COLLECTION: str = Field(default="research_papers")
    CHROMA_DOCS_COLLECTION: str = Field(default="dev_docs")

    # ── Retrieval ───────────────────────────────────────────────────────────
    # INTERVIEW: "Walk me through your retrieval pipeline."
    # Step 1: BM25 sparse + dense vector → top RETRIEVAL_TOP_K candidates
    # Step 2: Cross-encoder reranks → top RERANK_TOP_N passed to LLM
    # Why two-stage? Bi-encoder is O(1) with ANN. Cross-encoder is O(n*query_len).
    # You can't cross-encode all 10k docs, but you can rerank top-20.
    RETRIEVAL_TOP_K: int = Field(default=20)
    RERANK_TOP_N: int = Field(default=5)
    BM25_WEIGHT: float = Field(default=0.3)
    DENSE_WEIGHT: float = Field(default=0.7)

    # ── Chunking ────────────────────────────────────────────────────────────
    RESEARCH_CHUNK_SIZE: int = Field(default=512)
    RESEARCH_CHUNK_OVERLAP: int = Field(default=64)
    DOCS_CHUNK_SIZE: int = Field(default=384)
    DOCS_CHUNK_OVERLAP: int = Field(default=48)

    # ── Reranker ────────────────────────────────────────────────────────────
    # INTERVIEW: "Why this cross-encoder?"
    # ms-marco-MiniLM-L-6-v2: trained on MS MARCO passage ranking.
    # 22MB, runs on CPU in ~50ms for top-20 reranking. Free on HuggingFace.
    RERANKER_MODEL: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ── API ──────────────────────────────────────────────────────────────────
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    DEBUG: bool = Field(default=False)
    CORS_ORIGINS: list[str] = Field(
        # Include common local dev ports and 127.0.0.1 variants. Add production origins via env.
        default=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:3002",
            "http://127.0.0.1:3002",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )

    # ── Agents ──────────────────────────────────────────────────────────────
    # INTERVIEW: "How do you prevent infinite agent loops?"
    # MAX_AGENT_ITERATIONS is a hard cap. LangGraph's conditional edge checks
    # this counter — if reached, it routes to synthesizer regardless of critic score.
    MAX_AGENT_ITERATIONS: int = Field(default=3)
    MIN_CRITIC_SCORE: float = Field(default=0.70)

    # ── Pipeline Mode ──────────────────────────────────────────────────────
    # LIGHTWEIGHT_PIPELINE=true (default): Skip researcher + critic agents.
    #   Graph: retrieval → synthesizer → supervisor  (1 LLM call)
    # LIGHTWEIGHT_PIPELINE=false: Full multi-agent pipeline with retry loop.
    #   Graph: retrieval → researcher → critic → (retry?) → synthesizer → supervisor (3-10 LLM calls)
    LIGHTWEIGHT_PIPELINE: bool = Field(default=True)

    # ── Evaluation ──────────────────────────────────────────────────────────
    EVAL_SAMPLE_SIZE: int = Field(default=10)

    # ── Storage ──────────────────────────────────────────────────────────────
    UPLOAD_DIR: str = Field(default="./uploaded_pdfs")

    @field_validator("GROQ_TEMPERATURE")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    @field_validator("MIN_CRITIC_SCORE")
    @classmethod
    def validate_critic_score(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("MIN_CRITIC_SCORE must be 0.0–1.0")
        return v

    model_config = {"env_file": ".env", "case_sensitive": True}


@lru_cache()
def get_settings() -> Settings:
    """
    Singleton settings — created once, cached forever.
    INTERVIEW: "How do you test with different configs?"
    In tests: get_settings.cache_clear(), then monkeypatch os.environ, then call again.
    """
    return Settings()


def get_llm(settings: Settings | None = None):
    """
    INTERVIEW: "How do you make the LLM swappable?"
    This factory function is the single place that imports/instantiates the LLM.
    To switch providers: swap ChatGroq for ChatOpenAI, ChatAnthropic, etc.
    All agent code receives an LLM object and calls .invoke() / .astream() — provider-agnostic.

    Non-streaming variant — avoids ChatGroq streaming overhead when SSE is not needed.
    Used by: researcher, critic, non-streaming query endpoint.
    """
    from langchain_groq import ChatGroq

    s = settings or get_settings()
    return ChatGroq(
        api_key=s.GROQ_API_KEY,
        model=s.GROQ_MODEL_NAME,
        temperature=s.GROQ_TEMPERATURE,
        max_tokens=s.GROQ_MAX_TOKENS,
        streaming=False,
    )


def get_llm_streaming(settings: Settings | None = None):
    """
    Streaming variant of the LLM factory.
    streaming=True is needed for SSE token-by-token output in the /chat/stream endpoint.
    Used by: synthesizer (when called via the streaming endpoint).
    """
    from langchain_groq import ChatGroq

    s = settings or get_settings()
    return ChatGroq(
        api_key=s.GROQ_API_KEY,
        model=s.GROQ_MODEL_NAME,
        temperature=s.GROQ_TEMPERATURE,
        max_tokens=s.GROQ_MAX_TOKENS,
        streaming=True,
    )
