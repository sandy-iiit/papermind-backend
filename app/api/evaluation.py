"""
INTERVIEW: "How do you expose evaluation results to stakeholders?"
The evaluation endpoint runs RAGAS on a synthetic test set and returns metrics.
We also store evaluation history so you can see if quality improved after reingestion.
In the frontend, this powers the evaluation dashboard — line charts of metrics over time.

INTERVIEW: "When do you run evaluations in production?"
Nightly scheduled job (cron), or triggered after large ingestion batches.
Running per-query evaluation would cost too much in API credits.
We evaluate on a sample — statistically valid with N>=30 (we use N=10 for demo).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request

from app.agents.orchestrator import build_papermind_graph
from app.agents.state import PaperMindState
from app.config import get_settings
from app.evaluation.metrics import RAGASEvaluator
from app.evaluation.test_generator import TestSetGenerator
from app.models.schemas import EvalRequest, EvalResponse, Mode
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/evaluate", tags=["Evaluation"])

# In-memory eval history (replace with DB in production)
_eval_history: list[dict] = []


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_retriever(request: Request) -> HybridRetriever:
    return request.app.state.retriever


def get_reranker(request: Request) -> CrossEncoderReranker:
    return request.app.state.reranker


@router.post("/run", response_model=EvalResponse)
async def run_evaluation(
        eval_req: EvalRequest,
        vector_store: VectorStore = Depends(get_vector_store),
        retriever: HybridRetriever = Depends(get_retriever),
        reranker: CrossEncoderReranker = Depends(get_reranker),
):
    """
    Run full RAGAS evaluation pipeline.
    1. Generate synthetic QA test set
    2. Run each question through the RAG pipeline
    3. Collect answers + contexts
    4. Run RAGAS metrics
    5. Return scored metrics

    INTERVIEW: "How long does evaluation take?"
    For N=10 samples on Groq's free tier (30 RPM):
    - Test generation: ~3-5 LLM calls → ~10s
    - Pipeline answers: ~10 LLM calls → ~30s
    - RAGAS metrics: ~30-50 LLM calls → ~60-90s
    Total: ~2-3 minutes for N=10
    This is why we run it async/nightly in production.
    """
    settings = get_settings()
    start_time = time.time()
    mode = eval_req.mode.value

    if vector_store.count(mode) == 0:
        raise HTTPException(
            status_code=400,
            detail=f"No documents in {mode} collection. Please ingest documents first."
        )

    # Step 1: Generate synthetic QA test set
    logger.info(f"Generating test set for {mode} (n={eval_req.num_samples})")
    generator = TestSetGenerator(vector_store)
    qa_pairs = await generator.generate(mode=mode, num_samples=eval_req.num_samples)

    if not qa_pairs:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate test set. Check if documents are ingested."
        )

    # Step 2: Run each question through the RAG pipeline
    logger.info(f"Running {len(qa_pairs)} queries through pipeline...")
    graph = build_papermind_graph(retriever, reranker)

    ragas_samples = []
    for qa in qa_pairs:
        question = qa.get("question", "")
        if not question:
            continue

        try:
            initial_state: PaperMindState = {
                "query": question,
                "mode": mode,
                "session_id": "eval_run",
                "retrieval_top_k": settings.RETRIEVAL_TOP_K,
                "rerank_top_n": settings.RERANK_TOP_N,
                "iteration_count": 0,
                "max_iterations": 2,  # Fewer iterations during eval to save credits
                "retrieved_chunks": [],
                "final_answer": "",
                "citations": [],
                "critic_score": 0.0,
                "confidence_score": 0.0,
            }

            final_state = await graph.ainvoke(initial_state)

            ragas_samples.append({
                "question": question,
                "answer": final_state.get("final_answer", ""),
                "contexts": [
                    c.get("content", "")
                    for c in final_state.get("retrieved_chunks", [])
                ],
                "ground_truth": qa.get("ground_truth", ""),
            })
        except Exception as e:
            logger.warning(f"Pipeline failed for eval question: {e}")

    if not ragas_samples:
        raise HTTPException(status_code=500, detail="All pipeline runs failed during evaluation")

    # Step 3: Run RAGAS metrics
    logger.info(f"Running RAGAS evaluation on {len(ragas_samples)} samples...")
    evaluator = RAGASEvaluator()
    metrics = await evaluator.evaluate_pipeline(ragas_samples)

    elapsed_ms = (time.time() - start_time) * 1000

    response = EvalResponse(
        mode=eval_req.mode,
        metrics=metrics,
        num_samples=len(ragas_samples),
        evaluation_time_ms=elapsed_ms,
        model_used=settings.GROQ_MODEL_NAME,
        timestamp=datetime.utcnow(),
    )

    # Store in history
    _eval_history.append({
        **response.model_dump(),
        "mode": mode,
    })

    return response


@router.get("/history")
async def get_evaluation_history(mode: str = "research", limit: int = 20):
    """
    Get evaluation history — used by the frontend dashboard to plot metrics over time.
    INTERVIEW: "How do you build the evaluation dashboard?"
    Frontend fetches this endpoint periodically or after each eval run.
    Recharts (React) plots faithfulness, relevancy, recall, precision over time.
    Users can see if re-ingestion or tuning improved the pipeline.
    """
    filtered = [h for h in _eval_history if h.get("mode") == mode]
    return {
        "mode": mode,
        "history": filtered[-limit:],
        "total_runs": len(filtered),
    }