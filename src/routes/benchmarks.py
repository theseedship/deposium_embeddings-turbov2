"""Benchmark routes."""
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException

from .. import shared
from ..benchmarks import get_openbench_runner
from ..schemas.requests import BenchmarkRequest

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/benchmarks")
async def list_benchmarks(api_key: str = Depends(shared.verify_api_key)):
    """
    List available benchmark categories and providers.

    Returns information about:
    - Available categories (knowledge, coding, math, reasoning, cybersecurity, search)
    - Supported providers
    - Default configuration
    """
    try:
        runner = get_openbench_runner()
        return await runner.list_available_benchmarks()
    except Exception as e:
        logger.error(f"Benchmark list error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list benchmarks")


@router.post("/api/benchmarks/run")
async def run_benchmark(request: BenchmarkRequest, api_key: str = Depends(shared.verify_api_key)):
    """
    Run a benchmark evaluation.

    **Categories:**
    - `knowledge`: General knowledge (MMLU, TriviaQA)
    - `coding`: Code generation (HumanEval, MBPP)
    - `math`: Mathematical reasoning (GSM8K, MATH)
    - `reasoning`: Logic and deduction (ARC, HellaSwag)
    - `cybersecurity`: Security tasks
    - `search`: Retrieval quality (custom corpus supported)

    **For custom corpus evaluation:**
    ```json
    {
        "category": "search",
        "custom_corpus": [
            {
                "query": "What is machine learning?",
                "relevant_docs": ["doc1_content", "doc2_content"]
            }
        ]
    }
    ```

    **Returns:**
    - score: Overall benchmark score (0-1)
    - metrics: Detailed metrics
    - samples_evaluated: Number of samples tested
    - duration_seconds: Execution time
    """
    try:
        runner = get_openbench_runner()
        result = await runner.run_benchmark(
            category=request.category,
            provider=request.provider,
            model=request.model,
            custom_dataset=request.custom_corpus,
            sample_limit=request.sample_limit or 100
        )
        return result.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Benchmark run error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Benchmark execution failed")


@router.post("/api/benchmarks/corpus-eval")
async def evaluate_corpus(
    corpus_data: List[dict],
    provider: Optional[str] = "groq",
    model: Optional[str] = "llama-3.1-8b-instant",
    sample_limit: Optional[int] = 100,
    api_key: str = Depends(shared.verify_api_key)
):
    """
    Evaluate a Deposium corpus for retrieval quality.

    **Input format:**
    ```json
    [
        {
            "query": "Search query",
            "relevant_docs": ["Expected relevant document content..."],
            "context": "Optional context"
        }
    ]
    ```

    **Returns:**
    - Retrieval precision metrics
    - Per-query scores
    - Aggregate quality score
    """
    try:
        runner = get_openbench_runner()
        result = await runner.run_benchmark(
            category="search",
            provider=provider,
            model=model,
            custom_dataset=corpus_data,
            sample_limit=sample_limit
        )
        return result.to_dict()
    except Exception as e:
        logger.error(f"Corpus evaluation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Corpus evaluation failed")
