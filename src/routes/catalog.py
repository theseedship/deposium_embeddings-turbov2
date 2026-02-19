"""Catalog routes: root info, health, status, model listing."""
import logging
from fastapi import APIRouter, HTTPException

from .. import shared

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/")
async def root():
    model_info = {
        "m2v-bge-m3-1024d": "M2V-BGE-M3 (PRIMARY) - Distilled from BGE-M3 | MTEB: 0.47 | 3x energy efficient | 14k texts/s",
        "bge-m3-onnx": "BGE-M3 ONNX INT8 (CPU) - High quality embeddings | MTEB: ~0.60 | 1024D",
        "gemma-768d": "Gemma-768D Model2Vec (LEGACY) - Multilingual | MTEB: 0.55",
        "qwen3-rerank": "Qwen3 FP32 Reranking - FASTEST + BEST PRECISION (242ms for 3 docs!)",
        "vl-classifier": "Document Complexity Classifier - ResNet18 ONNX INT8 (93% accuracy, ~10ms)",
        "qwen2.5-coder-7b": "Qwen2.5-Coder-7B - LLM for code generation | 32K context | Tool calling",
        "qwen2.5-coder-3b": "Qwen2.5-Coder-3B - Lighter LLM | 32K context | Tool calling",
        "qwen2.5-coder-1.5b": "Qwen2.5-Coder-1.5B - Minimal LLM | 32K context",
        "whisper-base": "Whisper Base - Audio transcription (default) | 5.0% WER | ~1GB RAM",
        "whisper-small": "Whisper Small - Better accuracy transcription | 3.4% WER | ~2GB RAM",
    }

    return {
        "service": "Deposium Embeddings + Anthropic-compatible LLM API + Audio",
        "status": "running",
        "version": "13.0.0",
        "models": model_info,
        "recommended": "m2v-bge-m3-1024d for embeddings, qwen2.5-coder-7b for code generation via /v1/messages",
        "anthropic_api": {
            "endpoint": "/v1/messages",
            "description": "Anthropic-compatible API for local LLMs (Claude Code compatible)",
            "usage": "export ANTHROPIC_BASE_URL=http://localhost:8000 && claude --model qwen2.5-coder-7b",
            "features": ["streaming", "tool_calling", "system_prompts"],
        },
        "quality_metrics": {
            "m2v-bge-m3-1024d": {
                "overall_mteb": 0.47,
                "sts": 0.58,
                "classification": 0.66,
                "clustering": 0.28,
                "dimensions": 1024,
                "size_mb": 21,
                "params": "1.54B distilled to 21MB",
                "energy_efficiency": "3x more efficient than SentenceTransformers",
                "throughput": "14,171 texts/s (vs 2,587 for MiniLM)",
                "texts_per_wh": 3191024,
                "use_case": "Primary model - fast RAG, bulk processing, energy-efficient deployments"
            },
            "bge-m3-onnx": {
                "overall_mteb": 0.60,
                "dimensions": 1024,
                "size_mb": 150,
                "quantization": "INT8 ONNX",
                "device": "CPU optimized",
                "use_case": "High quality CPU embeddings when GPU not available"
            },
            "gemma-768d": {
                "overall_mteb": 0.55,
                "semantic_similarity": 0.59,
                "multilingual": 0.74,
                "dimensions": 768,
                "size_mb": 400,
                "params": "~50M",
                "use_case": "Legacy - multilingual support"
            },
            "qwen3-rerank": {
                "mteb_score": 64.33,
                "retrieval_score": 76.17,
                "params": "596M",
                "speed": "242ms for 3 docs (Railway vCPU)",
                "precision": "BEST (0.126 separation Paris-London)",
                "quantization": "FP32 (environment optimizations make it fastest!)",
                "use_case": "Reranking - best speed + precision on Railway vCPU"
            },
            "vl-classifier": {
                "architecture": "ResNet18 ONNX INT8",
                "accuracy": 0.93,
                "high_recall": 1.00,
                "params": "11M",
                "latency": "10-17ms on CPU",
                "size_mb": 11,
                "classes": ["LOW", "HIGH"],
                "use_case": "Document routing - simple (OCR) vs complex (VLM)"
            },
            "qwen2.5-coder-7b": {
                "params": "7B",
                "context_length": 32768,
                "quantization": "4-bit NF4",
                "vram_gb": 4.5,
                "features": ["code_generation", "tool_calling", "streaming"],
                "use_case": "Anthropic-compatible LLM for code generation (Claude Code compatible)"
            },
            "qwen2.5-coder-3b": {
                "params": "3B",
                "context_length": 32768,
                "quantization": "4-bit NF4",
                "vram_gb": 2.0,
                "features": ["code_generation", "tool_calling", "streaming"],
                "use_case": "Lighter LLM for code generation with less VRAM"
            },
            "qwen2.5-coder-1.5b": {
                "params": "1.5B",
                "context_length": 32768,
                "quantization": "4-bit NF4",
                "vram_gb": 1.2,
                "features": ["code_generation", "streaming"],
                "use_case": "Minimal LLM for basic code tasks with minimal VRAM"
            }
        },
        "energy_benchmark": {
            "note": "Custom benchmark using CodeCarbon (Model2Vec not compatible with AIEnergyScore)",
            "m2v-bge-m3-1024d": {"texts_per_wh": 3191024, "throughput": "14,171 texts/s"},
            "m2v-qwen3-1024d": {"texts_per_wh": 3047706, "throughput": "14,056 texts/s"},
            "all-MiniLM-L6-v2": {"texts_per_wh": 1129181, "throughput": "2,587 texts/s"},
            "comparison": "Model2Vec is ~3x more energy efficient than SentenceTransformers"
        }
    }


@router.get("/health")
async def health():
    if not shared.model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    status = shared.model_manager.get_status()
    return {
        "status": "healthy",
        "models_loaded": list(status.get("loaded_models", {})),
        "vram_used_mb": status.get("vram_used_mb", 0),
        "vram_free_mb": status.get("vram_free_mb", 0)
    }


@router.get("/api/status")
async def get_status():
    """Get detailed model manager status"""
    if not shared.model_manager:
        return {"error": "Model manager not initialized"}
    return shared.model_manager.get_status()


@router.get("/api/tags")
async def list_models():
    """Ollama-compatible endpoint to list models"""
    model_list = [
        {
            "name": "m2v-bge-m3-1024d",
            "size": 21000000,
            "digest": "m2v-bge-m3-1024d-distilled",
            "modified_at": "2025-12-05T00:00:00Z",
            "details": "M2V-BGE-M3 (PRIMARY) - Distilled from BGE-M3 | MTEB: 0.47 | 3x energy efficient | 14k texts/s"
        },
        {
            "name": "bge-m3-onnx",
            "size": 150000000,
            "digest": "bge-m3-onnx-int8",
            "modified_at": "2025-12-05T00:00:00Z",
            "details": "BGE-M3 ONNX INT8 (CPU) - High quality embeddings | MTEB: ~0.60 | 1024D"
        },
        {
            "name": "gemma-768d",
            "size": 400000000,
            "digest": "gemma-768d-m2v-deposium",
            "modified_at": "2025-10-13T00:00:00Z",
            "details": "Gemma-768D (LEGACY) - Multilingual | MTEB: 0.55"
        },
        {
            "name": "qwen3-rerank",
            "size": 600000000,
            "digest": "qwen3-rerank-fp32-optimized",
            "modified_at": "2025-10-14T00:00:00Z",
            "details": "Qwen3 FP32 Reranking - FASTEST (242ms) + BEST PRECISION on Railway vCPU!"
        },
        {
            "name": "vl-classifier",
            "size": 11000000,
            "digest": "resnet18-onnx-int8",
            "modified_at": "2025-10-22T00:00:00Z",
            "details": "Document Complexity Classifier - 93% accuracy, ~10ms latency"
        },
        {
            "name": "mxbai-embed-2d",
            "size": 800000000,
            "digest": "mxbai-embed-2d-large-v1",
            "modified_at": "2026-01-10T00:00:00Z",
            "details": "MXBAI-Embed-2D (24 layers) - 2D Matryoshka SOTA English | 1024D"
        },
        {
            "name": "mxbai-embed-2d-fast",
            "size": 400000000,
            "digest": "mxbai-embed-2d-large-v1-12layers",
            "modified_at": "2026-01-10T00:00:00Z",
            "details": "MXBAI-Embed-2D Fast (12 layers) - ~2x speedup | 768D"
        },
        {
            "name": "mxbai-embed-2d-turbo",
            "size": 250000000,
            "digest": "mxbai-embed-2d-large-v1-6layers",
            "modified_at": "2026-01-10T00:00:00Z",
            "details": "MXBAI-Embed-2D Turbo (6 layers) - ~4x speedup | 512D"
        },
        {
            "name": "mxbai-rerank-v2",
            "size": 250000000,
            "digest": "mxbai-rerank-base-v2-4bit",
            "modified_at": "2026-01-11T00:00:00Z",
            "details": "MXBAI-Rerank-V2 SOTA cross-encoder | BEIR 55.57 | 100+ languages | 4-bit NF4"
        },
        {
            "name": "mxbai-rerank-xsmall",
            "size": 150000000,
            "digest": "mxbai-rerank-xsmall-v1-4bit",
            "modified_at": "2026-01-11T00:00:00Z",
            "details": "MXBAI-Rerank-XSmall | 278M params | ~40% faster | 100+ languages | 4-bit NF4"
        },
        {
            "name": "lfm25-vl",
            "size": 3200000000,
            "digest": "lfm25-vl-1.6b",
            "modified_at": "2026-01-11T00:00:00Z",
            "details": "LFM2.5-VL-1.6B Vision-Language | Document OCR | Edge-first CPU design | 1.6B params"
        },
        {
            "name": "qwen2.5-coder-7b",
            "size": 4500000000,
            "digest": "qwen2.5-coder-7b-instruct-4bit",
            "modified_at": "2026-01-23T00:00:00Z",
            "details": "Qwen2.5-Coder-7B | LLM for /v1/messages | 32K context | Tool calling | 4-bit NF4"
        },
        {
            "name": "qwen2.5-coder-3b",
            "size": 2000000000,
            "digest": "qwen2.5-coder-3b-instruct-4bit",
            "modified_at": "2026-01-23T00:00:00Z",
            "details": "Qwen2.5-Coder-3B | Lighter LLM for /v1/messages | 32K context | Tool calling | 4-bit NF4"
        },
        {
            "name": "qwen2.5-coder-1.5b",
            "size": 1200000000,
            "digest": "qwen2.5-coder-1.5b-instruct-4bit",
            "modified_at": "2026-01-23T00:00:00Z",
            "details": "Qwen2.5-Coder-1.5B | Minimal LLM for /v1/messages | 32K context | 4-bit NF4"
        },
        {
            "name": "whisper-tiny",
            "size": 40000000,
            "digest": "whisper-tiny-ctranslate2",
            "modified_at": "2026-01-27T00:00:00Z",
            "details": "Whisper Tiny | Fastest transcription | 7.8% WER | ~40MB RAM"
        },
        {
            "name": "whisper-base",
            "size": 150000000,
            "digest": "whisper-base-ctranslate2",
            "modified_at": "2026-01-27T00:00:00Z",
            "details": "Whisper Base | Balanced speed/quality (default) | 5.0% WER | ~1GB RAM"
        },
        {
            "name": "whisper-small",
            "size": 500000000,
            "digest": "whisper-small-ctranslate2",
            "modified_at": "2026-01-27T00:00:00Z",
            "details": "Whisper Small | Better accuracy | 3.4% WER | ~2GB RAM"
        },
        {
            "name": "whisper-medium",
            "size": 1500000000,
            "digest": "whisper-medium-ctranslate2",
            "modified_at": "2026-01-27T00:00:00Z",
            "details": "Whisper Medium | High accuracy | 2.9% WER | ~5GB RAM"
        }
    ]

    return {"models": model_list}
