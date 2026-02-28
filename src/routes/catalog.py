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
        "bge-m3-matryoshka": "BGE-M3 Matryoshka ONNX INT8 - Fine-tuned for FR notarial | 1024D | CPU optimized",
        "pplx-embed-v1": "PPLX-Embed-v1 Q4 ONNX (0.6B) - Best FR notarial P@3=1.00 | 1024D | 380MB | CPU",
        "bge-reranker-v2-m3": "BGE-Reranker-v2-m3 ONNX INT8 (DEFAULT) - Cross-encoder | MIRACL FR 59.6 | 350ms | CPU",
        "mxbai-rerank-v2": "MXBAI-Rerank-V2 - Cross-encoder | BEIR 55.57 | 100+ languages",
        "mxbai-rerank-xsmall": "MXBAI-Rerank-XSmall - Lighter cross-encoder | 278M params",
        "vl-classifier": "Document Complexity Classifier - ResNet18 ONNX INT8 (93% accuracy, ~10ms)",
        "lfm25-vl": "LFM2.5-VL-1.6B Vision-Language | Document OCR | Edge-first CPU design",
        "whisper-base": "Whisper Base - Audio transcription (default) | 5.0% WER | ~1GB RAM",
        "whisper-small": "Whisper Small - Better accuracy transcription | 3.4% WER | ~2GB RAM",
    }

    return {
        "service": "Deposium Embeddings + Audio",
        "status": "running",
        "version": "14.0.0",
        "models": model_info,
        "recommended": "m2v-bge-m3-1024d for embeddings, bge-reranker-v2-m3 for reranking",
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
                "use_case": "Primary model - fast RAG, bulk processing, energy-efficient deployments"
            },
            "bge-m3-onnx": {
                "overall_mteb": 0.60,
                "dimensions": 1024,
                "size_mb": 571,
                "quantization": "INT8 ONNX",
                "device": "CPU optimized",
                "use_case": "High quality CPU embeddings when GPU not available"
            },
            "bge-m3-matryoshka": {
                "dimensions": 1024,
                "size_mb": 571,
                "quantization": "INT8 ONNX",
                "device": "CPU optimized",
                "fine_tuned": "FR notarial domain (Matryoshka loss)",
                "use_case": "Domain-specific French embeddings with flexible dimensionality"
            },
            "pplx-embed-v1": {
                "dimensions": 1024,
                "size_mb": 380,
                "quantization": "Q4 ONNX",
                "device": "CPU optimized",
                "params": "0.6B",
                "precision_at_3": 1.00,
                "spread_vs_bge_m3": "2.5x better",
                "latency_ms": 635,
                "use_case": "Best FR notarial embeddings - beats bge-m3 on domain benchmarks"
            },
            "bge-reranker-v2-m3": {
                "miracl_fr": 59.6,
                "params": "279M",
                "size_mb": 544,
                "speed": "350ms avg (Railway vCPU)",
                "cold_start": "275ms",
                "quantization": "INT8 ONNX (CPU optimized)",
                "precision_at_3": 1.00,
                "use_case": "DEFAULT reranker - cross-encoder, 10x faster than mxbai on CPU"
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
            }
        },
        "energy_benchmark": {
            "note": "Custom benchmark using CodeCarbon (Model2Vec not compatible with AIEnergyScore)",
            "m2v-bge-m3-1024d": {"texts_per_wh": 3191024, "throughput": "14,171 texts/s"},
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
            "size": 571000000,
            "digest": "bge-m3-onnx-int8",
            "modified_at": "2025-12-05T00:00:00Z",
            "details": "BGE-M3 ONNX INT8 (CPU) - High quality embeddings | MTEB: ~0.60 | 1024D"
        },
        {
            "name": "bge-m3-matryoshka",
            "size": 571000000,
            "digest": "bge-m3-matryoshka-1024d-onnx-int8",
            "modified_at": "2026-02-27T00:00:00Z",
            "details": "BGE-M3 Matryoshka ONNX INT8 - Fine-tuned FR notarial | 1024D | CPU optimized"
        },
        {
            "name": "pplx-embed-v1",
            "size": 380000000,
            "digest": "pplx-embed-v1-0.6b-q4-onnx",
            "modified_at": "2026-02-28T00:00:00Z",
            "details": "PPLX-Embed-v1 Q4 ONNX (0.6B) - Best FR notarial P@3=1.00 | 1024D | 380MB | CPU"
        },
        {
            "name": "bge-reranker-v2-m3",
            "size": 544000000,
            "digest": "bge-reranker-v2-m3-onnx-int8",
            "modified_at": "2026-02-27T00:00:00Z",
            "details": "BGE-Reranker-v2-m3 ONNX INT8 (DEFAULT) - Cross-encoder | MIRACL FR 59.6 | 350ms | CPU"
        },
        {
            "name": "vl-classifier",
            "size": 11000000,
            "digest": "resnet18-onnx-int8",
            "modified_at": "2025-10-22T00:00:00Z",
            "details": "Document Complexity Classifier - 93% accuracy, ~10ms latency"
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
