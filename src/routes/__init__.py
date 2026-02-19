"""Route module re-exports."""
from .catalog import router as catalog_router
from .embeddings import router as embeddings_router
from .reranking import router as reranking_router
from .classification import router as classification_router
from .vision import router as vision_router
from .audio import router as audio_router
from .benchmarks import router as benchmarks_router

__all__ = [
    "catalog_router",
    "embeddings_router",
    "reranking_router",
    "classification_router",
    "vision_router",
    "audio_router",
    "benchmarks_router",
]
