"""
Model Manager with Dynamic VRAM Management
===========================================

Manages model loading/unloading to stay within VRAM limits.
- Lazy loading: Models loaded only when needed
- LRU cache with VRAM limit (5GB on 6GB GPU)
- Priority system: Keep important models in memory
- Automatic unloading when VRAM exceeds limit
"""

import torch
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
from collections import OrderedDict
import time
import subprocess
import gc
import os
import sys

# Model2Vec imports
try:
    from model2vec import StaticModel
except ImportError:
    StaticModel = None
    
# SentenceTransformers imports  
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# Quantization imports for memory optimization
try:
    from transformers import BitsAndBytesConfig
    # Only test BitsAndBytesConfig availability, not bitsandbytes itself
    # to avoid import errors on systems without proper GPU support
    BNB_AVAILABLE = True
except ImportError:
    BitsAndBytesConfig = None
    BNB_AVAILABLE = False

# ONNX imports for BGE-M3 ONNX
try:
    import onnxruntime as ort
    from transformers import AutoTokenizer
    import numpy as np
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    AutoTokenizer = None
    np = None
    ONNX_AVAILABLE = False

logger = logging.getLogger(__name__)


class OnnxEmbeddingModel:
    """
    Wrapper for ONNX embedding models (like BGE-M3 ONNX INT8).
    Provides a compatible interface with encode() method.
    """

    def __init__(self, model_path: str, tokenizer_path: str = None):
        """
        Initialize ONNX embedding model.

        Args:
            model_path: Path to ONNX model file
            tokenizer_path: Path to tokenizer (defaults to model directory)
        """
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime and transformers required for ONNX models")

        self.model_path = model_path
        tokenizer_path = tokenizer_path or str(Path(model_path).parent)

        # Load ONNX session
        providers = ['CPUExecutionProvider']  # Force CPU for consistency
        self.session = ort.InferenceSession(model_path, providers=providers)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Get output name (dense_vecs for BGE-M3)
        self.output_name = "dense_vecs"

        logger.info(f"ONNX model loaded: {model_path}")

    def encode(self, texts, batch_size: int = 32, **kwargs):
        """
        Encode texts to embeddings.

        Args:
            texts: List of texts or single text
            batch_size: Batch size for encoding

        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=8192,  # BGE-M3 supports long context
                return_tensors="np"
            )

            # Run inference
            outputs = self.session.run(
                [self.output_name],
                {
                    "input_ids": inputs["input_ids"].astype(np.int64),
                    "attention_mask": inputs["attention_mask"].astype(np.int64)
                }
            )

            all_embeddings.append(outputs[0])

        return np.vstack(all_embeddings)


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    type: str  # "model2vec", "sentence_transformer", "sentence_transformer_2d", "mxbai_reranker", "causal_lm"
    path: Optional[str] = None  # Local path
    hub_id: Optional[str] = None  # HuggingFace ID
    priority: int = 0  # Higher = kept in memory longer
    estimated_vram_mb: int = 500  # Estimated VRAM usage
    device: str = "cuda"  # Device to load on
    # 2D Matryoshka options (for adaptive layer models)
    truncate_layers: Optional[int] = None  # Number of layers to use (None = all)
    truncate_dims: Optional[int] = None  # Embedding dimensions to keep (None = all)
    # Quantization options
    quantize_4bit: bool = False  # Enable 4-bit quantization (BitsAndBytes NF4)
    # Causal LM options
    context_length: int = 4096  # Maximum context length for causal LMs
    # Inference backend selection (for causal_lm models)
    # Options: "huggingface", "vllm_local", "vllm_remote", "remote_openai"
    # If None, uses LLM_BACKEND env var or auto-detects
    backend_type: Optional[str] = None


class ModelManager:
    """
    Manages models with dynamic VRAM allocation.
    
    Features:
    - Lazy loading (models loaded on first use)
    - VRAM monitoring and limits
    - LRU cache with priority system
    - Automatic model unloading when VRAM limit exceeded
    """
    
    def __init__(self, max_vram_mb: int = None):
        """
        Initialize model manager.

        Args:
            max_vram_mb: Maximum VRAM to use in MB (default from VRAM_LIMIT_MB env or 5GB)
        """
        self.max_vram_mb = max_vram_mb or int(os.getenv("VRAM_LIMIT_MB", "5000"))
        self.models: OrderedDict[str, Any] = OrderedDict()
        self.configs: Dict[str, ModelConfig] = {}
        self.last_used: Dict[str, float] = {}
        self.vram_usage: Dict[str, int] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Register available models
        self._register_models()
        
    def _register_models(self):
        """Register all available models with their configurations.

        Model HuggingFace IDs can be overridden via environment variables:
        - HF_MODEL_M2V_BGE_M3: Hub ID for m2v-bge-m3-1024d
        - HF_MODEL_BGE_M3_ONNX: Hub ID for bge-m3-onnx
        - HF_MODEL_GEMMA_768D: Hub ID for gemma-768d
        - HF_MODEL_QWEN3_EMBED: Hub ID for qwen3-embed
        """

        # M2V-BGE-M3-1024D (PRIMARY - best quality static embeddings)
        # MTEB: STS 0.58, Classification 0.66, Overall 0.47
        self.configs["m2v-bge-m3-1024d"] = ModelConfig(
            name="m2v-bge-m3-1024d",
            type="model2vec",
            path="models/m2v-bge-m3-1024d",
            hub_id=os.getenv("HF_MODEL_M2V_BGE_M3", "tss-deposium/m2v-bge-m3-1024d"),
            priority=1,  # Equal priority for all models
            estimated_vram_mb=500,
            device=self.device
        )

        # BGE-M3 ONNX INT8 (HIGH QUALITY - full transformer, CPU optimized)
        # Best quality but slower than Model2Vec
        self.configs["bge-m3-onnx"] = ModelConfig(
            name="bge-m3-onnx",
            type="onnx_embedding",
            path="models/bge-m3-onnx-int8",
            hub_id=os.getenv("HF_MODEL_BGE_M3_ONNX", "gpahal/bge-m3-onnx-int8"),
            priority=1,  # Equal priority for all models
            estimated_vram_mb=0,  # ONNX runs on CPU
            device="cpu"
        )

        # BGE-M3 Matryoshka (BEST - fine-tuned with MatryoshkaLoss, truncatable to 768/512/256D)
        # Full 1024D quality + Matryoshka: can truncate dimensions with minimal quality loss
        self.configs["bge-m3-matryoshka"] = ModelConfig(
            name="bge-m3-matryoshka",
            type="sentence_transformer",
            hub_id=os.getenv("HF_MODEL_BGE_M3_MATRYOSHKA", "tss-deposium/bge-m3-matryoshka-1024d"),
            priority=1,
            estimated_vram_mb=2300,  # BGE-M3 ~2.3GB on GPU
            device=self.device
        )

        # Gemma-768D (LEGACY - kept for backwards compatibility)
        self.configs["gemma-768d"] = ModelConfig(
            name="gemma-768d",
            type="model2vec",
            path="models/gemma-deposium-768d",
            hub_id=os.getenv("HF_MODEL_GEMMA_768D", "tss-deposium/gemma-deposium-768d"),
            priority=1,  # Equal priority for all models
            estimated_vram_mb=800,
            device=self.device
        )

        # VL Complexity Classifier (ONNX, CPU-based)
        self.configs["vl-classifier"] = ModelConfig(
            name="vl-classifier",
            type="onnx_classifier",
            path="src/models/complexity_classifier/model_quantized.onnx",
            priority=1,  # Equal priority for all models
            estimated_vram_mb=0,  # ONNX runs on CPU
            device="cpu"
        )

        # Qwen3-Embedding-0.6B (embeddings + reranking) - Memory optimized
        self.configs["qwen3-embed"] = ModelConfig(
            name="qwen3-embed",
            type="sentence_transformer",
            hub_id=os.getenv("HF_MODEL_QWEN3_EMBED", "Qwen/Qwen3-Embedding-0.6B"),
            priority=1,  # Equal priority for all models
            estimated_vram_mb=1200,  # ~2GB with float16 (was 4GB with float32)
            device=self.device
        )

        # Qwen3-rerank (alias for qwen3-embed)
        self.configs["qwen3-rerank"] = ModelConfig(
            name="qwen3-rerank",
            type="alias",
            path="qwen3-embed",  # Points to qwen3-embed
            priority=1,  # Equal priority for all models
            estimated_vram_mb=0,  # No additional VRAM (alias)
            device=self.device
        )

        # ============================================================
        # MXBAI-Embed-2D Models (2D Matryoshka - adaptive layer speedup)
        # https://huggingface.co/mixedbread-ai/mxbai-embed-2d-large-v1
        # 335M params, 1024D, 24 layers, English-only, Apache 2.0
        # ============================================================

        # MXBAI-Embed-2D Full (24 layers, 1024D - maximum quality)
        self.configs["mxbai-embed-2d"] = ModelConfig(
            name="mxbai-embed-2d",
            type="sentence_transformer_2d",
            hub_id=os.getenv("HF_MODEL_MXBAI_2D", "mixedbread-ai/mxbai-embed-2d-large-v1"),
            priority=1,
            estimated_vram_mb=800,
            device=self.device,
            truncate_layers=None,  # Use all 24 layers
            truncate_dims=None  # Use full 1024D
        )

        # MXBAI-Embed-2D Fast (12 layers, 768D - ~2x speedup, ~15% quality loss)
        self.configs["mxbai-embed-2d-fast"] = ModelConfig(
            name="mxbai-embed-2d-fast",
            type="sentence_transformer_2d",
            hub_id=os.getenv("HF_MODEL_MXBAI_2D", "mixedbread-ai/mxbai-embed-2d-large-v1"),
            priority=1,
            estimated_vram_mb=400,  # Less VRAM with fewer layers
            device=self.device,
            truncate_layers=12,  # Use 12 of 24 layers
            truncate_dims=768  # Truncate to 768D
        )

        # MXBAI-Embed-2D Turbo (6 layers, 512D - ~4x speedup, ~20% quality loss)
        self.configs["mxbai-embed-2d-turbo"] = ModelConfig(
            name="mxbai-embed-2d-turbo",
            type="sentence_transformer_2d",
            hub_id=os.getenv("HF_MODEL_MXBAI_2D", "mixedbread-ai/mxbai-embed-2d-large-v1"),
            priority=1,
            estimated_vram_mb=250,  # Even less VRAM
            device=self.device,
            truncate_layers=6,  # Use 6 of 24 layers
            truncate_dims=512  # Truncate to 512D
        )

        # ============================================================
        # LFM2.5-VL-1.6B (Vision-Language Model for Document OCR)
        # https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B
        # 1.6B params, Edge-first design (excellent CPU performance)
        # OCRBench v2: 41.44%, excellent document text extraction
        # Requires transformers >= 5.0 (trust_remote_code=True)
        # ============================================================
        self.configs["lfm25-vl"] = ModelConfig(
            name="lfm25-vl",
            type="vision_language",
            hub_id=os.getenv("HF_MODEL_LFM25_VL", "LiquidAI/LFM2.5-VL-1.6B"),
            priority=1,
            estimated_vram_mb=3200,  # ~3.2GB in BF16
            device=self.device
        )

        # ============================================================
        # MXBAI-Rerank-V2 (SOTA cross-encoder reranker, 100+ languages)
        # https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v2
        # 0.5B params (base), Qwen2 architecture, BEIR 55.57
        # Uses native mxbai-rerank library (not sentence-transformers)
        # 4-bit quantization: 1GB → ~250MB VRAM
        # ============================================================
        self.configs["mxbai-rerank-v2"] = ModelConfig(
            name="mxbai-rerank-v2",
            type="mxbai_reranker",
            hub_id=os.getenv("HF_MODEL_MXBAI_RERANK", "mixedbread-ai/mxbai-rerank-base-v2"),
            priority=1,
            estimated_vram_mb=250,  # ~250MB with 4-bit quantization (was 1GB FP32)
            device=self.device,
            quantize_4bit=True  # Enable 4-bit NF4 quantization
        )

        # ============================================================
        # MXBAI-Rerank-XSmall V1 (Lightweight DeBERTa-based, ~40% faster)
        # https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1
        # 100M params, DeBERTa architecture (V1 - no 4-bit quantization)
        # Fastest inference, good for high-throughput scenarios
        # ============================================================
        self.configs["mxbai-rerank-xsmall"] = ModelConfig(
            name="mxbai-rerank-xsmall",
            type="mxbai_reranker_v1",  # V1 uses DeBERTa, not Qwen2
            hub_id=os.getenv("HF_MODEL_MXBAI_RERANK_XSMALL", "mixedbread-ai/mxbai-rerank-xsmall-v1"),
            priority=1,
            estimated_vram_mb=200,  # ~200MB (no quantization for V1)
            device=self.device,
            quantize_4bit=False  # V1 doesn't support bitsandbytes quantization
        )

        # ============================================================
        # Causal Language Models (for Anthropic-compatible API)
        # These models are used with the /v1/messages endpoint
        # ============================================================

        # Qwen2.5-Coder-7B-Instruct (excellent for code generation)
        # https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
        # 7B params, 32K context, native tool calling
        self.configs["qwen2.5-coder-7b"] = ModelConfig(
            name="qwen2.5-coder-7b",
            type="causal_lm",
            hub_id=os.getenv("HF_MODEL_QWEN_CODER", "Qwen/Qwen2.5-Coder-7B-Instruct"),
            priority=1,
            estimated_vram_mb=4500,  # ~4.5GB with 4-bit quantization
            device=self.device,
            quantize_4bit=True,
            context_length=32768
        )

        # Qwen2.5-Coder-3B-Instruct (lighter alternative)
        # https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct
        # 3B params, 32K context, native tool calling
        self.configs["qwen2.5-coder-3b"] = ModelConfig(
            name="qwen2.5-coder-3b",
            type="causal_lm",
            hub_id=os.getenv("HF_MODEL_QWEN_CODER_3B", "Qwen/Qwen2.5-Coder-3B-Instruct"),
            priority=1,
            estimated_vram_mb=2000,  # ~2GB with 4-bit quantization
            device=self.device,
            quantize_4bit=True,
            context_length=32768
        )

        # Qwen2.5-Coder-1.5B-Instruct (minimal footprint)
        # https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct
        # 1.5B params, 32K context
        self.configs["qwen2.5-coder-1.5b"] = ModelConfig(
            name="qwen2.5-coder-1.5b",
            type="causal_lm",
            hub_id=os.getenv("HF_MODEL_QWEN_CODER_1B", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
            priority=1,
            estimated_vram_mb=1200,  # ~1.2GB with 4-bit quantization
            device=self.device,
            quantize_4bit=True,
            context_length=32768
        )

        # ============================================================
        # BitNet Models (CPU-only 1-bit quantization)
        # https://github.com/microsoft/BitNet
        # Use backend_type="bitnet" for these models
        # ============================================================

        # BitNet-b1.58-large (700M params, lightweight)
        # https://huggingface.co/1bitLLM/bitnet_b1_58-large
        # ~500MB memory, fast inference, good for simple tasks
        self.configs["bitnet-700m"] = ModelConfig(
            name="bitnet-700m",
            type="causal_lm",
            hub_id=os.getenv("HF_MODEL_BITNET_700M", "1bitLLM/bitnet_b1_58-large"),
            priority=1,
            estimated_vram_mb=0,  # CPU only
            device="cpu",
            quantize_4bit=False,  # Already 1-bit quantized
            context_length=2048,
            backend_type="bitnet"
        )

        # BitNet-b1.58-2B-4T (2.4B params, general purpose)
        # https://huggingface.co/1bitLLM/bitnet_b1_58-2B-4T
        # ~500MB memory, balanced quality/speed
        self.configs["bitnet-2b"] = ModelConfig(
            name="bitnet-2b",
            type="causal_lm",
            hub_id=os.getenv("HF_MODEL_BITNET_2B", "1bitLLM/bitnet_b1_58-2B-4T"),
            priority=1,
            estimated_vram_mb=0,  # CPU only
            device="cpu",
            quantize_4bit=False,  # Already 1-bit quantized
            context_length=2048,
            backend_type="bitnet"
        )

        # Llama3-8B-1.58-100B-tokens (8B params, high quality)
        # https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-100B-tokens
        # ~1GB memory, best quality for BitNet
        self.configs["bitnet-llama3-8b"] = ModelConfig(
            name="bitnet-llama3-8b",
            type="causal_lm",
            hub_id=os.getenv("HF_MODEL_BITNET_LLAMA3", "HF1BitLLM/Llama3-8B-1.58-100B-tokens"),
            priority=1,
            estimated_vram_mb=0,  # CPU only
            device="cpu",
            quantize_4bit=False,  # Already 1-bit quantized
            context_length=4096,
            backend_type="bitnet"
        )

    def get_vram_usage_mb(self) -> Tuple[int, int]:
        """
        Get current VRAM usage.

        Returns:
            (used_mb, free_mb)
        """
        if not torch.cuda.is_available():
            # CPU mode: use estimated VRAM from loaded models so that
            # _make_room_for_model() can correctly decide whether models
            # can coexist.  Returning (0, 0) caused every load to evict
            # all previously-loaded models — breaking embed + rerank
            # coexistence and triggering OOM during the swap.
            used = sum(self.vram_usage.values())
            return (used, self.max_vram_mb - used)
            
        try:
            # Try nvidia-smi first (more accurate)
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.free", 
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                used, free = map(int, result.stdout.strip().split(','))
                return (used, free)
        except Exception:
            pass

        # Fallback to PyTorch
        try:
            used = torch.cuda.memory_allocated() // (1024 * 1024)
            reserved = torch.cuda.memory_reserved() // (1024 * 1024)
            # Estimate based on reserved memory
            return (reserved, self.max_vram_mb - reserved)
        except Exception:
            return (0, self.max_vram_mb)
            
    def _unload_model(self, name: str):
        """Unload a model from VRAM."""
        if name not in self.models:
            return
            
        logger.info(f"Unloading model {name} to free VRAM")
        
        model = self.models[name]
        
        # Handle different model types for unloading
        try:
            from model2vec import StaticModel
            if isinstance(model, StaticModel):
                # Model2Vec doesn't have .cpu() method, just delete
                logger.info(f"Force deleting Model2Vec model: {name}")
                # model will be deleted at the end
            elif hasattr(model, 'cpu'):
                logger.info(f"Moving {name} to CPU")
                model.cpu()
            elif hasattr(model, 'to'):
                logger.info(f"Moving {name} to CPU via .to()")
                model.to('cpu')
            else:
                # ONNX models or other types - just delete
                logger.info(f"Force deleting model: {name}")
                # model will be deleted at the end
        except ImportError:
            # Fallback if Model2Vec not available
            if hasattr(model, 'cpu'):
                model.cpu()
            elif hasattr(model, 'to'):
                model.to('cpu')
        except Exception as e:
            logger.warning(f"Error during model unload (non-fatal): {e}")

        # Remove from cache
        del self.models[name]
        if name in self.vram_usage:
            del self.vram_usage[name]

        # CRITICAL FIX: Break circular references for SentenceTransformer
        # The logs showed SentenceTransformerModelCardData holding a reference
        if hasattr(model, "model_card_data"):
            logger.info(f"Breaking model_card_data reference for {name}")
            model.model_card_data = None
            
        # Clear other potential circular references
        if hasattr(model, "_modules"):
            model._modules.clear()

        # Log reference count (debug)
        try:
            ref_count = sys.getrefcount(model)
            logger.info(f"Model {name} refcount before deletion: {ref_count} (should be low)")
            
            if ref_count > 2:
                logger.warning(f"⚠️ High refcount detected for {name}!")
                referrers = gc.get_referrers(model)
                for i, ref in enumerate(referrers):
                    if ref is locals():
                        continue # Ignore local scope
                    logger.warning(f"  Ref {i}: {type(ref)}")
                    # Don't log full content of large objects, just type/str
                    try:
                        s = str(ref)
                        if len(s) > 200: s = s[:200] + "..."
                        logger.warning(f"    Value: {s}")
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Error checking refcount: {e}")
        
        # Clear torch compiler caches if applicable
        if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "reset"):
            try:
                logger.info("Resetting torch._dynamo to clear compiler caches")
                torch._dynamo.reset()
            except Exception as e:
                logger.warning(f"Failed to reset torch._dynamo: {e}")
        
        # Aggressive cleanup for SentenceTransformer
        if hasattr(model, "_modules"):
            try:
                for module_name, module in model._modules.items():
                    # Move to CPU to be sure
                    if hasattr(module, "cpu"):
                        module.cpu()
                    # Explicitly delete large attributes if possible
                    if hasattr(module, "auto_model"):
                        logger.info(f"Deleting auto_model from module {module_name}")
                        del module.auto_model
                model._modules.clear()
            except Exception as e:
                logger.warning(f"Error during aggressive module cleanup: {e}")

        # Delete model reference
        del model
            
        # Force garbage collection multiple times
        gc.collect()
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Force release of system memory back to OS (Linux only)
        try:
            import ctypes
            import platform
            if platform.system() == "Linux":
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
                logger.info("Called malloc_trim to release system memory")
        except Exception as e:
            logger.warning(f"Failed to call malloc_trim: {e}")
            
        logger.info(f"Successfully unloaded model: {name}")
            
    def _make_room_for_model(self, required_mb: int):
        """
        Free up VRAM for a new model.
        
        Args:
            required_mb: VRAM needed in MB
        """
        used_mb, free_mb = self.get_vram_usage_mb()
        
        if free_mb >= required_mb:
            return  # Enough space available
            
        # Sort models by priority and last used time
        models_to_unload = []
        for name, model in self.models.items():
            if name in self.configs:
                priority = self.configs[name].priority
                last_used = self.last_used.get(name, 0)
                models_to_unload.append((priority, last_used, name))
                
        # Sort by priority (ascending) then by last used (ascending)
        models_to_unload.sort(key=lambda x: (x[0], x[1]))
        
        # Unload models until we have enough space
        for _, _, name in models_to_unload:
            self._unload_model(name)
            used_mb, free_mb = self.get_vram_usage_mb()
            if free_mb >= required_mb:
                break

    # Models that require trust_remote_code for custom architectures.
    # All other models default to trust_remote_code=False for security.
    TRUSTED_REMOTE_CODE_PREFIXES = (
        "Qwen/",           # Qwen models use custom tokenizer/model code
        "LiquidAI/",       # LFM2.5-VL custom architecture
        "mixedbread-ai/",  # mxbai models use custom pooling
    )

    def _trust_remote_code(self, hub_id: str) -> bool:
        """Check if a model hub_id is whitelisted for trust_remote_code."""
        if os.getenv("HF_TRUST_REMOTE_CODE", "").lower() == "true":
            return True  # Explicit env override
        return any(hub_id.startswith(prefix) for prefix in self.TRUSTED_REMOTE_CODE_PREFIXES)

    def _load_model(self, name: str) -> Any:
        """
        Load a model into memory.

        Args:
            name: Model name

        Returns:
            Loaded model
        """
        config = self.configs[name]
        
        # Handle alias
        if config.type == "alias":
            return self.get_model(config.path)
            
        logger.info(f"Loading model {name} (type: {config.type}, priority: {config.priority})")
        
        # Make room in VRAM if needed
        self._make_room_for_model(config.estimated_vram_mb)
        
        model = None
        
        try:
            if config.type == "model2vec":
                # Load Model2Vec model
                if StaticModel is None:
                    raise ImportError("model2vec not installed")
                    
                # Try local path first
                local_path = Path(config.path) if config.path else None
                docker_path = Path(f"/app/local_models/{name}")
                
                if docker_path.exists():
                    model = StaticModel.from_pretrained(str(docker_path))
                    logger.info(f"✅ {name} loaded from Docker image")
                elif local_path and local_path.exists():
                    model = StaticModel.from_pretrained(str(local_path))
                    logger.info(f"✅ {name} loaded from local path")
                elif config.hub_id:
                    model = StaticModel.from_pretrained(config.hub_id)
                    logger.info(f"✅ {name} loaded from HuggingFace")
                else:
                    raise ValueError(f"No valid path for {name}")
                    
            elif config.type == "onnx_classifier":
                # Load ONNX classifier model
                try:
                    import onnxruntime as ort
                except ImportError:
                    raise ImportError("onnxruntime not installed")

                model_path = Path(config.path)
                if not model_path.exists():
                    raise ValueError(f"ONNX model not found at {config.path}")

                # Create ONNX inference session
                providers = ['CPUExecutionProvider']  # Force CPU for consistency
                model = ort.InferenceSession(str(model_path), providers=providers)
                logger.info(f"✅ {name} loaded (ONNX on CPU)")

            elif config.type == "onnx_embedding":
                # Load ONNX embedding model (BGE-M3 ONNX INT8)
                if not ONNX_AVAILABLE:
                    raise ImportError("onnxruntime and transformers required for ONNX embedding models")

                # Try local path first, then Docker path
                local_path = Path(config.path) if config.path else None
                docker_path = Path(f"/app/local_models/{name}")

                if docker_path.exists():
                    model_file = docker_path / "model_quantized.onnx"
                    model = OnnxEmbeddingModel(str(model_file), str(docker_path))
                    logger.info(f"✅ {name} loaded from Docker image (ONNX CPU)")
                elif local_path and local_path.exists():
                    model_file = local_path / "model_quantized.onnx"
                    model = OnnxEmbeddingModel(str(model_file), str(local_path))
                    logger.info(f"✅ {name} loaded from local path (ONNX CPU)")
                else:
                    # Download from HuggingFace
                    from huggingface_hub import snapshot_download
                    cache_dir = snapshot_download(repo_id=config.hub_id)
                    model_file = Path(cache_dir) / "model_quantized.onnx"
                    model = OnnxEmbeddingModel(str(model_file), cache_dir)
                    logger.info(f"✅ {name} loaded from HuggingFace (ONNX CPU)")

            elif config.type == "sentence_transformer":
                # Load SentenceTransformer model
                if SentenceTransformer is None:
                    raise ImportError("sentence-transformers not installed")
                
                # Special handling for Qwen3 models - optimize memory usage
                if name in ["qwen3-embed", "qwen3-rerank"]:
                    # Try 4-bit quantization first if available
                    quantization_attempted = False
                    
                    if BNB_AVAILABLE and BitsAndBytesConfig is not None and self.device == "cuda":
                        try:
                            # Use 4-bit quantization for memory efficiency
                            logger.info(f"Attempting to load {name} with 4-bit quantization...")
                            
                            # Create quantization config
                            quantization_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_use_double_quant=True,  # Further compression
                                bnb_4bit_quant_type="nf4",  # Normal Float 4 for better quality
                                bnb_4bit_compute_dtype=torch.float16  # Compute in fp16 for speed
                            )
                            
                            # Load with quantization
                            model = SentenceTransformer(
                                config.hub_id,
                                trust_remote_code=self._trust_remote_code(config.hub_id),
                                device=config.device,
                                model_kwargs={
                                    "quantization_config": quantization_config,
                                    "device_map": "auto",
                                    "torch_dtype": torch.float16
                                }
                            )
                            logger.info(f"✅ {name} loaded with 4-bit quantization (memory usage reduced by ~75%)")
                            quantization_attempted = True
                        except Exception as e:
                            logger.warning(f"4-bit quantization failed: {e}. Falling back to float16...")
                            quantization_attempted = False
                    
                    # Fallback to float16 for memory reduction
                    if not quantization_attempted:
                        logger.info(f"Loading {name} with float16 precision for memory optimization")
                        model = SentenceTransformer(
                            config.hub_id,
                            trust_remote_code=self._trust_remote_code(config.hub_id),
                            device=config.device,
                            model_kwargs={
                                "torch_dtype": torch.float16
                            }
                        )
                        logger.info(f"✅ {name} loaded with float16 (memory usage reduced by ~50%)")
                else:
                    # Standard loading for other models
                    model = SentenceTransformer(
                        config.hub_id,
                        trust_remote_code=self._trust_remote_code(config.hub_id),
                        device=config.device
                    )
                    logger.info(f"✅ {name} loaded (SentenceTransformer)")

                # Store truncate_dims for Matryoshka models
                if config.truncate_dims:
                    model._truncate_dims = config.truncate_dims
                    logger.info(f"   Embeddings will be truncated to {config.truncate_dims}D")

                # Apply torch.compile if enabled (Linux only usually)
                if os.getenv("ENABLE_TORCH_COMPILE", "0") == "1" and hasattr(torch, "compile"):
                    try:
                        logger.info(f"🚀 Compiling {name} with torch.compile()...")
                        model = torch.compile(model)
                        logger.info(f"✅ {name} compiled!")
                    except Exception as e:
                        logger.warning(f"Failed to compile {name}: {e}")

                # Log Attention Implementation
                try:
                    # Check if using SDPA (Scaled Dot Product Attention)
                    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                        logger.info(f"ℹ️  {name} using PyTorch SDPA (Flash Attention backend if available)")
                except Exception:
                    pass

            elif config.type == "sentence_transformer_2d":
                # Load 2D Matryoshka SentenceTransformer (supports layer truncation)
                # https://huggingface.co/mixedbread-ai/mxbai-embed-2d-large-v1
                if SentenceTransformer is None:
                    raise ImportError("sentence-transformers not installed")

                logger.info(f"Loading 2D Matryoshka model {name}...")
                model = SentenceTransformer(
                    config.hub_id,
                    trust_remote_code=self._trust_remote_code(config.hub_id),
                    device=config.device
                )

                # Apply layer truncation if specified
                if config.truncate_layers is not None:
                    original_layers = self._get_model_layer_count(model)
                    if original_layers and config.truncate_layers < original_layers:
                        self._truncate_model_layers(model, config.truncate_layers)
                        logger.info(f"✅ {name} truncated from {original_layers} to {config.truncate_layers} layers (~{original_layers/config.truncate_layers:.1f}x speedup)")
                    else:
                        logger.info(f"✅ {name} loaded with all {original_layers} layers")
                else:
                    logger.info(f"✅ {name} loaded (2D Matryoshka, full layers)")

                # Store truncate_dims for use during encoding
                if config.truncate_dims:
                    model._truncate_dims = config.truncate_dims
                    logger.info(f"   Embeddings will be truncated to {config.truncate_dims}D")

            elif config.type == "vision_language":
                # Load Vision-Language Model (LFM2.5-VL)
                # Requires transformers >= 5.0 with trust_remote_code=True
                try:
                    from transformers import AutoModelForImageTextToText, AutoProcessor
                except ImportError:
                    raise ImportError("transformers >= 5.0 required for vision-language models")

                logger.info(f"Loading Vision-Language model {name}...")

                # Load model with trust_remote_code for custom architecture
                model_kwargs = {
                    "torch_dtype": torch.bfloat16 if config.device == "cuda" else torch.float32,
                    "low_cpu_mem_usage": True,
                    "trust_remote_code": self._trust_remote_code(config.hub_id),
                }

                if config.device == "cuda":
                    model_kwargs["device_map"] = {"": config.device}

                vlm_model = AutoModelForImageTextToText.from_pretrained(
                    config.hub_id,
                    **model_kwargs
                )

                # Load processor (tokenizer + image processor)
                vlm_processor = AutoProcessor.from_pretrained(
                    config.hub_id,
                    trust_remote_code=self._trust_remote_code(config.hub_id)
                )

                # Move to device if CPU
                if config.device == "cpu":
                    vlm_model = vlm_model.to(config.device)

                # Store as tuple (model, processor)
                model = (vlm_model, vlm_processor)
                logger.info(f"✅ {name} loaded (Vision-Language Model)")

            elif config.type == "mxbai_reranker":
                # Load MXBAI Rerank V2 (SOTA cross-encoder)
                # Uses native mxbai-rerank library with optional 4-bit quantization
                try:
                    from mxbai_rerank import MxbaiRerankV2
                except ImportError:
                    raise ImportError("mxbai-rerank not installed. Install with: pip install mxbai-rerank")

                logger.info(f"Loading MXBAI Reranker {name}...")

                # Prepare kwargs for model loading
                model_kwargs = {
                    "device": config.device,
                    "torch_dtype": torch.float16,  # Use float16 by default
                }

                # Apply 4-bit quantization if enabled and on CUDA
                if config.quantize_4bit and config.device == "cuda" and BNB_AVAILABLE:
                    try:
                        logger.info(f"Applying 4-bit NF4 quantization to {name}...")
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.float16
                        )
                        model_kwargs["quantization_config"] = quantization_config
                        model_kwargs["device_map"] = "auto"
                        # Remove device from kwargs when using device_map
                        del model_kwargs["device"]
                    except Exception as e:
                        logger.warning(f"4-bit quantization setup failed, falling back to float16: {e}")

                model = MxbaiRerankV2(config.hub_id, **model_kwargs)

                # Enhanced logging for GPU/CPU and quantization status
                if "quantization_config" in model_kwargs:
                    logger.info(f"✅ {name} loaded with 4-bit NF4 on CUDA (~75% VRAM reduction)")
                    if torch.cuda.is_available():
                        mem_mb = torch.cuda.memory_allocated() / 1024 / 1024
                        logger.info(f"   VRAM allocated: {mem_mb:.0f}MB")
                elif config.device == "cuda":
                    logger.info(f"✅ {name} loaded on CUDA (float16, no quantization)")
                    if torch.cuda.is_available():
                        mem_mb = torch.cuda.memory_allocated() / 1024 / 1024
                        logger.info(f"   VRAM allocated: {mem_mb:.0f}MB")
                else:
                    logger.warning(f"⚠️ {name} loaded on CPU - inference will be slower")

            elif config.type == "mxbai_reranker_v1":
                # Load MXBAI Rerank V1 (DeBERTa-based, lighter alternative)
                # V1 models don't support bitsandbytes quantization
                try:
                    from mxbai_rerank import MxbaiRerankV1
                except ImportError:
                    raise ImportError("mxbai-rerank not installed. Install with: pip install mxbai-rerank")

                logger.info(f"Loading MXBAI Reranker V1 {name}...")

                # V1 uses simpler kwargs (no quantization support)
                model_kwargs = {
                    "device": config.device,
                }

                model = MxbaiRerankV1(config.hub_id, **model_kwargs)

                if config.device == "cuda":
                    logger.info(f"✅ {name} loaded on CUDA (V1 DeBERTa)")
                    if torch.cuda.is_available():
                        mem_mb = torch.cuda.memory_allocated() / 1024 / 1024
                        logger.info(f"   VRAM allocated: {mem_mb:.0f}MB")
                else:
                    logger.info(f"✅ {name} loaded on CPU (V1 DeBERTa - lightweight)")

            elif config.type == "causal_lm":
                # Load Causal Language Model for Anthropic-compatible API
                # Supports Qwen2.5-Coder, Llama, Mistral, etc.
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                except ImportError:
                    raise ImportError("transformers not installed")

                logger.info(f"Loading Causal LM {name}...")

                # Prepare model kwargs
                model_kwargs = {
                    "torch_dtype": torch.float16,
                    "low_cpu_mem_usage": True,
                    "trust_remote_code": self._trust_remote_code(config.hub_id),
                }

                # Apply 4-bit quantization if enabled and on CUDA
                if config.quantize_4bit and config.device == "cuda" and BNB_AVAILABLE:
                    try:
                        logger.info(f"Applying 4-bit NF4 quantization to {name}...")
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.float16
                        )
                        model_kwargs["quantization_config"] = quantization_config
                        model_kwargs["device_map"] = "auto"
                    except Exception as e:
                        logger.warning(f"4-bit quantization setup failed: {e}")
                        model_kwargs["device_map"] = {"": config.device}
                else:
                    if config.device == "cuda":
                        model_kwargs["device_map"] = {"": config.device}

                # Load model
                causal_model = AutoModelForCausalLM.from_pretrained(
                    config.hub_id,
                    **model_kwargs
                )

                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    config.hub_id,
                    trust_remote_code=self._trust_remote_code(config.hub_id)
                )

                # Ensure pad token is set
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                # Store as tuple (model, tokenizer)
                model = (causal_model, tokenizer)

                # Log loading info
                if config.quantize_4bit and "quantization_config" in model_kwargs:
                    logger.info(f"✅ {name} loaded with 4-bit NF4 quantization")
                else:
                    logger.info(f"✅ {name} loaded (float16)")

                if torch.cuda.is_available():
                    mem_mb = torch.cuda.memory_allocated() / 1024 / 1024
                    logger.info(f"   VRAM allocated: {mem_mb:.0f}MB")
                    logger.info(f"   Context length: {config.context_length}")

            else:
                raise ValueError(f"Unknown model type: {config.type}")
                
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")
            raise
            
        # Track the model
        self.models[name] = model
        self.last_used[name] = time.time()
        
        # Update VRAM tracking
        used_mb, _ = self.get_vram_usage_mb()
        self.vram_usage[name] = config.estimated_vram_mb
        
        logger.info(f"Model {name} loaded. VRAM used: {used_mb}MB / {self.max_vram_mb}MB")
        
        return model
        
    def get_model(self, name: str) -> Any:
        """
        Get a model (load if necessary).
        
        Args:
            name: Model name
            
        Returns:
            Model instance
        """
        if name not in self.configs:
            raise ValueError(f"Unknown model: {name}")
            
        # Update last used time
        self.last_used[name] = time.time()
        
        # Return cached model or load it
        if name in self.models:
            return self.models[name]
        else:
            return self._load_model(name)
            
    def preload_priority_models(self):
        """Preload high priority models."""
        priority_models = [
            name for name, config in self.configs.items()
            if config.priority >= 8
        ]
        
        for name in priority_models:
            try:
                logger.info(f"Preloading high-priority model: {name}")
                self.get_model(name)
            except Exception as e:
                logger.warning(f"Could not preload {name}: {e}")
                
    def _get_model_layer_count(self, model) -> Optional[int]:
        """
        Get the number of transformer layers in a SentenceTransformer model.

        Args:
            model: SentenceTransformer model

        Returns:
            Number of layers, or None if not determinable
        """
        try:
            # SentenceTransformer structure: model[0] is the Transformer module
            if hasattr(model, '__getitem__') and hasattr(model[0], 'auto_model'):
                auto_model = model[0].auto_model

                # Try different encoder structures
                if hasattr(auto_model, 'encoder') and hasattr(auto_model.encoder, 'layer'):
                    return len(auto_model.encoder.layer)
                elif hasattr(auto_model, 'layers'):
                    return len(auto_model.layers)
                elif hasattr(auto_model, 'transformer') and hasattr(auto_model.transformer, 'layer'):
                    return len(auto_model.transformer.layer)

            return None
        except Exception as e:
            logger.warning(f"Could not determine layer count: {e}")
            return None

    def _truncate_model_layers(self, model, num_layers: int):
        """
        Truncate a SentenceTransformer model to use fewer transformer layers.
        This enables 2D Matryoshka speedup for models trained with AdaptiveLayerLoss.

        Args:
            model: SentenceTransformer model
            num_layers: Number of layers to keep
        """
        try:
            if hasattr(model, '__getitem__') and hasattr(model[0], 'auto_model'):
                auto_model = model[0].auto_model

                # Try different encoder structures (BERT, XLM-R, etc.)
                if hasattr(auto_model, 'encoder') and hasattr(auto_model.encoder, 'layer'):
                    original = len(auto_model.encoder.layer)
                    auto_model.encoder.layer = auto_model.encoder.layer[:num_layers]
                    logger.info(f"Truncated encoder.layer: {original} -> {num_layers}")
                    return

                elif hasattr(auto_model, 'layers'):
                    original = len(auto_model.layers)
                    auto_model.layers = auto_model.layers[:num_layers]
                    logger.info(f"Truncated layers: {original} -> {num_layers}")
                    return

                elif hasattr(auto_model, 'transformer') and hasattr(auto_model.transformer, 'layer'):
                    original = len(auto_model.transformer.layer)
                    auto_model.transformer.layer = auto_model.transformer.layer[:num_layers]
                    logger.info(f"Truncated transformer.layer: {original} -> {num_layers}")
                    return

            logger.warning("Could not truncate model layers - unknown architecture")

        except Exception as e:
            logger.error(f"Failed to truncate model layers: {e}")

    def cleanup_inactive_models(self, timeout_seconds: int = None):
        """
        Unload models that haven't been used for timeout_seconds.
        
        Args:
            timeout_seconds: Seconds of inactivity before unloading (uses AUTO_UNLOAD_MODELS_TIME env var, default 180)
        """
        if timeout_seconds is None:
            timeout_seconds = int(os.getenv('AUTO_UNLOAD_MODELS_TIME', '180'))
        
        current_time = time.time()
        logger.info(f"Checking for inactive models... (Timeout: {timeout_seconds}s)")
        
        # Log status of all models
        for name in list(self.models.keys()):
            last = self.last_used.get(name, 0)
            inactive = current_time - last
            logger.info(f"  - {name}: inactive for {inactive:.1f}s (Threshold: {timeout_seconds}s)")

        models_to_unload = []
        # Find inactive models
        for name in list(self.models.keys()):
            last_used = self.last_used.get(name, 0)
            inactive_time = current_time - last_used
            
            if inactive_time > timeout_seconds:
                models_to_unload.append((name, inactive_time))
        
        # Log what we found
        if models_to_unload:
            logger.info(f"Found {len(models_to_unload)} models to unload: {[name for name, _ in models_to_unload]}")
        else:
            logger.info(f"No models to unload. Currently loaded: {list(self.models.keys())}")
        
        # Unload inactive models
        for name, inactive_time in models_to_unload:
            logger.info(f"Auto-unloading {name} after {inactive_time:.0f}s of inactivity")
            self._unload_model(name)
        
        # Log final status
        if models_to_unload:
            used_mb, free_mb = self.get_vram_usage_mb()
            logger.info(f"Cleanup complete. VRAM: {used_mb}MB used, {free_mb}MB free")
    
    def get_status(self) -> Dict:
        """
        Get status of model manager.
        
        Returns:
            Dictionary with status info
        """
        used_mb, free_mb = self.get_vram_usage_mb()
        
        loaded_models = {}
        for name in self.models:
            config = self.configs.get(name)
            loaded_models[name] = {
                "priority": config.priority if config else 0,
                "estimated_vram_mb": self.vram_usage.get(name, 0),
                "last_used": time.time() - self.last_used.get(name, 0)
            }
            
        return {
            "vram_used_mb": used_mb,
            "vram_free_mb": free_mb,
            "vram_limit_mb": self.max_vram_mb,
            "loaded_models": loaded_models,
            "available_models": list(self.configs.keys())
        }
        
        
# Global instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get global ModelManager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager