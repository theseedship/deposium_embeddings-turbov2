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

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    type: str  # "model2vec", "sentence_transformer"
    path: Optional[str] = None  # Local path
    hub_id: Optional[str] = None  # HuggingFace ID
    priority: int = 0  # Higher = kept in memory longer
    estimated_vram_mb: int = 500  # Estimated VRAM usage
    device: str = "cuda"  # Device to load on


class ModelManager:
    """
    Manages models with dynamic VRAM allocation.
    
    Features:
    - Lazy loading (models loaded on first use)
    - VRAM monitoring and limits
    - LRU cache with priority system
    - Automatic model unloading when VRAM limit exceeded
    """
    
    def __init__(self, max_vram_mb: int = 5000):
        """
        Initialize model manager.
        
        Args:
            max_vram_mb: Maximum VRAM to use in MB (default 5GB for 6GB GPU)
        """
        self.max_vram_mb = max_vram_mb
        self.models: OrderedDict[str, Any] = OrderedDict()
        self.configs: Dict[str, ModelConfig] = {}
        self.last_used: Dict[str, float] = {}
        self.vram_usage: Dict[str, int] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Register available models
        self._register_models()
        
    def _register_models(self):
        """Register all available models with their configurations."""
        
        # Qwen25-1024D (PRIMARY - highest priority)
        self.configs["qwen25-1024d"] = ModelConfig(
            name="qwen25-1024d",
            type="model2vec",
            path="models/qwen25-deposium-1024d",
            hub_id="tss-deposium/qwen25-deposium-1024d",
            priority=1,  # Equal priority for all models
            estimated_vram_mb=500,
            device=self.device
        )
        
        # Gemma-768D (SECONDARY)
        self.configs["gemma-768d"] = ModelConfig(
            name="gemma-768d", 
            type="model2vec",
            path="models/gemma-deposium-768d",
            hub_id="tss-deposium/gemma-deposium-768d",
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
            hub_id="Qwen/Qwen3-Embedding-0.6B",
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
        
    def get_vram_usage_mb(self) -> Tuple[int, int]:
        """
        Get current VRAM usage.
        
        Returns:
            (used_mb, free_mb)
        """
        if not torch.cuda.is_available():
            return (0, 0)
            
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
        except:
            pass
            
        # Fallback to PyTorch
        try:
            used = torch.cuda.memory_allocated() // (1024 * 1024)
            reserved = torch.cuda.memory_reserved() // (1024 * 1024)
            # Estimate based on reserved memory
            return (reserved, self.max_vram_mb - reserved)
        except:
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
                del model
            elif hasattr(model, 'cpu'):
                logger.info(f"Moving {name} to CPU")
                model.cpu()
            elif hasattr(model, 'to'):
                logger.info(f"Moving {name} to CPU via .to()")
                model.to('cpu')
            else:
                # ONNX models or other types - just delete
                logger.info(f"Force deleting model: {name}")
                del model
        except ImportError:
            # Fallback if Model2Vec not available
            if hasattr(model, 'cpu'):
                model.cpu()
            elif hasattr(model, 'to'):
                model.to('cpu')
            
        # Remove from cache
        del self.models[name]
        if name in self.vram_usage:
            del self.vram_usage[name]
            
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
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
                                trust_remote_code=True,
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
                            trust_remote_code=True,
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
                        trust_remote_code=True,
                        device=config.device
                    )
                    logger.info(f"✅ {name} loaded (SentenceTransformer)")

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
                except:
                    pass
                
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
                
    def cleanup_inactive_models(self, timeout_seconds: int = None):
        """
        Unload models that haven't been used for timeout_seconds.
        
        Args:
            timeout_seconds: Seconds of inactivity before unloading (uses AUTO_UNLOAD_MODELS_TIME env var, default 180)
        """
        if timeout_seconds is None:
            timeout_seconds = int(os.getenv('AUTO_UNLOAD_MODELS_TIME', '180'))
        
        current_time = time.time()
        logger.info(f"Running model cleanup with {timeout_seconds}s timeout (AUTO_UNLOAD_MODELS_TIME={os.getenv('AUTO_UNLOAD_MODELS_TIME', 'default')})")
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
            logger.debug(f"No models to unload. Currently loaded: {list(self.models.keys())}")
        
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