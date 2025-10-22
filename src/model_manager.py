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
            priority=10,  # Highest priority - always keep in memory
            estimated_vram_mb=500,
            device=self.device
        )
        
        # Gemma-768D (SECONDARY)
        self.configs["gemma-768d"] = ModelConfig(
            name="gemma-768d", 
            type="model2vec",
            path="models/gemma-deposium-768d",
            hub_id="tss-deposium/gemma-deposium-768d",
            priority=5,  # Medium priority
            estimated_vram_mb=800,
            device=self.device
        )
        
        # EmbeddingGemma-300M (full-size, low priority)
        self.configs["embeddinggemma-300m"] = ModelConfig(
            name="embeddinggemma-300m",
            type="sentence_transformer",
            hub_id="google/embeddinggemma-300m",
            priority=2,  # Low priority
            estimated_vram_mb=1000,
            device=self.device
        )
        
        # Qwen3-Embedding-0.6B (embeddings + reranking)
        self.configs["qwen3-embed"] = ModelConfig(
            name="qwen3-embed",
            type="sentence_transformer",
            hub_id="Qwen/Qwen3-Embedding-0.6B",
            priority=7,  # High priority for reranking
            estimated_vram_mb=1500,
            device=self.device
        )
        
        # Qwen3-rerank (alias for qwen3-embed)
        self.configs["qwen3-rerank"] = ModelConfig(
            name="qwen3-rerank",
            type="alias",
            path="qwen3-embed",  # Points to qwen3-embed
            priority=8,  # High priority
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
        
        # Move to CPU or delete
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
                    
            elif config.type == "sentence_transformer":
                # Load SentenceTransformer model
                if SentenceTransformer is None:
                    raise ImportError("sentence-transformers not installed")
                    
                model = SentenceTransformer(
                    config.hub_id,
                    trust_remote_code=True,
                    device=config.device
                )
                logger.info(f"✅ {name} loaded (SentenceTransformer)")
                
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