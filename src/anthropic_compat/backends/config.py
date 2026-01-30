"""
Backend configuration management.

Handles environment variable parsing and configuration
for different inference backends.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BackendType(Enum):
    """Available backend types."""

    HUGGINGFACE = "huggingface"
    VLLM_LOCAL = "vllm_local"
    VLLM_REMOTE = "vllm_remote"
    REMOTE_OPENAI = "remote_openai"
    BITNET = "bitnet"

    @classmethod
    def from_string(cls, value: str) -> "BackendType":
        """Parse backend type from string."""
        value = value.lower().strip()
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Unknown backend type: {value}. Valid options: {[m.value for m in cls]}")


class QuantizationType(Enum):
    """Quantization methods supported."""

    NONE = "none"
    BITSANDBYTES = "bitsandbytes"  # NF4/INT8
    AWQ = "awq"
    GPTQ = "gptq"
    MARLIN = "marlin"
    FP8 = "fp8"


@dataclass
class HuggingFaceConfig:
    """Configuration for HuggingFace backend."""

    device: str = "cuda"
    torch_dtype: str = "auto"
    quantization: QuantizationType = QuantizationType.BITSANDBYTES
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    use_flash_attention: bool = True
    trust_remote_code: bool = True

    @classmethod
    def from_env(cls) -> "HuggingFaceConfig":
        """Load configuration from environment variables."""
        return cls(
            device=os.getenv("HF_DEVICE", "cuda"),
            torch_dtype=os.getenv("HF_TORCH_DTYPE", "auto"),
            quantization=QuantizationType(os.getenv("HF_QUANTIZATION", "bitsandbytes")),
            load_in_4bit=os.getenv("HF_LOAD_4BIT", "true").lower() == "true",
            load_in_8bit=os.getenv("HF_LOAD_8BIT", "false").lower() == "true",
            use_flash_attention=os.getenv("HF_FLASH_ATTENTION", "true").lower() == "true",
            trust_remote_code=os.getenv("HF_TRUST_REMOTE_CODE", "true").lower() == "true",
        )


@dataclass
class VLLMLocalConfig:
    """Configuration for local vLLM backend."""

    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: Optional[int] = None
    quantization: Optional[str] = None  # "bitsandbytes", "awq", "gptq", "marlin"
    dtype: str = "auto"  # "auto", "half", "float16", "bfloat16"
    enforce_eager: bool = False  # Disable CUDA graphs for debugging
    enable_prefix_caching: bool = True
    max_num_seqs: int = 256  # Maximum concurrent sequences

    @classmethod
    def from_env(cls) -> "VLLMLocalConfig":
        """Load configuration from environment variables."""
        return cls(
            tensor_parallel_size=int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")),
            gpu_memory_utilization=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90")),
            max_model_len=int(os.getenv("VLLM_MAX_MODEL_LEN")) if os.getenv("VLLM_MAX_MODEL_LEN") else None,
            quantization=os.getenv("VLLM_QUANTIZATION"),
            dtype=os.getenv("VLLM_DTYPE", "auto"),
            enforce_eager=os.getenv("VLLM_ENFORCE_EAGER", "false").lower() == "true",
            enable_prefix_caching=os.getenv("VLLM_PREFIX_CACHING", "true").lower() == "true",
            max_num_seqs=int(os.getenv("VLLM_MAX_NUM_SEQS", "256")),
        )


@dataclass
class VLLMRemoteConfig:
    """Configuration for remote vLLM server."""

    base_url: str = "http://localhost:8001/v1"
    api_key: Optional[str] = None
    timeout: float = 120.0
    max_retries: int = 3
    retry_delay: float = 1.0

    @classmethod
    def from_env(cls) -> "VLLMRemoteConfig":
        """Load configuration from environment variables."""
        return cls(
            base_url=os.getenv("VLLM_REMOTE_URL", "http://localhost:8001/v1"),
            api_key=os.getenv("VLLM_REMOTE_API_KEY"),
            timeout=float(os.getenv("VLLM_REMOTE_TIMEOUT", "120.0")),
            max_retries=int(os.getenv("VLLM_REMOTE_MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("VLLM_REMOTE_RETRY_DELAY", "1.0")),
        )


@dataclass
class RemoteOpenAIConfig:
    """Configuration for OpenAI-compatible remote endpoints."""

    base_url: str = "https://api.openai.com/v1"
    api_key: Optional[str] = None
    organization: Optional[str] = None
    timeout: float = 120.0
    max_retries: int = 3
    model_override: Optional[str] = None  # Override model name in requests

    @classmethod
    def from_env(cls) -> "RemoteOpenAIConfig":
        """Load configuration from environment variables."""
        return cls(
            base_url=os.getenv("REMOTE_OPENAI_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("REMOTE_OPENAI_API_KEY"),
            organization=os.getenv("REMOTE_OPENAI_ORG"),
            timeout=float(os.getenv("REMOTE_OPENAI_TIMEOUT", "120.0")),
            max_retries=int(os.getenv("REMOTE_OPENAI_MAX_RETRIES", "3")),
            model_override=os.getenv("REMOTE_OPENAI_MODEL"),
        )


@dataclass
class BitNetConfig:
    """Configuration for BitNet backend (CPU-only 1-bit quantization)."""

    bitnet_path: str = "/app/BitNet"  # Path to BitNet installation
    model_path: str = "/app/models/bitnet/model.gguf"  # Path to GGUF model
    threads: int = 4  # Number of CPU threads
    context_length: int = 2048  # Maximum context length
    timeout: float = 120.0  # Generation timeout in seconds

    @classmethod
    def from_env(cls) -> "BitNetConfig":
        """Load configuration from environment variables."""
        return cls(
            bitnet_path=os.getenv("BITNET_PATH", "/app/BitNet"),
            model_path=os.getenv("BITNET_MODEL_PATH", "/app/models/bitnet/model.gguf"),
            threads=int(os.getenv("BITNET_THREADS", "4")),
            context_length=int(os.getenv("BITNET_CONTEXT_LENGTH", "2048")),
            timeout=float(os.getenv("BITNET_TIMEOUT", "120.0")),
        )


@dataclass
class BackendConfig:
    """Master configuration for inference backends."""

    backend_type: BackendType = BackendType.HUGGINGFACE

    # Backend-specific configs
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    vllm_local: VLLMLocalConfig = field(default_factory=VLLMLocalConfig)
    vllm_remote: VLLMRemoteConfig = field(default_factory=VLLMRemoteConfig)
    remote_openai: RemoteOpenAIConfig = field(default_factory=RemoteOpenAIConfig)
    bitnet: BitNetConfig = field(default_factory=BitNetConfig)

    @classmethod
    def from_env(cls) -> "BackendConfig":
        """Load full configuration from environment variables."""
        backend_type_str = os.getenv("LLM_BACKEND", "huggingface")

        try:
            backend_type = BackendType.from_string(backend_type_str)
        except ValueError:
            backend_type = cls._auto_detect_backend()

        return cls(
            backend_type=backend_type,
            huggingface=HuggingFaceConfig.from_env(),
            vllm_local=VLLMLocalConfig.from_env(),
            vllm_remote=VLLMRemoteConfig.from_env(),
            remote_openai=RemoteOpenAIConfig.from_env(),
            bitnet=BitNetConfig.from_env(),
        )

    @staticmethod
    def _auto_detect_backend() -> BackendType:
        """Auto-detect the best available backend."""
        # Check for vLLM
        try:
            import vllm
            return BackendType.VLLM_LOCAL
        except ImportError:
            pass

        # Check for remote vLLM
        if os.getenv("VLLM_REMOTE_URL"):
            return BackendType.VLLM_REMOTE

        # Check for remote OpenAI-compatible
        if os.getenv("REMOTE_OPENAI_URL"):
            return BackendType.REMOTE_OPENAI

        # Check for BitNet (CPU-only)
        if os.getenv("BITNET_PATH") or os.getenv("BITNET_MODEL_PATH"):
            return BackendType.BITNET

        # Default to HuggingFace
        return BackendType.HUGGINGFACE

    def get_active_config(self):
        """Get the configuration for the active backend type."""
        mapping = {
            BackendType.HUGGINGFACE: self.huggingface,
            BackendType.VLLM_LOCAL: self.vllm_local,
            BackendType.VLLM_REMOTE: self.vllm_remote,
            BackendType.REMOTE_OPENAI: self.remote_openai,
            BackendType.BITNET: self.bitnet,
        }
        return mapping[self.backend_type]
