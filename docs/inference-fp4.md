# Inference Backends and FP4/NF4 Quantization Analysis

This document provides a comprehensive analysis of the inference backend options available in this repository, including quantization strategies for efficient GPU memory usage.

## Overview

The Anthropic-compatible API (`/v1/messages`) supports multiple inference backends:

| Backend | Use Case | Performance | Requirements |
|---------|----------|-------------|--------------|
| HuggingFace | Default, simple setup | 1x baseline | Any GPU |
| vLLM Local | High throughput | 24x faster | CUDA 12.1+, 8GB+ VRAM |
| vLLM Remote | Distributed inference | 24x faster | External vLLM server |
| Remote OpenAI | Cloud/third-party | Varies | Network access |

## Architecture

```
                    FastAPI Application (main.py)
                    /v1/messages  /api/embed  /api/rerank
                              │
                    InferenceBackendFactory
                    (config, env, auto-detect)
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    HuggingFace         vLLM Local          Remote API
    (current)            (new)               (new)
    ├────────────┤    ├────────────┤    ├────────────┤
    │ BNB NF4    │    │ PagedAttn  │    │ OpenAI-    │
    │ 1 GPU      │    │ AWQ/GPTQ   │    │ compatible │
    │ Streaming  │    │ Multi-GPU  │    │ Cloud GPU  │
    └────────────┘    └────────────┘    └────────────┘
```

## Backend Comparison

### Performance Metrics

| Metric | HuggingFace | vLLM |
|--------|-------------|------|
| Throughput | 1x baseline | **24x** |
| KV-Cache waste | 60-80% | **<4%** |
| Batching | None | **Continuous** |
| Memory efficiency | Good | **Excellent** |

### vLLM Advantages

1. **PagedAttention**: Revolutionary memory management that treats KV-cache like virtual memory
   - Traditional transformers waste 60-80% of allocated memory
   - vLLM wastes less than 4% through paging

2. **Continuous Batching**: Dynamically adjusts batch composition
   - New requests join mid-batch
   - Completed requests immediately free resources
   - 24x throughput improvement in real-world scenarios

3. **Quantization Support**: Multiple quantization methods
   - AWQ, GPTQ, bitsandbytes
   - Marlin kernels for faster inference
   - FP8 on Hopper/Ada GPUs

## Quantization Analysis

### NF4 vs FP4

| Format | Hardware | VRAM Reduction | Quality | Status |
|--------|----------|----------------|---------|--------|
| **NF4** (Normal Float 4) | All GPUs | 3.5x | Excellent | Recommended |
| FP4 (Float Point 4) | Blackwell only | 3.5x | Good | Future |
| AWQ | All GPUs | 3-4x | Very good | Available |
| GPTQ | All GPUs | 3-4x | Very good | Available |

### Why NF4 for Transformers

NF4 (Normal Float 4-bit) is specifically designed for the weight distribution in neural networks:

1. **Information-theoretic optimality**: NF4 quantization levels are placed where weights are most dense
2. **Zero-centered**: Transformer weights cluster around zero - NF4 captures this
3. **Double quantization**: Quantizes the quantization constants for additional savings

```python
# NF4 configuration (current default)
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # ~0.4 bits/weight savings
    bnb_4bit_quant_type="nf4",       # Optimal for transformers
    bnb_4bit_compute_dtype=torch.float16
)
```

### Memory Estimates (Qwen2.5-Coder)

| Model | FP16 | 4-bit (NF4/AWQ) | Reduction |
|-------|------|-----------------|-----------|
| 7B | ~14GB | ~4.5GB | 3.1x |
| 3B | ~6GB | ~2GB | 3x |
| 1.5B | ~3GB | ~1.2GB | 2.5x |

## Configuration

### Environment Variables

```bash
# Backend selection (auto-detect if not set)
LLM_BACKEND=huggingface|vllm_local|vllm_remote|remote_openai

# HuggingFace options
HF_DEVICE=cuda
HF_QUANTIZATION=bitsandbytes
HF_LOAD_4BIT=true
HF_FLASH_ATTENTION=true

# vLLM Local options
VLLM_TENSOR_PARALLEL_SIZE=1      # Multi-GPU: 2, 4, 8
VLLM_GPU_MEMORY_UTILIZATION=0.90 # Use 90% of GPU memory
VLLM_QUANTIZATION=bitsandbytes   # or awq, gptq, marlin
VLLM_MAX_MODEL_LEN=32768         # Context length
VLLM_PREFIX_CACHING=true         # Cache repeated prompts

# vLLM Remote options
VLLM_REMOTE_URL=http://gpu-server:8001/v1
VLLM_REMOTE_API_KEY=your-key
VLLM_REMOTE_TIMEOUT=120

# Remote OpenAI-compatible options
REMOTE_OPENAI_URL=https://api.cloud-provider.com/v1
REMOTE_OPENAI_API_KEY=your-key
REMOTE_OPENAI_MODEL=model-name   # Override model name
```

### Per-Model Backend Configuration

Models can specify their preferred backend in `model_manager.py`:

```python
ModelConfig(
    name="qwen2.5-coder-7b",
    type="causal_lm",
    hub_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    backend_type="vllm_local",  # Override default backend
    quantize_4bit=True,
    context_length=32768,
)
```

## Usage Examples

### Default HuggingFace Backend

```bash
# No configuration needed - uses NF4 quantization by default
uvicorn src.main:app --host 0.0.0.0 --port 8000

# Test
curl -X POST http://localhost:8000/v1/messages \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder-7b",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### vLLM Local Backend

```bash
# Install vLLM first
pip install vllm>=0.6.0

# Run with vLLM
LLM_BACKEND=vllm_local \
VLLM_GPU_MEMORY_UTILIZATION=0.90 \
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### Remote vLLM Server

```bash
# On GPU server - start vLLM
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --port 8001

# On API server - connect to remote
VLLM_REMOTE_URL=http://gpu-server:8001/v1 \
LLM_BACKEND=vllm_remote \
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## Known Issues

### vLLM Tool Calling Parser

**Issue**: vLLM issue #29192 - Qwen2.5-Coder tool_calling parser is broken in vLLM.

**Mitigation**: This repository uses a custom `tool_calling.py` parser that works correctly with all backends. The custom parser:
- Extracts `<tool_call>` XML blocks from model output
- Parses JSON tool calls
- Works with streaming responses

### vLLM Installation

vLLM has specific requirements:
- Linux only (no Windows/macOS support)
- CUDA 12.1 or higher
- 8GB+ VRAM recommended
- PyTorch 2.1+ with CUDA support

For installation issues, see: https://docs.vllm.ai/en/latest/getting_started/installation.html

## Backend Selection Guide

| Scenario | Recommended Backend | Reason |
|----------|---------------------|--------|
| Single GPU, simple setup | HuggingFace | Easy, stable, NF4 quantization |
| High throughput needed | vLLM Local | 24x faster, continuous batching |
| Multiple GPUs | vLLM Local | Tensor parallelism |
| Separate GPU server | vLLM Remote | Dedicated inference |
| Cloud deployment | Remote OpenAI | Use cloud GPU providers |
| Edge/low memory | HuggingFace + NF4 | Minimal footprint |

## Files Structure

```
src/anthropic_compat/backends/
├── __init__.py          # Factory + registry (create_backend)
├── base.py              # Abstract interface (InferenceBackend)
├── config.py            # Configuration dataclasses
├── huggingface.py       # HuggingFace Transformers backend
├── vllm_local.py        # vLLM local inference
└── remote_openai.py     # OpenAI-compatible remote APIs
```

## API Reference

### InferenceBackend Interface

```python
class InferenceBackend(ABC):
    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, Any]],
        config: GenerationConfig,
    ) -> GenerationResult:
        """Synchronous generation."""
        pass

    @abstractmethod
    def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        config: GenerationConfig,
    ) -> Generator[StreamChunk, None, None]:
        """Streaming generation."""
        pass

    @property
    def capabilities(self) -> BackendCapabilities:
        """Backend feature set."""
        pass

    def count_tokens(self, text: str) -> int:
        """Token counting."""
        pass
```

### GenerationConfig

```python
@dataclass
class GenerationConfig:
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
```

### BackendCapabilities

```python
@dataclass
class BackendCapabilities:
    streaming: bool = True
    tool_calling: bool = True
    batching: bool = False
    multi_gpu: bool = False
    quantization: List[str] = field(default_factory=list)
    max_context_length: int = 4096
    continuous_batching: bool = False
    paged_attention: bool = False
```

## Future: FP4 on Blackwell

NVIDIA Blackwell GPUs (2024+) will support native FP4 compute:

- **Hardware FP4**: Native 4-bit floating point operations
- **2x throughput**: Compared to INT4/NF4
- **No quality loss**: True floating point vs fixed-point quantization

When Blackwell GPUs become available:
1. vLLM will add FP4 support
2. Update `VLLM_QUANTIZATION=fp4`
3. Expect 2x additional speedup over current 4-bit methods

## References

- [vLLM Paper](https://arxiv.org/abs/2309.06180) - Efficient Memory Management for Large Language Model Serving
- [QLoRA Paper](https://arxiv.org/abs/2305.14314) - NF4 Quantization Analysis
- [PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) - vLLM Technical Blog
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) - 4-bit Quantization Library
