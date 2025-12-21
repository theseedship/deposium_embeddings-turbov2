FROM python:3.11-slim

# Install uv (Rust-based Python package installer - 10-100x faster than pip)
# https://github.com/astral-sh/uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Performance optimization environment variables

# PyTorch Threading Optimization
ENV OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    TORCH_NUM_THREADS=4 \
    KMP_AFFINITY=granularity=fine,compact,1,0 \
    KMP_BLOCKTIME=0

# ONNX Runtime Optimization
ENV ORT_NUM_THREADS=4 \
    ORT_ENABLE_CPU_FP16_OPS=1 \
    ORT_OPTIMIZATION_LEVEL=3

# PyTorch JIT
ENV PYTORCH_JIT=1

# Python Optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Model Configuration (defaults - can be overridden at runtime)
ENV DEFAULT_EMBEDDING_MODEL=m2v-bge-m3-1024d \
    DEFAULT_RERANK_MODEL=qwen3-rerank

# Model Cache (uses Railway volume at /app/models)
# Note: Using only HF_HOME (TRANSFORMERS_CACHE is deprecated in transformers v5)
ENV HF_HOME=/app/models

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
# --system flag installs into system python, avoiding venv overhead in Docker
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application
COPY src/ ./src/

# Note: Complexity classifier model is included in the Docker image:
# - ResNet18 ONNX INT8 quantized model (11MB) at src/models/complexity_classifier/model_quantized.onnx
# - Binary classification: LOW (simple docs → OCR) vs HIGH (complex docs → VLM)
# - Performance: 93% accuracy, 100% HIGH recall, ~10ms latency

# Note: Embedding models will be downloaded from Hugging Face at startup:
# - M2V-BGE-M3-1024D (PRIMARY) ~21MB - tss-deposium/m2v-bge-m3-1024d
# - BGE-M3-ONNX INT8 (CPU) ~150MB - gpahal/bge-m3-onnx-int8
# - Gemma-768D (LEGACY) ~400MB - tss-deposium/gemma-deposium-768d
# - Qwen3-Embedding-0.6B (RERANK) ~600MB - Qwen/Qwen3-Embedding-0.6B
# Total first download: ~170MB (cached on Railway volume between deployments)

# Create cache directory for models (Railway volume will override)
RUN mkdir -p /app/models/transformers

# Note: Running as root to allow writing to Railway volume at /app/models
# The Railway volume is mounted with root permissions and cannot be written by non-root user
# This is safe as the container is isolated by Railway's infrastructure

# Expose port (Railway will override this with its own PORT)
EXPOSE 11435

# Add healthcheck using the PORT environment variable
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests, os; requests.get(f'http://localhost:{os.environ.get(\"PORT\", \"11435\")}/health', timeout=5)"

# Run FastAPI with PORT from environment (Railway auto-injects PORT)
# Use environment variable HOST (default to 0.0.0.0) for flexibility
# Set HOST=:: in Railway environment for IPv6 support (required for .railway.internal networking)
CMD uvicorn src.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-11435}
