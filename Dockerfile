FROM python:3.11-slim

# Install system dependencies for optimization
RUN apt-get update && apt-get install -y \
    libjemalloc2 \
    && rm -rf /var/lib/apt/lists/*

# Performance optimization environment variables
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2

# PyTorch Threading Optimization
ENV OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    TORCH_NUM_THREADS=4 \
    KMP_AFFINITY=granularity=fine,compact,1,0 \
    KMP_BLOCKTIME=0

# ONNX Runtime Optimization
ENV ORT_NUM_THREADS=4 \
    ORT_ENABLE_CPU_FP16_OPS=1

# Python Optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Model Cache (uses Railway volume at /app/models)
# Note: Using only HF_HOME (TRANSFORMERS_CACHE is deprecated in transformers v5)
ENV HF_HOME=/app/models

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/

# Note: Models will be downloaded from Hugging Face at startup:
# - Qwen25-1024D Model2Vec (PRIMARY) ~65MB - from tss-deposium/qwen25-deposium-1024d
# - Gemma-768D Model2Vec (SECONDARY) ~400MB - from tss-deposium/gemma-deposium-768d
# - EmbeddingGemma-300M (optional, full-size) ~300MB
# - Qwen3-Embedding-0.6B (optional, full-size) ~600MB
# Total first download: ~465MB (cached on Railway volume between deployments)

# Create cache directory for models (Railway volume will override)
RUN mkdir -p /app/models/transformers

# Note: Running as root to allow writing to Railway volume at /app/models
# The Railway volume is mounted with root permissions and cannot be written by non-root user
# This is safe as the container is isolated by Railway's infrastructure

# Expose port
EXPOSE 11435

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD ["python", "-c", "import requests; requests.get('http://localhost:11435/health', timeout=5)"]

# Run FastAPI
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "11435"]
