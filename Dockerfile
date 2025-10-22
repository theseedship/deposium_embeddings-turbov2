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

# Note: Complexity classifier model is included in the Docker image:
# - ResNet18 ONNX INT8 quantized model (11MB) at src/models/complexity_classifier/model_quantized.onnx
# - Binary classification: LOW (simple docs → OCR) vs HIGH (complex docs → VLM)
# - Performance: 93% accuracy, 100% HIGH recall, ~10ms latency

# Note: Embedding models will be downloaded from Hugging Face at startup:
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

# Expose port (Railway will override this with its own PORT)
EXPOSE 11435

# Add healthcheck using the PORT environment variable
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests, os; requests.get(f'http://localhost:{os.environ.get(\"PORT\", \"11435\")}/health', timeout=5)"

# Run FastAPI with PORT from environment (Railway auto-injects PORT)
# Force 0.0.0.0 binding for Railway IPv4+IPv6 dual-stack support
# Railway's internal networking requires IPv4 compatibility for external access
CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-11435}
