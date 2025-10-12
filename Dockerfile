FROM python:3.11-slim

# Install dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/

# Note: Models downloaded from Hugging Face at startup:
# - Gemma-768D Model2Vec (PRIMARY) ~400MB - from theseedship/gemma-deposium-768d
# - int8 reranker ~30MB - from C10X/int8
# Total download: ~430MB (cached on Railway between deployments)

# Create non-root user for security
RUN useradd -m -u 1001 -s /bin/bash appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 11435

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD ["python", "-c", "import requests; requests.get('http://localhost:11435/health', timeout=5)"]

# Run FastAPI
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "11435"]
