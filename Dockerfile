FROM python:3.11-slim

# Install dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/

# Expose port
EXPOSE 11435

# Run FastAPI
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "11435"]
