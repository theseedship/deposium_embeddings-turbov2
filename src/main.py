from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from model2vec import StaticModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Deposium Embeddings TurboX.v2",
    description="Ultra-fast static embeddings service compatible with Ollama API",
    version="1.0.0"
)

# Load model at startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    logger.info("Loading TurboX.v2 model...")
    model = StaticModel.from_pretrained("C10X/Qwen3-Embedding-TurboX.v2")
    logger.info("✅ Model loaded successfully!")

# Request/Response models
class EmbedRequest(BaseModel):
    model: str = "turbov2"
    input: str | List[str]

class EmbedResponse(BaseModel):
    model: str
    embeddings: List[List[float]]

@app.get("/")
async def root():
    return {
        "service": "Deposium Embeddings TurboX.v2",
        "status": "running",
        "model": "C10X/Qwen3-Embedding-TurboX.v2"
    }

@app.get("/health")
async def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

@app.get("/api/tags")
async def list_models():
    """Ollama-compatible endpoint to list models"""
    return {
        "models": [
            {
                "name": "turbov2",
                "size": 30000000,  # ~30MB
                "digest": "turbov2",
                "modified_at": "2025-10-09T00:00:00Z"
            }
        ]
    }

@app.post("/api/embed")
async def create_embedding(request: EmbedRequest):
    """Ollama-compatible embedding endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Handle both string and list inputs
        texts = [request.input] if isinstance(request.input, str) else request.input

        # Generate embeddings
        embeddings = model.encode(texts, show_progress_bar=False)

        # Convert to list format
        embeddings_list = [emb.tolist() for emb in embeddings]

        return {
            "model": request.model,
            "embeddings": embeddings_list
        }

    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/embeddings")
async def create_embedding_alt(request: EmbedRequest):
    """Alternative endpoint (some clients use /api/embeddings)"""
    return await create_embedding(request)
