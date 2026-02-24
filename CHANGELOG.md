# Changelog

All notable changes to Deposium Embeddings TurboV2 will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Additional embedding model benchmarks
- Multi-GPU support

---

## [3.2.0] - 2026-02-24

### Added
- **LLM Inference (Anthropic-compatible API)**
  - `POST /v1/messages` - Chat completion (streaming, tool calling)
  - `GET /v1/models` - List available LLM models
  - Backends: HuggingFace, vLLM, BitNet (CPU), Remote OpenAI
  - Models: qwen2.5-coder-7b/3b/1.5b, bitnet-700m/2b/llama3-8b

- **BitNet CPU Backend** - 1-bit quantized inference without GPU
  - Microsoft BitNet integration via subprocess
  - ~500MB RAM for 2.4B model, 10-20 tok/s
  - Ideal for Railway/edge deployments

- **mxbai-rerank-xsmall** - Lightweight DeBERTa V1 reranker (~200MB)

### Fixed
- **CRITICAL: CUDA memory leak** - VRAM grew from 358MB to 5807MB over 29h
  - Root cause: inference tensors never freed between requests
  - Added `torch.inference_mode()` to vision, classification, reranking routes
  - Added explicit `del inputs, outputs` + `gc.collect()` + `torch.cuda.empty_cache()`
  - Fixed 7 leak points across 4 files (vision.py, classification.py, reranking.py, huggingface.py)

- **Security hardening**
  - Restrict `trust_remote_code` to whitelisted models only
  - Add input size limits (max texts, max document count, max base64 size)
  - Sanitize error messages (no internal paths leaked to clients)
  - Add `torch.inference_mode()` to all remaining encode() paths

### Changed
- README updated with LLM backends, authentication docs, /v1/messages endpoint
- .gitignore updated to exclude internal docs from publication

---

## [3.1.0] - 2026-01-27

### Added
- **Audio Transcription (Whisper)** - New audio processing module
  - `faster-whisper` integration (4x faster than OpenAI Whisper)
  - Models: whisper-tiny, whisper-base (default), whisper-small, whisper-medium
  - CPU/CUDA inference with INT8 quantization
  - Automatic language detection (100+ languages)
  - Word-level timestamps support
  - Voice Activity Detection (VAD)
  - Translation to English

- **Audio API Endpoints**
  - `POST /api/transcribe` - Transcribe audio files (multipart upload)
  - `POST /api/transcribe/base64` - Transcribe from base64 encoded audio
  - `POST /api/audio/embed` - Audio embedding pipeline (transcribe + embed)

- **WhisperHandler** - Standalone audio processing handler
  - Lazy model loading with caching
  - Model hot-swapping between sizes
  - Memory cleanup on unload

### Changed
- Bumped to v13.0.0 (minor: new audio feature)
- Updated README with audio transcription documentation
- Added faster-whisper to requirements.txt

---

## [3.0.0] - 2025-01-11

### Added
- **mxbai-rerank-v2** - SOTA cross-encoder reranker
  - BEIR 55.57, 100+ languages support
  - Native `mxbai-rerank` library integration
  - 4-bit NF4 quantization (1GB -> ~250MB VRAM)

- **mxbai-embed-2d** - 2D Matryoshka embeddings
  - Full: 24 layers, 1024D (maximum quality)
  - Fast: 12 layers, 768D (~2x speedup, -15% quality)
  - Turbo: 6 layers, 512D (~4x speedup, -20% quality)
  - Adaptive layer truncation for speed/quality tradeoff

- **LFM2.5-VL-1.6B** - Vision-Language model
  - Document OCR and text extraction
  - Edge-first CPU design (1.6B params)
  - OCRBench v2: 41.44% accuracy
  - `/api/vision` endpoint for document processing

### Fixed
- **mxbai-rerank-v2 RankResult** - Fixed attribute access
  - Changed `.get()` dict access to proper `.index`, `.document`, `.score` attributes
  - Fixes 500 error on reranking requests

### Changed
- Bumped to v3.0.0 (major: new models, API changes)
- Updated README with comprehensive model documentation

---

## [2.0.0] - 2025-12-21

### Added
- **OpenBench API** - LLM benchmarking endpoints
  - `GET /api/benchmarks` - List available benchmarks
  - `POST /api/benchmarks/run` - Run standardized benchmarks
  - `POST /api/benchmarks/corpus-eval` - Custom corpus evaluation

### Changed
- **uv Build Optimization**
  - Build time reduced from 3-5min to 30-60s
  - Multi-platform wheel caching
  - Improved layer caching strategy

### Fixed
- Numpy 2.0 compatibility (pinned to 1.26.4)
- ONNX Runtime version alignment

---

## [1.5.0] - 2025-11-16

### Added
- **Qwen3 Rerank Model**
  - `qwen3-rerank` - High-quality reranking (242ms, FP32)
  - Float16 optimization (~70% VRAM reduction)

- **VL Classifier** - Document complexity routing
  - ResNet18 distilled from CLIP ViT-B/32
  - 100% accuracy, 100% HIGH recall
  - 11MB ONNX INT8 (97% smaller than CLIP)
  - ~10ms latency (2.5x faster)
  - Binary routing: LOW->OCR, HIGH->VLM

### Changed
- Circuit breaker pattern for external calls
- Fallback to Ollama when TurboV2 unavailable

---

## [1.0.0] - 2025-09-15

### Added
- **Initial Release**
  - Embedding models:
    - m2v-bge-m3-1024d (PRIMARY)
    - bge-m3-onnx (CPU fallback)
    - gemma-768d (LEGACY multilingual)
    - qwen3-embed
  - REST API endpoints:
    - `/api/embed` - Generate embeddings
    - `/api/rerank` - Rerank documents
    - `/api/classify` - Document classification
  - Docker deployment
  - N8N integration support
  - Ollama-compatible API

---

## Model Summary

| Model | Version Added | Type | Status |
|-------|--------------|------|--------|
| m2v-bge-m3-1024d | 1.0.0 | Embedding | PRIMARY |
| bge-m3-onnx | 1.0.0 | Embedding | Active |
| gemma-768d | 1.0.0 | Embedding | LEGACY |
| qwen3-embed | 1.0.0 | Embedding | Active |
| qwen3-rerank | 1.5.0 | Reranking | Active |
| vl-classifier | 1.5.0 | Classification | Active |
| mxbai-embed-2d | 3.0.0 | Embedding | Active |
| mxbai-embed-2d-fast | 3.0.0 | Embedding | Active |
| mxbai-embed-2d-turbo | 3.0.0 | Embedding | Active |
| mxbai-rerank-v2 | 3.0.0 | Reranking | RECOMMENDED |
| lfm25-vl | 3.0.0 | Vision-Language | Active |

---

## Links

- [README](README.md) - API documentation
- [DEPLOYMENT_SUCCESS](DEPLOYMENT_SUCCESS.md) - VL Classifier deployment notes
