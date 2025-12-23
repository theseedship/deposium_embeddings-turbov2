# Changelog

All notable changes to Deposium Embeddings TurboV2 will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- OpenBench Python integration
- Additional reranking models

## [2.0.0] - 2025-12-21

### Changed
- **uv Build Optimization**
  - Build time reduced from 3-5min to 30-60s
  - Multi-platform wheel caching
  - Improved layer caching strategy

### Fixed
- Numpy 2.0 compatibility (pinned to 1.26.4)
- ONNX Runtime version alignment

## [1.5.0] - 2025-11-16

### Added
- **Qwen3 Rerank Model**
  - `qwen3-rerank` - High-quality reranking (242ms, FP32)
  - 93% accuracy on internal benchmarks

- **VL Classifier**
  - Vision-language classification (~10ms)
  - Document type detection

### Changed
- Circuit breaker pattern for external calls
- Fallback to Ollama when TurboV2 unavailable

## [1.0.0] - 2025-09-15

### Added
- **Initial Release**
  - 5+ embedding models
    - m2v-bge-m3-1024d (default)
    - bge-m3-onnx
    - gemma-768d
    - qwen3-embed
    - all-MiniLM-L6-v2
  - REST API endpoints
    - `/api/embed` - Generate embeddings
    - `/api/rerank` - Rerank documents
    - `/api/classify` - Document classification
  - Docker deployment
  - N8N integration support

---

## Links

- [README](README.md) - API documentation
- [Models](models/) - Available model configurations
