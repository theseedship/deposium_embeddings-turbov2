---
language: en
license: apache-2.0
library_name: pytorch
tags:
- embeddings
- sentence-transformers
- text-embeddings
- semantic-search
- int8-quantized
- knowledge-distillation
- leaf
- embeddinggemma
pipeline_tag: feature-extraction
base_model: google/embeddinggemma-300m
---

# LEAF Embeddings - INT8 Quantized (FAILED v1 - DO NOT USE)

**🚨 CRITICAL: This model FAILED quality evaluation - DO NOT USE for production.**

**⚠️ This is experiment v1 (512 tokens) - kept for research purposes only.**

**Status**: Training completed successfully but MTEB evaluation shows critical quality loss. This serves as a baseline for comparison with the improved v2 model (2048 tokens, better architecture) currently in development.

## Model Description

This model is a **distilled and quantized version** of [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) trained using the **LEAF (Layer-wise Early-exit Alignment Framework)** methodology. It generates 768-dimensional embeddings optimized for fast CPU inference with INT8 quantization.

### What is LEAF?

LEAF is a knowledge distillation framework that:
- **Compresses** larger embedding models into smaller, faster versions
- **Preserves** semantic quality through multi-objective training (distillation + alignment + contrastive losses)
- **Optimizes** for CPU deployment with INT8 post-training quantization

### Architecture

| Property | This Model (LEAF) | Base Model (EmbeddingGemma-300m) |
|----------|-------------------|----------------------------------|
| **Dimensions** | 768D | 768D (also 512D, 256D, 128D via Matryoshka) |
| **Parameters** | ~75M (6 layers, compressed) | 300M (full architecture) |
| **Max Tokens** | 512 | 2048 |
| **Quantization** | INT8 (441MB) | FP32 (~600MB) |
| **Inference Speed** | 695 texts/s (CPU) | ~50-100 texts/s (CPU) |

**Trade-offs**:
- ✅ **6-10x faster** inference on CPU
- ✅ **Smaller model size** (441MB vs ~600MB)
- ✅ **Lower memory** footprint
- ⚠️ **Reduced context** length (512 vs 2048 tokens)
- ⚠️ **Possible quality loss** from distillation (not yet benchmarked)

## Performance

### Inference Speed (CPU)
- **Throughput**: 695 texts/second
- **Latency**: ~1.4ms per text
- **Memory**: ~500MB RAM
- **Hardware**: Standard CPU, no GPU required

### ❌ ACTUAL QUALITY (MTEB Evaluation - FAILED)

**Evaluation Date**: 2025-10-12
**Status**: ❌ **CRITICAL FAILURE** - Model does not capture semantic relationships

| Dataset | Metric | This Model (v1) | Base Model | Quality Loss |
|---------|--------|-----------------|------------|--------------|
| **STSBenchmark** | Spearman | **0.223** | 0.81 | **-72%** ❌ |
| **STS22 English** | Spearman | **0.373** | 0.75 | **-50%** ❌ |
| **STS22 Average** | Spearman | **~0.21** | 0.65 | **-68%** ❌ |
| **Cross-lingual** | Spearman | **-0.14 to 0.12** | 0.55 | **Complete loss** ❌ |

**Detailed STS22 Results by Language**:

| Language | Spearman | Status |
|----------|----------|--------|
| 🇨🇳 Chinese | 0.499 | 🟡 Moderate (best) |
| 🇸🇦 Arabic | 0.469 | 🟡 Moderate |
| 🇮🇹 Italian | 0.435 | 🟡 Moderate |
| 🇪🇸 Spanish | 0.403 | 🟠 Poor |
| 🇬🇧 English | 0.373 | 🟠 Poor |
| 🇫🇷 French | 0.300 | 🔴 Very poor |
| 🇷🇺 Russian | 0.268 | 🔴 Very poor |
| 🇹🇷 Turkish | 0.247 | 🔴 Very poor |
| 🇩🇪 German | 0.163 | ❌ Critical |
| 🇵🇱 Polish | 0.132 | ❌ Critical |

**Cross-lingual pairs (translation tasks)**: All **FAILED** (scores 0.002 to -0.143)

**Conclusion**: This model **cannot be used for semantic search, similarity tasks, or any production use**. The embeddings do not preserve semantic meaning from the base model.

### Training Quality Analysis

**Training Metrics** (from [WandB logs](https://wandb.ai/seedship/embeddinggemma-leaf/runs/savq3l32)):

| Metric | Final Value | Status |
|--------|-------------|--------|
| Distillation Loss | 0.976 | ✅ Good - Model learned from teacher |
| Alignment Loss | 2.18 | ⚠️ Moderate - Semantic space alignment could improve |
| Training Steps | 12,500 (3 epochs) | ✅ Complete |
| Training Time | 2h10min | ✅ Efficient |
| Eval Loss | NaN | ❌ Bug in evaluation aggregation |

**Observations**:
- ✅ Training **converged smoothly** without crashes
- ✅ Distillation loss **stable and low** (0.976) - good knowledge transfer
- ⚠️ Alignment loss **moderate** (2.18) - room for improvement
- ❌ Evaluation metrics **not computed** (NaN) - needs separate MTEB evaluation
- 📊 **17 checkpoints saved** - can select best performing model

**Quality Verdict**: ❌ **FAILED** - Despite low distillation loss, the model failed to learn semantic representations.

### 🔍 Failure Analysis

**What went wrong**:

1. **Architecture Too Aggressive** ❌
   - 6 layers too small for semantic preservation (should be 12+)
   - 4x compression (300M→75M) lost critical information
   - Hidden size ratio 0.5x insufficient

2. **Insufficient Training Data** ❌
   - Only 50k samples for 100+ languages
   - Mostly English data (NLI, STS, MS MARCO)
   - No multilingual balance

3. **Misleading Distillation Loss** ⚠️
   - Low distillation loss (0.976) doesn't guarantee semantic quality
   - **High alignment loss (2.18) was the real warning sign**
   - Model learned to mimic output distribution but not semantic meaning

4. **Evaluation Bug** ❌
   - Eval loss = NaN prevented early detection of failure
   - Should have caught quality issues during training

**Lessons learned for v2**:
- ✅ Monitor **alignment loss** as primary metric (target: <1.0)
- ✅ Increase student size to 120M params (12 layers)
- ✅ Use 200k+ multilingual samples
- ✅ Implement proper eval during training (MTEB subset every 500 steps)
- ✅ Train with 2048 token context
- ✅ Curriculum learning: 512→1024→2048 tokens progressively

## Training Details

### Methodology
1. **Knowledge Distillation** from EmbeddingGemma-300m (300M → 75M params)
2. **LEAF Framework** with multi-objective training:
   - Distillation loss (0.5 weight)
   - Alignment loss (1.0 weight)
   - Contrastive loss (0.3 weight)
3. **INT8 Quantization** for CPU optimization

### Training Configuration
- **Teacher Model**: `google/embeddinggemma-300m`
- **Training Data**: 50,000 samples from:
  - `sentence-transformers/all-nli`
  - `sentence-transformers/stsb`
  - `ms_marco`
- **Validation**: 5,000 samples
- **Training Steps**: 12,500 (3 epochs)
- **Hardware**: NVIDIA RTX 4050 (6GB VRAM)
- **Training Time**: ~2h10min
- **Final Losses**:
  - Distillation: 0.976
  - Alignment: 2.18

### Student Architecture
- **Layers**: 6 (vs more in teacher)
- **Attention Heads**: 6
- **Hidden Size Ratio**: 0.5x
- **Compression Ratio**: 4x

### Training Logs
View full training metrics on [WandB](https://wandb.ai/seedship/embeddinggemma-leaf/runs/savq3l32)

## Usage

### Requirements

```bash
pip install torch>=2.6.0 transformers>=4.57.0 huggingface-hub
```

### Basic Usage

```python
import torch
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

# Download model
model_path = hf_hub_download(
    repo_id="tss-deposium/gemma300-leaf-embeddings-test",
    filename="model_quantized.pt"
)

# Load model
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
model = checkpoint['model']
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "tss-deposium/gemma300-leaf-embeddings-test"
)
model.set_tokenizer(tokenizer)

# Generate embeddings
texts = ["Hello world", "Machine learning"]
with torch.no_grad():
    embeddings = model.encode(texts, device='cpu', normalize=True)

print(embeddings.shape)  # (2, 768)
```

### API Integration

This model is deployed as part of a FastAPI service:

```python
import requests

response = requests.post(
    "https://your-api-url/api/embed",
    json={"model": "leaf", "input": "Your text here"}
)

embeddings = response.json()["embeddings"]
```

## Model Card

| Property | Value |
|----------|-------|
| **Base Model** | google/embeddinggemma-300m |
| **Framework** | LEAF (Knowledge Distillation) |
| **Model Type** | Sentence Embeddings |
| **Dimensions** | 768 |
| **Max Tokens** | 512 (reduced from 2048 for efficiency) |
| **Quantization** | INT8 |
| **PyTorch Version** | 2.6+ |
| **Language** | English (base model supports 100+ languages) |
| **Training Dataset** | 50k samples (NLI, STS, MS MARCO) |

## Files

- `model_quantized.pt` (441MB) - INT8 quantized model for CPU inference
- `model_fp32.pt` (477MB) - FP32 full precision version (optional)
- `tokenizer.json` (33MB) - Tokenizer vocabulary
- `config.json` - Model configuration
- `tokenizer_config.json` - Tokenizer settings

## Limitations

### Context Length
- **512 tokens maximum** (vs 2048 in base model)
- Longer texts will be truncated
- Consider chunking for documents >512 tokens

### Quality Trade-offs
- **Distillation**: Compressed from 300M → 75M parameters may reduce quality
- **Quantization**: INT8 quantization may introduce small accuracy loss
- **Training Data**: 50k samples may not cover all domains

### Language Support
- Primarily tested on **English**
- Base model supports 100+ languages, but distilled model not yet evaluated on multilingual tasks

### Experimental Status
- **Not production-ready**: Requires thorough evaluation
- **No MTEB scores**: Quality benchmarks pending
- **Limited testing**: More evaluation needed on downstream tasks

## ❌ DO NOT USE - Model Failed Quality Checks

**This model is NOT suitable for ANY production use cases.**

**❌ NOT suitable for**:
- ❌ **Semantic search** - Scores too low (0.22 Spearman)
- ❌ **Document similarity** - Does not capture semantic meaning
- ❌ **Text clustering** - Embeddings not semantically meaningful
- ❌ **Information retrieval** - Poor correlation with human judgments
- ❌ **Duplicate detection** - Unreliable similarity scores
- ❌ **Any production deployment** - Quality insufficient
- ❌ **Multilingual tasks** - Cross-lingual capabilities destroyed
- ❌ **Mission-critical applications** - Do not use

**✅ Only suitable for**:
- ✅ **Research purposes** - Understanding failure modes in knowledge distillation
- ✅ **Baseline comparison** - For comparing with improved v2 model
- ✅ **Educational purposes** - Learning what NOT to do in model compression

## Comparison with Base Model

| Metric | LEAF v1 (This Model) | EmbeddingGemma-300m | Quality Gap |
|--------|----------------------|---------------------|-------------|
| **Parameters** | ~75M | 300M | -75% |
| **Size (INT8/FP32)** | 441MB | ~600MB | -26% ✅ |
| **Speed (CPU)** | 695 texts/s | ~50-100 texts/s | +6-10x ✅ |
| **Context Length** | 512 | 2048 | -75% ❌ |
| **STSBenchmark** | 0.223 | 0.81 | **-72%** ❌ |
| **STS22 English** | 0.373 | 0.75 | **-50%** ❌ |
| **MTEB Score (est.)** | ~25 | 61.15 | **-59%** ❌ |
| **Latency** | ~1.4ms | ~10-20ms | -85% ✅ |

**Verdict**: **Speed improvements do NOT justify the catastrophic quality loss. Use base model instead.**

## Future Work - Version 2 (In Development)

**Based on lessons learned from this failed v1 experiment, we are developing v2 with:**

### Architecture Improvements
- ✅ **12 layers** (vs 6 in v1) - 2x deeper for semantic preservation
- ✅ **120M parameters** (vs 75M) - Less aggressive compression (2.5x vs 4x)
- ✅ **2048 token context** (vs 512) - Full context length like base model
- ✅ **Hidden size ratio 0.75** (vs 0.5) - Better capacity

### Training Improvements
- ✅ **200k samples** (vs 50k) - 4x more data
- ✅ **Multilingual balanced** - 100+ languages with proper distribution
- ✅ **Curriculum learning** - Progressive 512→1024→2048 tokens
- ✅ **10 epochs** (vs 3) - More training time
- ✅ **Alignment loss priority** - Weight 2.5 (vs 1.0) + triplet loss

### Evaluation Improvements
- ✅ **Eval every 500 steps** - Early detection of quality issues
- ✅ **MTEB subset validation** - STSBenchmark during training
- ✅ **Alignment loss < 1.0 target** - Primary quality metric
- ✅ **Early stopping** - On alignment loss, not distillation loss

### Quality Targets (v2)
- 🎯 **STSBenchmark**: 0.70+ Spearman (vs 0.22 in v1)
- 🎯 **STS22 Average**: 0.50+ Spearman (vs 0.21 in v1)
- 🎯 **MTEB Score**: 55+ (vs ~25 estimated in v1)
- 🎯 **Cross-lingual**: 0.30+ (vs -0.14 in v1)

**Expected release**: After full training and validation (~12-15 hours on RTX 4050)

## Citation

```bibtex
@misc{leaf-embeddings-test,
  author = {TSS Deposium},
  title = {LEAF Embeddings INT8 - Distilled from EmbeddingGemma-300m},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/tss-deposium/gemma300-leaf-embeddings-test}},
  note = {Based on google/embeddinggemma-300m}
}

@misc{embeddinggemma,
  author = {Google},
  title = {EmbeddingGemma-300m},
  year = {2024},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/google/embeddinggemma-300m}}
}
```

## Acknowledgments

- **Base Model**: [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)
- **Training Framework**: Custom LEAF implementation
- **Datasets**: Sentence Transformers, MS MARCO

## Contact

For questions or issues, please open an issue on the model repository.

---

**Disclaimer**: This is an experimental model for testing purposes. Performance and quality may vary. Thorough evaluation recommended before production use.
