# Qwen2.5-7B-Instruct → Model2Vec Distillation Guide

**Target Model:** Qwen/Qwen2.5-7B-Instruct (SOTA 2025)
**Output:** Model2Vec 1024D (~65MB)
**Expected Performance:** 91-95% quality (+7-11% vs Qwen2.5-1.5B baseline)

---

## 🎯 Why Qwen2.5-7B?

### Benchmarks (Full Model)
- **MMLU**: 83.5% (General knowledge)
- **GSM8K**: 93.6% (Math reasoning)
- **HumanEval**: 89.5% (Code generation)

### Advantages
- ✅ **SOTA 2025** - Best performing Qwen model
- ✅ **Multilingual** - 29+ languages
- ✅ **Code-aware** - Trained on massive code corpus
- ✅ **Instruction-tuned** - Excellent for RAG/Q&A
- ✅ **Long context** - 128K tokens
- ✅ **Efficient** - Better than GPT-3.5 at 7B params

### Model2Vec Benefits
- ⚡ **500-1000x faster** inference
- 📦 **~65MB** (vs 14GB full model = 215x smaller)
- 💰 **10-100x cheaper** compute costs
- 🚀 **<1ms latency** (vs 50-500ms)
- 🔋 **Edge deployable** (mobile, IoT, embedded)

---

## 📋 Prerequisites

### Hardware Requirements

**Minimum (CPU-only):**
- RAM: 32GB+
- Storage: 50GB free
- Time: 10-20+ hours

**Recommended (GPU):**
- GPU: NVIDIA with 16GB+ VRAM (RTX 4090, A100, etc.)
- RAM: 32GB+
- Storage: 50GB free
- Time: 2-4 hours

### Software Requirements

```bash
# Python 3.10+
python3 --version

# Virtual environment
python3 -m venv venv
source venv/bin/activate

# Dependencies
pip install -r requirements.txt

# Verify CUDA (if using GPU)
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 🚀 Quick Start (Automated)

### Option 1: Full Pipeline (Recommended)

```bash
# 1. Run distillation (2-4 hours with GPU)
./run_qwen25_7b_distillation.sh

# 2. Test model
./test_qwen25_7b_model.sh

# 3. Evaluate quality
./evaluate_qwen25_7b.sh

# 4. If quality ≥ 91%, deploy
./deploy_qwen25_7b.sh
```

### Option 2: Manual Step-by-Step

```bash
# 1. Distillation
python3 distill_qwen25_7b.py

# 2. Testing
python3 test_qwen25_7b_model.py

# 3. Evaluation
python3 quick_eval_qwen25_7b_1024d.py
```

---

## 📊 Expected Results

### Quality Metrics

| Metric | Qwen2.5-1.5B | Qwen2.5-7B Target | Improvement |
|--------|--------------|-------------------|-------------|
| Overall Quality | 68.2% | **91-95%** | **+23-27%** |
| Instruction Awareness | 95.3% | **96-98%** | **+1-3%** |
| Semantic Similarity | 95.0% | **96-98%** | **+1-3%** |
| Code Understanding | 86.4% | **92-96%** | **+6-10%** |
| Domain Knowledge | 65-70% | **88-92%** | **+18-25%** |
| Multilingual | 60-65% | **85-90%** | **+20-28%** |

### Model Specs

| Spec | Value |
|------|-------|
| Model Size | ~65MB |
| Dimensions | 1024D |
| Vocabulary | 32K tokens (Qwen tokenizer) |
| Inference Speed | 500-1000x faster than full model |
| Latency | <1ms per query |
| Context Length | N/A (static embeddings) |

---

## 🔧 Configuration Options

### Distillation Parameters

Edit `distill_qwen25_7b.py` to customize:

```python
CONFIG = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "dimensions": 1024,           # Final embedding dimensions
    "pca_dims": 1024,             # PCA dimensions (1024 recommended)
    "apply_pca": True,            # Apply PCA dimensionality reduction
    "apply_zipf": True,           # Apply Zipf weighting (improves quality)
    "num_layers": -1,             # Use all layers (-1) for best quality
    "corpus_size": 1_000_000,     # Training corpus size (1M recommended)
}
```

### Quality vs Speed Trade-offs

**For Maximum Quality (91-95% target):**
```python
dimensions = 1024
pca_dims = 1024
apply_pca = True
apply_zipf = True
num_layers = -1  # All layers
```

**For Faster Distillation (88-92% quality):**
```python
dimensions = 768
pca_dims = 768
apply_pca = True
apply_zipf = True
num_layers = 24  # Last 24 layers only
```

**For Smallest Size (85-89% quality):**
```python
dimensions = 512
pca_dims = 512
apply_pca = True
apply_zipf = True
num_layers = 16  # Last 16 layers only
```

---

## 🐛 Troubleshooting

### Out of Memory (OOM)

**GPU OOM:**
```bash
# Reduce batch size or use CPU
export CUDA_VISIBLE_DEVICES=""  # Force CPU
python3 distill_qwen25_7b.py
```

**CPU OOM:**
```bash
# Increase swap space
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Slow Distillation

**Expected times:**
- GPU (RTX 4090): 2-3 hours
- GPU (RTX 3090): 3-4 hours
- GPU (RTX 3060): 4-6 hours
- CPU (32 cores): 10-15 hours
- CPU (8 cores): 20-30 hours

**Speed up:**
- Use GPU if available
- Reduce `corpus_size` (min: 500K)
- Use `distillation_method = "simple"` (faster but -2-3% quality)

### Low Quality Score

**If score < 91%:**

1. **Re-run with better parameters:**
   ```python
   pca_dims = 1536  # Higher dimensions
   corpus_size = 2_000_000  # Larger corpus
   ```

2. **Check model loaded correctly:**
   ```bash
   python3 test_qwen25_7b_model.py
   ```

3. **Compare with baseline:**
   ```bash
   python3 quick_eval_qwen25_1024d.py  # Old model
   python3 quick_eval_qwen25_7b_1024d.py  # New model
   ```

4. **Verify source model:**
   ```python
   from transformers import AutoModel
   model = AutoModel.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
   print(model.config)
   ```

---

## 📈 Performance Monitoring

### During Distillation

Monitor progress in terminal:
```bash
# GPU usage
watch -n 1 nvidia-smi

# Memory usage
watch -n 1 free -h

# CPU usage
htop
```

### After Distillation

Check model size and quality:
```bash
# Model size
du -sh models/qwen25-7b-deposium-1024d/

# Quality check
python3 quick_eval_qwen25_7b_1024d.py
```

---

## 🚀 Deployment

### Update API

Edit `api.py`:
```python
# Change model path
MODEL_PATH = "models/qwen25-7b-deposium-1024d"

# Version info
VERSION = "11.0.0"
MODEL_NAME = "Qwen2.5-7B-1024D"
```

### Build Docker

```bash
# Update Dockerfile to include new model
docker build -t deposium-embeddings-v11 .

# Test
docker run -p 8080:8080 deposium-embeddings-v11

# Deploy
docker tag deposium-embeddings-v11 registry.example.com/deposium-embeddings:v11
docker push registry.example.com/deposium-embeddings:v11
```

### Update Documentation

Update `README.md`:
- Model specs
- Benchmark results
- Performance metrics
- Comparison tables

---

## 📊 Evaluation Results Template

After running evaluation, document results:

```markdown
# Qwen2.5-7B-1024D Evaluation Results

**Date:** YYYY-MM-DD
**Model:** Qwen2.5-7B-1024D Model2Vec
**Size:** 65MB

## Results

| Metric | Score | vs Target | vs Baseline |
|--------|-------|-----------|-------------|
| Overall Quality | XX.X% | ±X.X% | +X.X% |
| Instruction Awareness | XX.X% | ±X.X% | +X.X% |
| Semantic Similarity | XX.X% | ±X.X% | +X.X% |
| Code Understanding | XX.X% | ±X.X% | +X.X% |
| Domain Knowledge | XX.X% | ±X.X% | +X.X% |
| Multilingual | XX.X% | ±X.X% | +X.X% |

## Conclusion

[✅/⚠️/❌] Target achieved/close/missed
[Details...]
```

---

## 🎯 Success Criteria

**Ready for Production if:**
- ✅ Overall quality ≥ 91%
- ✅ Instruction awareness ≥ 95%
- ✅ Code understanding ≥ 90%
- ✅ Model size ≤ 70MB
- ✅ All tests pass

**Need Re-distillation if:**
- ❌ Overall quality < 88%
- ❌ Instruction awareness < 90%
- ❌ Tests fail
- ❌ Model size > 80MB

---

## 📚 Additional Resources

### Model2Vec Documentation
- https://github.com/MinishLab/model2vec

### Qwen2.5 Documentation
- https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- https://github.com/QwenLM/Qwen2.5

### Benchmarks
- MTEB: https://huggingface.co/spaces/mteb/leaderboard
- Custom evaluation scripts in this repo

---

## 🤝 Support

**Issues:**
- Check logs in `distillation_metadata.txt`
- Review error messages in terminal
- Compare with baseline results

**Questions:**
- Refer to Model2Vec documentation
- Check Qwen2.5 model card
- Review distillation script comments

---

**Status:** Ready for distillation
**Last Updated:** 2025-10-14
**Priority:** ABSOLUTE
