# Qwen2.5-7B-1024D Quick Start

**🎯 Target: 91-95% quality (+7-11% improvement)**
**⚡ Time: 2-4 hours with GPU**

---

## ✅ Prerequisites Checklist

- [ ] Python 3.10+ installed
- [ ] 32GB+ RAM available
- [ ] 50GB+ free disk space
- [ ] GPU with 16GB+ VRAM (recommended) OR patience for CPU (10-20h)
- [ ] Virtual environment activated
- [ ] Dependencies installed

---

## 🚀 3-Step Quick Start

### Step 1: Distill (2-4 hours)

```bash
./run_qwen25_7b_distillation.sh
```

**What this does:**
- Downloads Qwen/Qwen2.5-7B-Instruct (14GB)
- Distills to Model2Vec 1024D (~65MB)
- Saves to `models/qwen25-7b-deposium-1024d/`

**Expected output:**
```
🔥 Starting distillation...
✅ Model loaded! Dimensions: 1024
⏱️  Duration: 2-4 hours
💾 Model size: 65MB
✅ Distillation complete!
```

### Step 2: Test (2 minutes)

```bash
./test_qwen25_7b_model.sh
```

**What this does:**
- Basic encoding test
- Semantic similarity check
- Instruction awareness test
- Code understanding test
- Multilingual test

**Expected output:**
```
✅ Basic encoding: Working
✅ Semantic similarity: Working
✅ Instruction awareness: Yes
✅ Code understanding: Working
✅ Multilingual: Working
🎉 All tests PASSED!
```

### Step 3: Evaluate (5 minutes)

```bash
./evaluate_qwen25_7b.sh
```

**What this does:**
- Comprehensive quality evaluation
- 6 category scores
- Comparison with baseline

**Expected output:**
```
🏆 OVERALL QUALITY: 0.920 (92.0%)

📊 Comparison:
  Previous (Qwen2.5-1.5B): 68.2%
  Current (Qwen2.5-7B):    92.0%
  Improvement:             +23.8%

✅ TARGET ACHIEVED!
```

---

## 🎯 If Quality ≥ 91%: Deploy!

```bash
./deploy_qwen25_7b.sh
```

**What this does:**
1. Updates `api.py` to use new model
2. Updates `Dockerfile`
3. Builds Docker image `deposium-embeddings-v11`
4. Tests container
5. Creates deployment summary

**Next: Production Deployment**
```bash
# Tag and push
docker tag deposium-embeddings-v11:latest your-registry/deposium-embeddings:v11
docker push your-registry/deposium-embeddings:v11

# Deploy (adjust for your setup)
kubectl set image deployment/deposium-embeddings ...
# OR
docker-compose up -d
```

---

## ⚠️ If Quality < 91%: Troubleshoot

### Check 1: Verify Model Loaded Correctly
```bash
python3 test_qwen25_7b_model.py
```

### Check 2: Re-distill with Better Parameters

Edit `distill_qwen25_7b.py`:
```python
CONFIG = {
    "pca_dims": 1536,  # Increase from 1024
    "corpus_size": 2_000_000,  # Increase from 1M
}
```

Then re-run:
```bash
./run_qwen25_7b_distillation.sh
```

### Check 3: Compare with Baseline
```bash
python3 quick_eval_qwen25_1024d.py      # Old: 68.2%
python3 quick_eval_qwen25_7b_1024d.py   # New: should be 91-95%
```

---

## 📊 Expected Timeline

| Step | GPU Time | CPU Time |
|------|----------|----------|
| 1. Distillation | 2-4 hours | 10-20 hours |
| 2. Testing | 2 minutes | 2 minutes |
| 3. Evaluation | 5 minutes | 5 minutes |
| 4. Deployment | 10 minutes | 10 minutes |
| **Total** | **2-4 hours** | **10-20 hours** |

---

## 🎉 Success Indicators

- ✅ Distillation completes without errors
- ✅ All tests pass
- ✅ Overall quality ≥ 91%
- ✅ Instruction awareness ≥ 95%
- ✅ Code understanding ≥ 90%
- ✅ Model size ≤ 70MB
- ✅ Docker container runs successfully

---

## 🆘 Common Issues

### Issue: Out of Memory
**Solution:** Use CPU mode or increase swap
```bash
export CUDA_VISIBLE_DEVICES=""  # Force CPU
```

### Issue: Slow Distillation
**Solution:** This is normal on CPU. Expected: 10-20 hours

### Issue: Low Quality Score
**Solution:** Re-distill with better parameters (see above)

### Issue: Model Not Found
**Solution:** Check path in scripts
```bash
ls -lh models/qwen25-7b-deposium-1024d/
```

---

## 📚 Full Documentation

- Comprehensive guide: `QWEN25_7B_DISTILLATION_GUIDE.md`
- Script documentation: Comments in each `.py` file
- Deployment guide: `DEPLOYMENT_SUMMARY_V11.md` (after deployment)

---

## 🔄 Complete Command Sequence

```bash
# 1. Setup (one-time)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Distill → Test → Evaluate → Deploy
./run_qwen25_7b_distillation.sh   # 2-4h
./test_qwen25_7b_model.sh          # 2min
./evaluate_qwen25_7b.sh            # 5min
./deploy_qwen25_7b.sh              # 10min

# 3. Production deployment
docker run -p 8080:8080 deposium-embeddings-v11:latest
```

---

**Ready? Start here:**
```bash
./run_qwen25_7b_distillation.sh
```

**Questions? See:** `QWEN25_7B_DISTILLATION_GUIDE.md`

---

**Status:** ✅ Ready to start
**Priority:** 🔥 ABSOLUTE
**Expected outcome:** 91-95% quality model in 2-4 hours
