# ✅ Pre-Distillation Checklist - Qwen2.5-7B

**Date:** 2025-10-14
**Target:** Qwen2.5-7B-Instruct → Model2Vec 1024D
**Expected Duration:** 2-4 hours (GPU) or 10-20 hours (CPU)

---

## 🔍 System Requirements

### Hardware Check

- [ ] **RAM:** 32GB+ available
  ```bash
  free -h
  # Should show 32GB+ total
  ```

- [ ] **Disk Space:** 50GB+ free
  ```bash
  df -h .
  # Should show 50GB+ available
  ```

- [ ] **GPU (Recommended):** 16GB+ VRAM
  ```bash
  nvidia-smi
  # Should show GPU with 16GB+ memory
  ```

- [ ] **CPU (Alternative):** 8+ cores for reasonable speed
  ```bash
  nproc
  # Should show 8+
  ```

---

## 🐍 Python Environment

### Python Version

- [ ] **Python 3.10+** installed
  ```bash
  python3 --version
  # Should show Python 3.10.x or higher
  ```

### Virtual Environment

- [ ] **Virtual environment** created and activated
  ```bash
  # Create if not exists
  python3 -m venv venv

  # Activate
  source venv/bin/activate

  # Verify
  which python3
  # Should show path inside venv/
  ```

### Dependencies

- [ ] **All dependencies** installed
  ```bash
  pip install -r requirements.txt

  # Verify critical packages
  python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
  python3 -c "import model2vec; print('Model2Vec: OK')"
  python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
  ```

---

## 🚀 GPU Configuration (if using GPU)

### CUDA Check

- [ ] **CUDA available** in PyTorch
  ```bash
  python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
  # Should show: CUDA available: True
  ```

- [ ] **GPU details** correct
  ```bash
  python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
  python3 -c "import torch; print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')"
  ```

---

## 📁 File System Check

### Scripts Present

- [ ] **Distillation script** exists
  ```bash
  ls -lh distill_qwen25_7b.py
  # Should show ~5KB file
  ```

- [ ] **Test script** exists
  ```bash
  ls -lh test_qwen25_7b_model.py
  # Should show ~6KB file
  ```

- [ ] **Evaluation script** exists
  ```bash
  ls -lh quick_eval_qwen25_7b_1024d.py
  # Should show ~13KB file
  ```

- [ ] **Automation scripts** executable
  ```bash
  ls -lh run_qwen25_7b_distillation.sh
  # Should show -rwxr-xr-x (executable)
  ```

### Output Directory

- [ ] **Output directory** writable
  ```bash
  mkdir -p models/
  touch models/.test_write && rm models/.test_write
  # Should complete without error
  ```

---

## 🌐 Network & Download

### HuggingFace Access

- [ ] **Can access HuggingFace**
  ```bash
  curl -s https://huggingface.co > /dev/null && echo "✅ HuggingFace accessible" || echo "❌ Cannot reach HuggingFace"
  ```

- [ ] **HuggingFace CLI** configured (optional)
  ```bash
  # Optional: Login if you want faster downloads
  huggingface-cli login
  ```

### Disk Space for Download

- [ ] **15GB+ free** for model download
  ```bash
  df -h .
  # Qwen2.5-7B is ~14GB, need buffer for distillation
  ```

---

## ⚙️ Configuration Review

### Distillation Parameters

- [ ] **Review configuration** in `distill_qwen25_7b.py`
  ```python
  CONFIG = {
      "model_name": "Qwen/Qwen2.5-7B-Instruct",  # ✓ Correct
      "dimensions": 1024,                         # ✓ Target dimensions
      "pca_dims": 1024,                           # ✓ Same as dimensions
      "device": "cuda" or "cpu",                  # ✓ Auto-detected
  }
  ```

- [ ] **Confirm distillation method**
  ```python
  "apply_pca": True,      # ✓ Recommended for quality
  "apply_zipf": True,     # ✓ Improves quality
  "num_layers": -1,       # ✓ Use all layers for best quality
  ```

---

## 📊 Baseline Check

### Current Model Performance (for comparison)

- [ ] **Know baseline score** (Qwen2.5-1.5B: 68.2%)
  ```bash
  # Optional: Re-run baseline to confirm
  python3 quick_eval_qwen25_1024d.py
  ```

- [ ] **Understand target** (91-95% for Qwen2.5-7B)

---

## 🕐 Time Planning

### Estimate Completion Time

- [ ] **GPU users:** Plan for 2-4 hours
  - Can leave unattended
  - Check back after 3 hours

- [ ] **CPU users:** Plan for 10-20 hours
  - Consider running overnight
  - Use `screen` or `tmux` to avoid interruption
  ```bash
  # Start screen session
  screen -S distillation

  # Run distillation
  ./run_qwen25_7b_distillation.sh

  # Detach: Ctrl+A, then D
  # Reattach later: screen -r distillation
  ```

---

## 🔒 Process Management

### Long-Running Process Setup

- [ ] **Consider using screen/tmux** (especially for CPU users)
  ```bash
  # Install if needed
  sudo apt-get install screen  # or tmux

  # Start session
  screen -S qwen25_distillation
  ```

- [ ] **Or run in background** with nohup
  ```bash
  nohup ./run_qwen25_7b_distillation.sh > distillation.log 2>&1 &

  # Monitor with
  tail -f distillation.log
  ```

---

## 📝 Monitoring Plan

### How to Monitor Progress

- [ ] **Know how to check GPU usage** (if using GPU)
  ```bash
  watch -n 1 nvidia-smi
  ```

- [ ] **Know how to check logs**
  ```bash
  # If running in screen
  screen -r distillation

  # If using nohup
  tail -f distillation.log
  ```

- [ ] **Understand expected output**
  - Initial: "Loading model..."
  - Middle: Progress bars for encoding
  - Final: "✅ Distillation complete!"

---

## 🎯 Success Criteria

### What to Expect

- [ ] **Distillation completes** without errors
- [ ] **Model size** approximately 65MB
- [ ] **Output directory** contains model files:
  ```
  models/qwen25-7b-deposium-1024d/
  ├── config.json
  ├── model.safetensors (or .bin)
  ├── tokenizer.json
  └── distillation_metadata.txt
  ```

---

## 🆘 Emergency Contacts

### If Something Goes Wrong

- [ ] **Know how to stop the process**
  ```bash
  # If in foreground: Ctrl+C

  # If in background:
  ps aux | grep distill
  kill <PID>
  ```

- [ ] **Have troubleshooting guide ready**
  ```bash
  cat QWEN25_7B_DISTILLATION_GUIDE.md
  # Section: Troubleshooting
  ```

---

## ✅ Final Check

### Ready to Start?

All items above checked? ✅

If yes, proceed with:
```bash
./run_qwen25_7b_distillation.sh
```

If no, review unchecked items and resolve before starting.

---

## 📋 Post-Distillation Steps

### After Completion

Once distillation finishes:

1. [ ] **Verify model saved**
   ```bash
   ls -lh models/qwen25-7b-deposium-1024d/
   ```

2. [ ] **Check model size** (~65MB expected)
   ```bash
   du -sh models/qwen25-7b-deposium-1024d/
   ```

3. [ ] **Run tests**
   ```bash
   ./test_qwen25_7b_model.sh
   ```

4. [ ] **Run evaluation**
   ```bash
   ./evaluate_qwen25_7b.sh
   ```

5. [ ] **If score ≥ 91%, deploy**
   ```bash
   ./deploy_qwen25_7b.sh
   ```

---

## 🔄 Rollback Plan

### If Distillation Fails

- [ ] **Original models preserved** (nothing modified)
- [ ] **Can restart** distillation anytime
- [ ] **Can adjust parameters** in `distill_qwen25_7b.py`
- [ ] **Can try CPU mode** if GPU OOM

---

**Checklist Completed:** _______________
**Ready to Start:** YES / NO
**Start Time:** _______________
**Expected End:** _______________ (+ 2-4 hours)

---

**Good luck! 🚀**
