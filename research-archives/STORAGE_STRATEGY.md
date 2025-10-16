# 💾 Storage Strategy - Research Models

## 📦 What's Stored Where

### ✅ Committed to Git (79 files)
- ✅ All research documentation (`.md` files)
- ✅ All test scripts (`.py`, `.sh`)
- ✅ All test results (`.log`, `.txt`)
- ✅ MTEB benchmark results (`.json`)
- ✅ Distillation and evaluation tools

### ⛔ Excluded from Git (stored locally only)
- ⛔ Large model files (`.safetensors` - 502MB total)
  - `research-archives/granite-4.0-micro/granite-4.0-micro-deposium-1024d/` (200MB)
  - `research-archives/qwen25-3b-deposium-1024d/` (302MB)

**Reason**: GitHub has a 100MB file size limit. Models can be regenerated using included scripts.

---

## 🔍 How Models Are NOT Loaded Automatically

### Current Production API (`src/main.py`)

The API only loads models from these locations:
1. `/app/local_models/` (Docker image - not used currently)
2. `models/` (local dev environment - gitignored)
3. HuggingFace Hub (downloaded at startup)

### Research Archives Location

Research models are in `research-archives/**/*-deposium-*/` which is:
- ✅ **NOT** in Docker image
- ✅ **NOT** in `models/` directory
- ✅ **NOT** loaded by `src/main.py`
- ✅ **Gitignored** by pattern `research-archives/**/*-deposium-*/`

**Result**: Research models are **NEVER loaded into RAM** unless explicitly used by a test script.

---

## 🔄 How to Regenerate Models

If you need to recreate the research models:

```bash
# Granite 4.0 Micro (200MB, ~2h on RTX 4050)
python3 research-archives/granite-4.0-micro/distill_granite_4_0_micro.py

# Qwen2.5-3B (302MB, ~1h30 on RTX 4050)
python3 distill_qwen25_3b.py
```

Models will be regenerated in `research-archives/` and remain local-only.

---

## 📊 Directory Structure

```
deposium_embeddings-turbov2/
├── models/                           # ❌ Gitignored - production models
│   ├── qwen25-deposium-1024d/        # ✅ Used by API (downloaded from HF)
│   └── gemma-deposium-768d/          # ✅ Used by API (downloaded from HF)
│
├── research-archives/                # ✅ Partial commit (docs only)
│   ├── README.md                     # ✅ Committed
│   ├── STORAGE_STRATEGY.md           # ✅ Committed
│   │
│   ├── granite-4.0-micro/
│   │   ├── GRANITE_FINAL_DECISION.md      # ✅ Committed
│   │   ├── granite_comparison_results.txt # ✅ Committed
│   │   ├── granite_full_comparison.log    # ✅ Committed
│   │   ├── granite_multilingual_results.log # ✅ Committed
│   │   ├── compare_all_models_v2.py       # ✅ Committed
│   │   ├── test_multilingual_granite.py   # ✅ Committed
│   │   ├── distill_granite_4_0_micro.py   # ✅ Committed
│   │   └── granite-4.0-micro-deposium-1024d/  # ❌ Gitignored (200MB)
│   │       ├── model.safetensors (197MB)  # ⛔ Local only
│   │       ├── tokenizer.json (3.1MB)     # ⛔ Local only
│   │       └── config.json                # ⛔ Local only
│   │
│   └── qwen25-3b-deposium-1024d/     # ❌ Gitignored (302MB)
│       ├── model.safetensors (297MB) # ⛔ Local only
│       ├── tokenizer.json (3.1MB)    # ⛔ Local only
│       └── config.json               # ⛔ Local only
│
└── src/main.py                       # ✅ Committed - API (loads from models/ only)
```

---

## 🎯 Benefits of This Strategy

1. ✅ **No Git bloat**: 502MB models not in repository
2. ✅ **Full documentation**: All research results preserved
3. ✅ **Reproducible**: Can regenerate models anytime
4. ✅ **No RAM usage**: Models not loaded unless explicitly tested
5. ✅ **Fast cloning**: New clones don't download 502MB of unused models
6. ✅ **Local preservation**: Models available locally for future comparisons

---

## 🚀 When You Need Archived Models

### For Testing/Comparison
```bash
# Update test script to use archived models
cd research-archives/granite-4.0-micro/
python3 compare_all_models_v2.py
```

### For Deployment (if needed)
```bash
# Move to production location
cp -r research-archives/granite-4.0-micro/granite-4.0-micro-deposium-1024d/ models/

# Update src/main.py to load it
# (Not recommended - Granite failed evaluation)
```

---

## 📝 Summary

**What's in Git:**
- 79 files: documentation, scripts, logs, MTEB results
- ~5MB total commit size

**What's local only:**
- 2 model directories: 502MB total
- Preserved on your machine
- Regenerable from scripts
- Never loaded into RAM by API

**Result:** Best of both worlds - full research archive without Git bloat or RAM usage.
