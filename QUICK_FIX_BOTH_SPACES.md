# QUICK FIX: Both HuggingFace Spaces (5 min)

## STATUS
- **Qwen Space**: Crashed with ImportError (wrong API)
- **Nemotron Space**: Crashed with build error (mamba-ssm OOM)
- **Solution**: Update 2 files per Space with corrected code

---

## FIX #1: Qwen2.5-7B Space

### Step 1: Edit requirements.txt
Go to: Your Qwen Space → Files → Edit `requirements.txt`

**Replace entire file with:**
```txt
model2vec[distill]>=0.6.0
torch>=2.0.0
transformers>=4.50.0
gradio>=4.0.0
numpy>=1.24.0
sentencepiece>=0.1.99
protobuf>=3.20.0
```

Key change: `model2vec[distill]>=0.6.0` (added `[distill]` extra)

### Step 2: Edit app.py
Go to: Your Qwen Space → Files → Edit `app.py`

**Replace entire file with content from:**
```bash
cat huggingface_space_app_FIXED.py
```

Or make these 3 manual changes:
1. Line 9: Change `from model2vec import distill_model` → `from model2vec.distill import distill`
2. Lines 59-63: Change `model = distill_model(...)` → `model = distill(...)`
3. Line 82: Change `model.encode(test_texts, show_progress_bar=False)` → `model.encode(test_texts)`

### Step 3: Commit and Relaunch
- Click "Commit changes to main"
- Wait 5-10 min for rebuild
- Go to App tab → Click "🚀 Start Distillation"
- Wait 30-60 min
- Download ZIP when done

**Cost**: ~$0.50-1.00 on A10G small

---

## FIX #2: Nemotron Space

### Option A: Fix Existing Space (if already created)

#### Step 1: Edit requirements.txt
Go to: Your Nemotron Space → Files → Edit `requirements.txt`

**Replace entire file with:**
```txt
model2vec[distill]>=0.6.0
torch>=2.0.0
transformers>=4.50.0
gradio>=4.0.0
numpy>=1.24.0
sentencepiece>=0.1.99
protobuf>=3.20.0
```

**IMPORTANT**: NO `mamba-ssm` line! (causes OOM during build)

#### Step 2: Edit app.py
Go to: Your Nemotron Space → Files → Edit `app.py`

**Replace entire file with content from:**
```bash
cat huggingface_nemotron_app.py
```

(This file is now corrected with proper API)

#### Step 3: Commit and Launch
- Click "Commit changes to main"
- Wait 10-15 min for rebuild (larger model)
- Go to App tab → Click "🚀 Start Distillation"
- Wait 1-2 hours
- Download ZIP when done

**Hardware**: Use A10G large (46GB VRAM) - $1.50/h
**Cost**: ~$1.50-3.00

---

### Option B: Create New Space (recommended if build keeps failing)

1. Go to: https://huggingface.co/spaces
2. Click "Create new Space"
3. Settings:
   - Name: `deposium-nemotron-nano-distillation`
   - License: Apache 2.0
   - SDK: Gradio
   - Hardware: A10G large ($1.50/h)
4. Upload files:
   - `requirements.txt` from `requirements_FIXED_NEMOTRON.txt`
   - `app.py` from `huggingface_nemotron_app.py`
5. Launch!

---

## VERIFICATION

After editing files, check these lines are correct:

**requirements.txt (both Spaces):**
```txt
model2vec[distill]>=0.6.0  ← Must have [distill]
```

**app.py (both Spaces):**
```python
from model2vec.distill import distill  ← Line ~9

model = distill(  ← Line ~59-77
    model_name="...",
    pca_dims=1024,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

embeddings = model.encode(test_texts)  ← NO show_progress_bar parameter
```

---

## TIMELINE

```
NOW         - Fix Qwen Space (2 min edit)
            - Fix/Create Nemotron Space (2 min edit)

+5 min      - Qwen rebuild complete
+10 min     - Nemotron rebuild complete

+15 min     - Launch both distillations

+45 min     - Qwen finishes (download ZIP)
+2h         - Nemotron finishes (download ZIP)

+2.5h       - Test both locally
            - Compare results
            - Deploy best model
```

---

## FILES REFERENCE

**For Qwen**:
- Corrected app.py: `huggingface_space_app_FIXED.py`
- Corrected requirements: `requirements_FIXED.txt`

**For Nemotron**:
- Corrected app.py: `huggingface_nemotron_app.py`
- Corrected requirements: `requirements_FIXED_NEMOTRON.txt`

---

## EXPECTED RESULTS

**Qwen2.5-7B Model2Vec**:
- Size: ~65MB
- Quality: 91-95%
- Instruction-awareness: 96-98%
- Best for: General-purpose embeddings

**Nemotron-Nano-9B Model2Vec**:
- Size: ~268MB (4x larger due to 131K vocab)
- Quality: 90-94%
- Instruction-awareness: 95-97%
- Best for: Advanced reasoning, NVIDIA hardware

**Recommendation**: Start with Qwen (cheaper, proven). Add Nemotron if needed for specialized tasks.

---

## TOTAL COST ESTIMATE

- Qwen: $0.50-1.00
- Nemotron: $1.50-3.00
- **Total: ~$2-4** for both models

---

**Status**: Ready to fix!
**ETA**: 2-3 hours total (including distillations)
