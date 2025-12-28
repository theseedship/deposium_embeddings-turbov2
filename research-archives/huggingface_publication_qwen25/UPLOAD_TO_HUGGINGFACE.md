# How to Upload to HuggingFace

## Method 1: Using HuggingFace Web Interface (Easiest)

1. Go to https://huggingface.co/tss-deposium/qwen25-deposium-1024d
2. Click "Files and versions" tab
3. Click "Add file" → "Upload files"
4. Upload these files:
   ```
   BENCHMARKS.md
   QUICK_START.md
   requirements.txt
   examples/instruction_awareness_demo.py
   examples/real_world_use_cases.py
   ```
5. Add commit message: "Add examples and documentation"
6. Click "Commit changes"

## Method 2: Using Git (For Batch Upload)

```bash
# 1. Install huggingface-cli
pip install huggingface_hub

# 2. Login (one-time)
huggingface-cli login
# Enter your token from https://huggingface.co/settings/tokens

# 3. Clone the repo
git clone https://huggingface.co/tss-deposium/qwen25-deposium-1024d
cd qwen25-deposium-1024d

# 4. Copy files from this folder
cp ../huggingface_publication/BENCHMARKS.md .
cp ../huggingface_publication/QUICK_START.md .
cp ../huggingface_publication/requirements.txt .
mkdir -p examples
cp ../huggingface_publication/examples/*.py examples/

# 5. Commit and push
git add .
git commit -m "Add examples, benchmarks, and documentation

- BENCHMARKS.md: Detailed comparison with other models
- QUICK_START.md: Quick start guide
- examples/instruction_awareness_demo.py: Interactive demo
- examples/real_world_use_cases.py: Real-world use cases
- requirements.txt: Dependencies
"
git push

# Done! Files are now on HuggingFace
```

## Method 3: Using HuggingFace Hub API (Programmatic)

```python
from huggingface_hub import HfApi, login

# Login (one-time)
login()  # Enter your token

# Upload files
api = HfApi()

files_to_upload = {
    "BENCHMARKS.md": "huggingface_publication/BENCHMARKS.md",
    "QUICK_START.md": "huggingface_publication/QUICK_START.md",
    "requirements.txt": "huggingface_publication/requirements.txt",
    "examples/instruction_awareness_demo.py": "huggingface_publication/examples/instruction_awareness_demo.py",
    "examples/real_world_use_cases.py": "huggingface_publication/examples/real_world_use_cases.py",
}

for hf_path, local_path in files_to_upload.items():
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=hf_path,
        repo_id="tss-deposium/qwen25-deposium-1024d",
        repo_type="model",
        commit_message=f"Add {hf_path}"
    )

print("✅ All files uploaded!")
```

## Files to Upload

```
✅ README.md (already uploaded)
⬜ BENCHMARKS.md (comprehensive comparison)
⬜ QUICK_START.md (quick start guide)
⬜ requirements.txt (dependencies)
⬜ examples/instruction_awareness_demo.py (interactive demo)
⬜ examples/real_world_use_cases.py (use cases)
```

## Verification

After upload, check:
- Files appear in https://huggingface.co/tss-deposium/qwen25-deposium-1024d/tree/main
- Examples folder exists
- README.md displays correctly
- Users can download examples

---

**Recommended:** Use Method 1 (Web Interface) for simplicity, or Method 2 (Git) for batch upload.
