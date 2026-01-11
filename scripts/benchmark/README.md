# VLM Benchmark for Document OCR

Benchmark Vision-Language Models for document text extraction and classification.

## Models Tested

| Model | DocVQA | OCRBench | VRAM (INT4) | CPU Viable |
|-------|--------|----------|-------------|------------|
| **InternVL2-2B** | 86.9% | 784 | 1-2GB | No |
| **Moondream2** | 79.3% | 61.2 | ~1GB | Slow |
| **LFM2.5-VL-1.6B** | - | 41.4%* | ~1.5GB | Yes |

*LFM2.5-VL uses OCRBench v2 (different metric)

## Installation

### Option 1: Conda Environment (Recommended)

```bash
# Create environment
conda create -n vlm-benchmark python=3.10 -y
conda activate vlm-benchmark

# Install PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install transformers>=4.37.2 accelerate bitsandbytes pillow

# Optional: Flash Attention for InternVL2 (Linux only, requires CUDA)
pip install flash-attn --no-build-isolation

# Optional: timm for InternVL2
pip install timm
```

### Option 2: Quick Install (existing environment)

```bash
pip install torch torchvision transformers accelerate bitsandbytes pillow
```

## Usage

### 1. Add Test Documents

Place document images in `test_documents/`:

```bash
# Copy your test documents
cp /path/to/facture.png test_documents/
cp /path/to/formulaire.png test_documents/
cp /path/to/ticket.jpg test_documents/
```

Recommended test cases:
- Simple text document (single column)
- Invoice/receipt
- Form with fields
- Document with tables
- Multi-column layout

### 2. Run Benchmark

```bash
# Test all models
python vlm_benchmark.py

# Test specific model(s)
python vlm_benchmark.py --models internvl2
python vlm_benchmark.py --models moondream2,lfm25_vl

# Specify custom image directory
python vlm_benchmark.py --images-dir /path/to/documents

# Force CPU (for LFM2.5-VL)
python vlm_benchmark.py --models lfm25_vl --device cpu
```

### 3. View Results

Results are saved to `vlm_benchmark_results.json`.

```bash
cat vlm_benchmark_results.json | python -m json.tool
```

## Quick Test (Single Model)

### Test InternVL2-2B (requires GPU)

```python
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image

model = AutoModel.from_pretrained(
    "OpenGVLab/InternVL2-2B",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
    trust_remote_code=True,
    device_map="auto"
).eval()

tokenizer = AutoTokenizer.from_pretrained(
    "OpenGVLab/InternVL2-2B",
    trust_remote_code=True
)

image = Image.open("test_documents/your_document.png").convert("RGB")
pixel_values = model.process_image(image).to(model.device, dtype=model.dtype)

response = model.chat(
    tokenizer,
    pixel_values,
    "<image>\nExtract all text from this document.",
    dict(max_new_tokens=512)
)
print(response)
```

### Test LFM2.5-VL-1.6B (CPU compatible)

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

model = AutoModelForImageTextToText.from_pretrained(
    "LiquidAI/LFM2.5-VL-1.6B",
    device_map="cpu",
    low_cpu_mem_usage=True
)
processor = AutoProcessor.from_pretrained("LiquidAI/LFM2.5-VL-1.6B")

image = Image.open("test_documents/your_document.png").convert("RGB")

conversation = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Extract all text from this document."},
    ],
}]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    tokenize=True,
)

outputs = model.generate(**inputs, max_new_tokens=512)
print(processor.batch_decode(outputs, skip_special_tokens=True)[0])
```

### Test Moondream2

```python
from transformers import AutoModelForCausalLM
from PIL import Image

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True,
    device_map="auto"
)

image = Image.open("test_documents/your_document.png").convert("RGB")
result = model.query(image, "Extract all text from this document.")
print(result)
```

## Prompts Tested

| Prompt | Purpose |
|--------|---------|
| `ocr_simple` | Basic text extraction |
| `ocr_structured` | Text + structure preservation |
| `classify` | Document complexity classification |
| `summary` | Content summarization |

## Expected Performance

### GPU (RTX 3060 12GB, NVIDIA A100)

| Model | Load Time | Latency/Image |
|-------|-----------|---------------|
| InternVL2-2B (4bit) | ~30s | ~200-500ms |
| Moondream2 | ~20s | ~300-600ms |
| LFM2.5-VL-1.6B | ~15s | ~100-300ms |

### CPU (AMD Ryzen 5800X)

| Model | Load Time | Latency/Image |
|-------|-----------|---------------|
| LFM2.5-VL-1.6B | ~30s | ~200-500ms |
| Moondream2 | ~30s | ~1-3s |
| InternVL2-2B | N/A | Not recommended |

## Recommendations

### Best Quality (GPU required)
**InternVL2-2B** - 86.9% DocVQA, comparable to Llama 4 Maverick

### Best for CPU/Edge
**LFM2.5-VL-1.6B** - Edge-first design, 2,975 tok/s prefill on CPU

### Balanced (GPU preferred)
**Moondream2** - Good balance of quality and speed

## Troubleshooting

### CUDA Out of Memory

```bash
# Use 4-bit quantization (default for InternVL2)
# Or reduce batch size in the script

# Check VRAM usage
nvidia-smi
```

### Flash Attention Error

```bash
# Flash attention requires CUDA and Linux
# If installation fails, the script falls back to standard attention
pip uninstall flash-attn
# Re-run benchmark without flash attention
```

### Model Download Slow

```bash
# Set HuggingFace cache
export HF_HOME=/path/to/large/disk/.cache/huggingface

# Or use huggingface-cli
huggingface-cli download OpenGVLab/InternVL2-2B
huggingface-cli download vikhyatk/moondream2 --revision 2025-01-09
huggingface-cli download LiquidAI/LFM2.5-VL-1.6B
```
