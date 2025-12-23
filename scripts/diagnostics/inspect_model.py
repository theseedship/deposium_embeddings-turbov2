import torch
from transformers import AutoConfig, AutoModel
import os

# Set env to allow offline if needed, though we might need to download config
os.environ["HF_HUB_OFFLINE"] = "0" 

model_id = "Qwen/Qwen2.5-0.5B" # Using a known similar model as proxy if Qwen3 doesn't exist publicly yet or is private
# The code uses "Qwen/Qwen3-Embedding-0.6B". I'll try to load its config if possible, otherwise fallback.
real_model_id = "Qwen/Qwen3-Embedding-0.6B"

try:
    print(f"Attempting to load config for {real_model_id}...")
    config = AutoConfig.from_pretrained(real_model_id, trust_remote_code=True)
    print(f"Successfully loaded config for {real_model_id}")
    print(f"Architecture: {config.architectures}")
    print(f"Model Type: {config.model_type}")
except Exception as e:
    print(f"Could not load {real_model_id}: {e}")
    print(f"Falling back to {model_id} for capability check...")
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        print(f"Loaded fallback config for {model_id}")
        print(f"Model Type: {config.model_type}")
    except Exception as e2:
        print(f"Failed to load fallback: {e2}")

print("\nChecking PyTorch version and CUDA:")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Flash Attention available (SDPA): {hasattr(torch.nn.functional, 'scaled_dot_product_attention')}")

# Check if flash_attn is installed
try:
    import flash_attn
    print(f"flash_attn package installed: {flash_attn.__version__}")
except ImportError:
    print("flash_attn package NOT installed")

# Check if vllm is installed
try:
    import vllm
    print(f"vllm installed: {vllm.__version__}")
except ImportError:
    print("vllm NOT installed")
