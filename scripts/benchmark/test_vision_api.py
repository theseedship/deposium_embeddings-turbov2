#!/usr/bin/env python3
"""
Test script for the /api/vision endpoint with LFM2.5-VL-1.6B
"""
import base64
import requests
import json
import sys
from pathlib import Path

# Configuration
API_URL = "http://localhost:11436"  # Default deposium_embeddings port
API_KEY = "test"  # Will be ignored in dev mode if EMBEDDINGS_API_KEY not set

def test_vision_endpoint(image_path: str, prompt: str = "Extract all text from this document."):
    """Test the /api/vision endpoint with a local image."""

    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return None

    # Read and encode image to base64
    with open(image_path, "rb") as f:
        image_data = f.read()

    image_base64 = base64.b64encode(image_data).decode("utf-8")

    # Determine mime type
    ext = Path(image_path).suffix.lower()
    mime_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    mime_type = mime_types.get(ext, "image/png")

    # Create data URI
    data_uri = f"data:{mime_type};base64,{image_base64}"

    print(f"🔍 Testing /api/vision endpoint")
    print(f"   Image: {image_path}")
    print(f"   Prompt: {prompt}")
    print(f"   Image size: {len(image_data) / 1024:.1f} KB")

    # Send request
    try:
        response = requests.post(
            f"{API_URL}/api/vision",
            headers={
                "Content-Type": "application/json",
                "X-API-Key": API_KEY
            },
            json={
                "model": "lfm25-vl",
                "image": data_uri,
                "prompt": prompt,
                "max_tokens": 512
            },
            timeout=120  # 2 minute timeout for model loading + inference
        )

        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Success!")
            print(f"   Model: {result['model']}")
            print(f"   Latency: {result['latency_ms']:.0f}ms")
            print(f"\n📝 Response:")
            print("-" * 60)
            print(result['response'])
            print("-" * 60)
            return result
        else:
            print(f"\n❌ Error: HTTP {response.status_code}")
            print(f"   {response.text}")
            return None

    except requests.exceptions.ConnectionError:
        print(f"\n❌ Connection error: Could not connect to {API_URL}")
        print("   Make sure the server is running with:")
        print("   cd /home/nico/code_source/tss/deposium_embeddings-turbov2")
        print("   uvicorn src.main:app --host 0.0.0.0 --port 11436")
        return None
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


def test_vision_file_endpoint(image_path: str, prompt: str = "Extract all text from this document."):
    """Test the /api/vision/file endpoint with a local image file."""

    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return None

    print(f"\n🔍 Testing /api/vision/file endpoint")
    print(f"   Image: {image_path}")
    print(f"   Prompt: {prompt}")

    # Send multipart request
    try:
        with open(image_path, "rb") as f:
            response = requests.post(
                f"{API_URL}/api/vision/file",
                headers={"X-API-Key": API_KEY},
                files={"file": (Path(image_path).name, f, "image/png")},
                data={"prompt": prompt, "model": "lfm25-vl", "max_tokens": 512},
                timeout=120
            )

        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Success!")
            print(f"   Model: {result['model']}")
            print(f"   Latency: {result['latency_ms']:.0f}ms")
            print(f"\n📝 Response:")
            print("-" * 60)
            print(result['response'])
            print("-" * 60)
            return result
        else:
            print(f"\n❌ Error: HTTP {response.status_code}")
            print(f"   {response.text}")
            return None

    except requests.exceptions.ConnectionError:
        print(f"\n❌ Connection error: Could not connect to {API_URL}")
        return None
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


def direct_model_test(image_path: str, prompt: str = "Extract all text from this document."):
    """Test LFM2.5-VL directly without the API (for debugging)."""

    print(f"\n🔬 Direct Model Test (bypassing API)")
    print(f"   Image: {image_path}")
    print(f"   Prompt: {prompt}")

    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return None

    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor
        import time

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Device: {device}")

        # Load model
        print("   Loading LFM2.5-VL-1.6B...")
        start_load = time.time()

        model = AutoModelForImageTextToText.from_pretrained(
            "LiquidAI/LFM2.5-VL-1.6B",
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map={"": device} if device == "cuda" else None
        )

        processor = AutoProcessor.from_pretrained(
            "LiquidAI/LFM2.5-VL-1.6B",
            trust_remote_code=True
        )

        if device == "cpu":
            model = model.to(device)

        load_time = time.time() - start_load
        print(f"   Model loaded in {load_time:.1f}s")

        if device == "cuda":
            vram = torch.cuda.memory_allocated() / 1e9
            print(f"   VRAM used: {vram:.2f} GB")

        # Load and process image
        image = Image.open(image_path).convert("RGB")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(model.device)

        # Generate
        print("   Generating response...")
        start_gen = time.time()
        outputs = model.generate(**inputs, max_new_tokens=512)
        gen_time = time.time() - start_gen

        response = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # Extract assistant response
        if "assistant" in response.lower():
            parts = response.split("assistant")
            if len(parts) > 1:
                response = parts[-1].strip()

        print(f"\n✅ Success!")
        print(f"   Generation time: {gen_time*1000:.0f}ms")
        print(f"\n📝 Response:")
        print("-" * 60)
        print(response)
        print("-" * 60)

        return response

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Default test image
    test_image = "test_documents/contrat.png"

    if len(sys.argv) > 1:
        test_image = sys.argv[1]

    prompt = "Extract all text from this document."
    if len(sys.argv) > 2:
        prompt = sys.argv[2]

    print("=" * 60)
    print("LFM2.5-VL Vision API Test")
    print("=" * 60)

    # Check if running from correct directory
    if not Path(test_image).exists():
        # Try relative to script directory
        script_dir = Path(__file__).parent
        test_image = script_dir / test_image

    # Try API test first
    result = test_vision_endpoint(str(test_image), prompt)

    if result is None:
        print("\n" + "=" * 60)
        print("API test failed. Running direct model test...")
        print("=" * 60)
        direct_model_test(str(test_image), prompt)
