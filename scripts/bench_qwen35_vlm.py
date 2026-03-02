#!/usr/bin/env python3
"""
Benchmark: Qwen3.5-0.8B Q2 GGUF for SIMPLE/COMPLEX document classification.

Tests:
- Load time (one-time)
- Inference latency per image
- Classification quality on synthetic doc images

Usage:
  python scripts/bench_qwen35_vlm.py
  python scripts/bench_qwen35_vlm.py --n-threads 4 --image path/to/doc.png
"""

import argparse
import base64
import io
import time
import sys
from pathlib import Path

# Paths
ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models/qwen35-0.8b/Qwen3.5-0.8B-UD-IQ2_XXS.gguf"
MMPROJ_PATH = ROOT / "models/qwen35-0.8b/mmproj-F16.gguf"


def make_test_images():
    """Generate synthetic document images for testing."""
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np

    images = {}

    # SIMPLE: plain text document (white background, black lines of text)
    img_simple = Image.new("RGB", (400, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img_simple)
    for i in range(12):
        y = 20 + i * 22
        width = 300 + np.random.randint(-80, 50)
        draw.rectangle([20, y, 20 + width, y + 12], fill=(30, 30, 30))
    images["SIMPLE"] = img_simple

    # COMPLEX: document with table/grid + mixed content
    img_complex = Image.new("RGB", (400, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img_complex)
    # Grid / table
    for col in range(4):
        for row in range(6):
            x, y = 20 + col * 90, 20 + row * 40
            draw.rectangle([x, y, x + 82, y + 32], outline=(0, 0, 0), width=1)
            draw.rectangle([x + 4, y + 8, x + 60 + (col * 5), y + 16], fill=(80, 80, 180))
    # Bar chart at bottom
    for i in range(8):
        h = 20 + i * 8
        x = 30 + i * 40
        draw.rectangle([x, 260 - h, x + 28, 260], fill=(200, 100, 50))
    images["COMPLEX"] = img_complex

    return images


def image_to_base64_uri(img):
    """Convert PIL image to data URI."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


CLASSIFY_PROMPT = """\
Look at this document page image.
Classify it as exactly one word:
- SIMPLE: if it contains mainly plain text paragraphs, lists, or simple layouts
- COMPLEX: if it contains charts, tables, diagrams, formulas, or mixed multi-column layouts

Answer with one word only: SIMPLE or COMPLEX"""


def run_classification(llm, image, label=None):
    """Run one classification inference, return result dict."""
    uri = image_to_base64_uri(image)

    t0 = time.perf_counter()
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": uri}},
                    {"type": "text", "text": CLASSIFY_PROMPT},
                ],
            }
        ],
        max_tokens=10,
        temperature=0.0,
    )
    latency = (time.perf_counter() - t0) * 1000

    raw = response["choices"][0]["message"]["content"].strip()
    # Normalize: extract first word
    word = raw.split()[0].upper() if raw else "?"
    result = "SIMPLE" if "SIMPLE" in word else "COMPLEX" if "COMPLEX" in word else f"?({raw[:20]})"

    ok = ""
    if label:
        ok = " ✓" if result == label else f" ✗ (expected {label})"

    print(f"  [{label or '?'}] → {result}{ok}  ({latency:.0f}ms)  raw='{raw}'")
    return {"result": result, "latency_ms": latency, "correct": result == label if label else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-threads", type=int, default=4, help="CPU threads")
    parser.add_argument("--n-ctx", type=int, default=2048, help="Context window")
    parser.add_argument("--image", type=str, default=None, help="Custom image path to classify")
    parser.add_argument("--n-runs", type=int, default=2, help="Runs per test image")
    args = parser.parse_args()

    print(f"\n=== Qwen3.5-0.8B Q2 VLM Benchmark ===")
    print(f"Model : {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / 1e6:.0f} MB)")
    print(f"mmproj: {MMPROJ_PATH.name} ({MMPROJ_PATH.stat().st_size / 1e6:.0f} MB)")
    print(f"Threads: {args.n_threads} | ctx: {args.n_ctx}\n")

    # Load model
    print("Loading model + mmproj...")
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Qwen25VLChatHandler

    t_load = time.perf_counter()
    chat_handler = Qwen25VLChatHandler(
        clip_model_path=str(MMPROJ_PATH),
        verbose=False,
    )
    llm = Llama(
        model_path=str(MODEL_PATH),
        chat_handler=chat_handler,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=0,  # CPU only
        verbose=False,
        logits_all=False,
    )
    load_ms = (time.perf_counter() - t_load) * 1000
    print(f"Load time: {load_ms:.0f}ms\n")

    # Test on synthetic images
    print("Generating test images...")
    test_images = make_test_images()

    print(f"\nClassification results ({args.n_runs} runs each):")
    latencies = []
    correct = 0
    total = 0

    for label, img in test_images.items():
        for _ in range(args.n_runs):
            r = run_classification(llm, img, label=label)
            latencies.append(r["latency_ms"])
            if r["correct"] is not None:
                total += 1
                if r["correct"]:
                    correct += 1

    # Custom image
    if args.image:
        from PIL import Image
        print(f"\nCustom image: {args.image}")
        img = Image.open(args.image)
        run_classification(llm, img)

    # Summary
    print(f"\n--- Summary ---")
    print(f"Accuracy  : {correct}/{total} ({100*correct/total:.0f}%)" if total else "")
    if latencies:
        avg = sum(latencies) / len(latencies)
        print(f"Latency   : avg={avg:.0f}ms  min={min(latencies):.0f}ms  max={max(latencies):.0f}ms")
    print(f"Load time : {load_ms:.0f}ms (one-time)")
    print(f"\nTotal GGUF: {(MODEL_PATH.stat().st_size + MMPROJ_PATH.stat().st_size) / 1e6:.0f} MB on disk")


if __name__ == "__main__":
    main()
