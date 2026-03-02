#!/usr/bin/env python3
"""
Benchmark: SmolVLM-256M-Instruct for SIMPLE/COMPLEX document classification.

SmolVLM-256M: 256M params, transformers native, document-understanding focused.
No extra deps — works with current stack (transformers 4.57.6).

Usage:
  python scripts/bench_smolvlm.py
  python scripts/bench_smolvlm.py --image path/to/doc.png
"""

import argparse
import io
import time
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

CLASSIFY_PROMPT = """\
Look at this document page image.
Classify it as exactly one word:
- SIMPLE: if it contains mainly plain text paragraphs, lists, or simple layouts
- COMPLEX: if it contains charts, tables, diagrams, formulas, or mixed multi-column layouts

Answer with one word only: SIMPLE or COMPLEX"""


def make_test_images():
    from PIL import Image, ImageDraw
    import numpy as np
    images = {}

    # SIMPLE: text lines
    img = Image.new("RGB", (400, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i in range(12):
        y = 20 + i * 22
        w = 300 + np.random.randint(-80, 50)
        draw.rectangle([20, y, 20 + w, y + 12], fill=(30, 30, 30))
    images["SIMPLE"] = img

    # COMPLEX: table + bar chart
    img2 = Image.new("RGB", (400, 300), color=(255, 255, 255))
    draw2 = ImageDraw.Draw(img2)
    for col in range(4):
        for row in range(6):
            x, y = 20 + col * 90, 20 + row * 40
            draw2.rectangle([x, y, x + 82, y + 32], outline=(0, 0, 0), width=1)
            draw2.rectangle([x + 4, y + 8, x + 60, y + 16], fill=(80, 80, 180))
    for i in range(8):
        h = 20 + i * 8
        x = 30 + i * 40
        draw2.rectangle([x, 260 - h, x + 28, 260], fill=(200, 100, 50))
    images["COMPLEX"] = img2

    return images


def run_classification(model, processor, image, label=None):
    import torch
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": CLASSIFY_PROMPT},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
        )
    latency = (time.perf_counter() - t0) * 1000

    # Decode only new tokens
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    raw = processor.decode(new_tokens, skip_special_tokens=True).strip()
    word = raw.split()[0].upper() if raw else "?"
    result = "SIMPLE" if "SIMPLE" in word else "COMPLEX" if "COMPLEX" in word else f"?({raw[:20]})"

    ok = ""
    if label:
        ok = " ✓" if result == label else f" ✗ (expected {label})"
    print(f"  [{label or '?'}] → {result}{ok}  ({latency:.0f}ms)  raw='{raw}'")
    return {"result": result, "latency_ms": latency, "correct": result == label if label else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--n-runs", type=int, default=1)
    args = parser.parse_args()

    print(f"\n=== SmolVLM-256M Benchmark ===")
    print(f"Model: {args.model_id}\n")

    from transformers import SmolVLMForConditionalGeneration, SmolVLMProcessor
    import torch

    print("Loading model (CPU, float32)...")
    t_load = time.perf_counter()
    processor = SmolVLMProcessor.from_pretrained(args.model_id)
    model = SmolVLMForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()
    load_ms = (time.perf_counter() - t_load) * 1000
    print(f"Load time: {load_ms:.0f}ms\n")

    # RAM usage
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Model params RAM: {param_bytes / 1e6:.0f} MB\n")

    # Warmup
    print("Warmup pass...")
    test_images = make_test_images()
    run_classification(model, processor, next(iter(test_images.values())))

    print(f"\nClassification results ({args.n_runs} runs each):")
    latencies = []
    correct = 0
    total = 0
    for label, img in test_images.items():
        for _ in range(args.n_runs):
            r = run_classification(model, processor, img, label=label)
            latencies.append(r["latency_ms"])
            if r["correct"] is not None:
                total += 1
                if r["correct"]:
                    correct += 1

    if args.image:
        from PIL import Image
        print(f"\nCustom image: {args.image}")
        img = Image.open(args.image)
        run_classification(model, processor, img)

    print(f"\n--- Summary ---")
    if total:
        print(f"Accuracy  : {correct}/{total} ({100*correct/total:.0f}%)")
    if latencies:
        avg = sum(latencies) / len(latencies)
        print(f"Latency   : avg={avg:.0f}ms  min={min(latencies):.0f}ms  max={max(latencies):.0f}ms")
    print(f"Load time : {load_ms:.0f}ms (one-time)")
    print(f"RAM       : {param_bytes / 1e6:.0f} MB model params")


if __name__ == "__main__":
    main()
