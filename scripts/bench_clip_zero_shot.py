#!/usr/bin/env python3
"""
Benchmark: CLIP ViT-B/32 ONNX (uint8) zero-shot document complexity classification.

Flow:
  1. At startup: encode labels → text embeddings (once, cached)
  2. Per image:  encode image → image embedding → cosine similarity → label

Models (Xenova/clip-vit-base-patch32):
  - vision_model_uint8.onnx : 85 MB  (runs per image)
  - text_model_uint8.onnx   : 62 MB  (runs once at startup)

Input: base64 image OR PIL Image
Output: "SIMPLE" / "COMPLEX" + confidence score

Usage:
  python scripts/bench_clip_zero_shot.py
  python scripts/bench_clip_zero_shot.py --image path/to/doc.png
"""

import argparse
import base64
import io
import json
import time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models/clip-vit-base-patch32"
VISION_ONNX = MODEL_DIR / "onnx/vision_model_uint8.onnx"
TEXT_ONNX   = MODEL_DIR / "onnx/text_model_uint8.onnx"

# ── Labels ──────────────────────────────────────────────────────────────────
# Longer, more descriptive labels tend to work better with CLIP
LABELS = {
    "SIMPLE":  "a simple plain text document page with paragraphs and lines of text",
    "COMPLEX": "a complex document page with charts, tables, diagrams, formulas or mixed layouts",
}


# ── Tokenizer (minimal, no external dep) ────────────────────────────────────

def load_tokenizer(model_dir: Path):
    """Load BPE tokenizer from tokenizer.json (no transformers needed)."""
    with open(model_dir / "tokenizer.json") as f:
        tok = json.load(f)

    vocab    = tok["model"]["vocab"]
    merges   = tok["model"]["merges"]
    id2token = {v: k for k, v in vocab.items()}

    # Build merge priority map
    merge_ranks = {tuple(m.split()): i for i, m in enumerate(merges)}

    def bpe(word):
        chars = list(word)
        while len(chars) > 1:
            pairs = [(chars[i], chars[i+1]) for i in range(len(chars)-1)]
            best  = min(pairs, key=lambda p: merge_ranks.get(p, float("inf")))
            if best not in merge_ranks:
                break
            first, second = best
            new = []
            i = 0
            while i < len(chars):
                if i < len(chars)-1 and chars[i] == first and chars[i+1] == second:
                    new.append(first + second)
                    i += 2
                else:
                    new.append(chars[i])
                    i += 1
            chars = new
        return chars

    sot = vocab.get("<|startoftext|>", 49406)
    eot = vocab.get("<|endoftext|>",   49407)
    unk = vocab.get("<|endoftext|>",   49407)

    def tokenize(text: str, max_len: int = 77):
        text = text.lower().strip()
        # Simple whitespace split + BPE
        tokens = [sot]
        for word in text.split():
            word_bytes = word + "</w>"
            word_tokens = bpe(word_bytes)
            for t in word_tokens:
                tokens.append(vocab.get(t, unk))
        tokens.append(eot)
        # Pad / truncate to max_len
        if len(tokens) > max_len:
            tokens = tokens[:max_len-1] + [eot]
        ids = tokens + [0] * (max_len - len(tokens))
        return np.array([ids], dtype=np.int64)

    return tokenize


# ── Image preprocessor ───────────────────────────────────────────────────────

def preprocess_image(image, size=224):
    """Resize + center-crop + normalize → [1, 3, H, W] float32."""
    from PIL import Image as PILImage
    if not isinstance(image, PILImage.Image):
        image = PILImage.open(io.BytesIO(image))
    image = image.convert("RGB")

    # Resize shortest side to size
    w, h = image.size
    scale = size / min(w, h)
    image = image.resize((round(w*scale), round(h*scale)), PILImage.BICUBIC)

    # Center crop
    w, h = image.size
    left = (w - size) // 2
    top  = (h - size) // 2
    image = image.crop((left, top, left+size, top+size))

    arr = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275,  0.40821073], dtype=np.float32)
    std  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1)[np.newaxis]  # [1, 3, 224, 224]


# ── CLIP zero-shot classifier ────────────────────────────────────────────────

class ClipZeroShotClassifier:
    def __init__(self, labels=LABELS, n_threads=4):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = n_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_cpu_mem_arena = False

        print(f"  Loading vision model ({VISION_ONNX.stat().st_size/1e6:.0f} MB)...")
        self.vision_sess = ort.InferenceSession(str(VISION_ONNX), opts)

        print(f"  Loading text model  ({TEXT_ONNX.stat().st_size/1e6:.0f} MB)...")
        self.text_sess   = ort.InferenceSession(str(TEXT_ONNX), opts)

        self.tokenize = load_tokenizer(MODEL_DIR)
        self.labels   = labels

        # Pre-compute text embeddings (done ONCE)
        print(f"  Encoding {len(labels)} text labels...")
        self.text_embs = {}
        for name, text in labels.items():
            ids = self.tokenize(text)
            out = self.text_sess.run(None, {"input_ids": ids})
            emb = out[0][0]  # [512]
            self.text_embs[name] = emb / np.linalg.norm(emb)

    def predict(self, image, return_scores=False):
        """
        image: PIL.Image or bytes or base64 str

        Returns dict with class_name, confidence, scores, latency_ms
        """
        from PIL import Image as PILImage

        # Decode input
        if isinstance(image, str):
            if image.startswith("data:"):
                image = image.split(",", 1)[1]
            image = PILImage.open(io.BytesIO(base64.b64decode(image)))
        elif isinstance(image, (bytes, bytearray)):
            image = PILImage.open(io.BytesIO(image))

        t0 = time.perf_counter()

        # Vision encoding
        pixel_values = preprocess_image(image)
        vis_out = self.vision_sess.run(None, {"pixel_values": pixel_values})
        img_emb = vis_out[0][0]  # [512]
        img_emb = img_emb / np.linalg.norm(img_emb)

        # Cosine similarity → softmax scores
        scores = {}
        for name, text_emb in self.text_embs.items():
            scores[name] = float(np.dot(img_emb, text_emb))

        latency = (time.perf_counter() - t0) * 1000

        # Softmax over logit scores (scale by 100 like CLIP does)
        logits  = np.array(list(scores.values())) * 100
        probs   = np.exp(logits - logits.max())
        probs  /= probs.sum()
        probs_d = {k: float(p) for k, p in zip(scores, probs)}

        best = max(probs_d, key=probs_d.get)
        return {
            "class_name":  best,
            "confidence":  probs_d[best],
            "probabilities": probs_d,
            "raw_scores":  scores,
            "latency_ms":  latency,
        }


# ── Synthetic test images ────────────────────────────────────────────────────

def make_test_images():
    from PIL import Image, ImageDraw
    images = {}

    # SIMPLE: dense text lines
    img = Image.new("RGB", (600, 800), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i in range(32):
        y = 30 + i * 22
        w = 520 + (i % 3) * 20 - 30
        draw.rectangle([40, y, 40+w, y+10], fill=(20, 20, 20))
        if i % 8 == 7:  # paragraph break
            pass
    images["SIMPLE"] = img

    # COMPLEX: table + bar chart
    img2 = Image.new("RGB", (600, 800), (255, 255, 255))
    draw2 = ImageDraw.Draw(img2)
    # Table (5 cols x 8 rows)
    for col in range(5):
        for row in range(8):
            x, y = 30 + col*110, 30 + row*45
            draw2.rectangle([x, y, x+104, y+40], outline=(0, 0, 0), width=1)
            if row == 0:
                draw2.rectangle([x+2, y+2, x+102, y+38], fill=(60, 60, 200))
            else:
                draw2.rectangle([x+4, y+12, x+80+col*4, y+22], fill=(100, 100, 180))
    # Bar chart
    for i in range(10):
        h = 40 + i * 15
        x = 30 + i*52
        draw2.rectangle([x, 430-h, x+40, 430], fill=(200, 80, 50))
        draw2.line([30, 430, 560, 430], fill=(0, 0, 0), width=2)
    # Pie chart approximation (arcs)
    draw2.ellipse([50, 460, 250, 660], outline=(0,0,0), width=2)
    draw2.chord([50, 460, 250, 660], 0, 120, fill=(255, 100, 100))
    draw2.chord([50, 460, 250, 660], 120, 260, fill=(100, 200, 100))
    images["COMPLEX"] = img2

    return images


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--n-runs",    type=int, default=3)
    parser.add_argument("--image",     type=str, default=None, help="Path or base64 image to classify")
    args = parser.parse_args()

    print(f"\n=== CLIP ViT-B/32 ONNX uint8 — Zero-Shot Classifier ===")
    print(f"Threads: {args.n_threads}\n")

    print("Loading models...")
    t0 = time.perf_counter()
    clf = ClipZeroShotClassifier(n_threads=args.n_threads)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"Load time (vision+text+label encoding): {load_ms:.0f}ms\n")

    # Warmup
    from PIL import Image
    dummy = Image.new("RGB", (224, 224), (128, 128, 128))
    clf.predict(dummy)

    # Synthetic tests
    test_images = make_test_images()
    print(f"Classification ({args.n_runs} runs each):")
    latencies = []
    correct = 0
    total   = 0

    for label, img in test_images.items():
        run_lats = []
        result_label = None
        for _ in range(args.n_runs):
            r = clf.predict(img)
            run_lats.append(r["latency_ms"])
            result_label = r["class_name"]
            confidence   = r["confidence"]

        avg_lat = sum(run_lats) / len(run_lats)
        ok = "✓" if result_label == label else f"✗ (expected {label})"
        scores_str = "  ".join(f"{k}={v:.3f}" for k,v in r["raw_scores"].items())
        print(f"  [{label:7s}] → {result_label:7s} {ok}  conf={confidence:.1%}  avg={avg_lat:.1f}ms  [{scores_str}]")
        latencies.extend(run_lats)
        total += 1
        if result_label == label:
            correct += 1

    # Custom image
    if args.image:
        print(f"\nCustom image: {args.image}")
        r = clf.predict(args.image)
        print(f"  → {r['class_name']} ({r['confidence']:.1%})  {r['latency_ms']:.1f}ms")
        for k, v in r["probabilities"].items():
            print(f"     {k}: {v:.1%}")

    # Summary
    print(f"\n{'─'*55}")
    print(f"Accuracy   : {correct}/{total} ({100*correct/total:.0f}%)")
    avg = sum(latencies) / len(latencies)
    print(f"Latency    : avg={avg:.1f}ms  min={min(latencies):.1f}ms  max={max(latencies):.1f}ms  (vision only)")
    print(f"Load time  : {load_ms:.0f}ms  (one-time, text labels cached)")
    print(f"Disk       : {(VISION_ONNX.stat().st_size + TEXT_ONNX.stat().st_size)/1e6:.0f} MB total")
    print(f"\nText encoder runs ONCE → text embeddings cached.")
    print(f"Per-image cost = vision encoder only ({VISION_ONNX.stat().st_size/1e6:.0f} MB).")


if __name__ == "__main__":
    main()
