"""
CLIP ViT-B/32 ONNX Zero-Shot Document Complexity Classifier
============================================================

Classifies document pages as SIMPLE or COMPLEX using CLIP zero-shot inference.

Models (Xenova/clip-vit-base-patch32, uint8):
  - vision_model_uint8.onnx : 85 MB  (runs per image, ~20ms CPU)
  - text_model_uint8.onnx   : 62 MB  (runs ONCE at startup, labels cached)

Flow:
  1. Startup : encode text labels → cached text embeddings
  2. Per image: preprocess → vision ONNX → cosine sim → SIMPLE/COMPLEX

Benchmark results: avg=20.5ms, 100% accuracy on synthetic test set.
"""

from __future__ import annotations

import io
import json
import time
import base64
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models/clip-vit-base-patch32"

LABELS: Dict[str, str] = {
    "SIMPLE":  "a simple plain text document page with paragraphs and lines of text",
    "COMPLEX": "a complex document page with charts, tables, diagrams, formulas or mixed layouts",
}


# ── Tokenizer (minimal BPE, no transformers dependency) ──────────────────────

def _load_tokenizer(model_dir: Path):
    """Load BPE tokenizer from tokenizer.json (no transformers needed)."""
    with open(model_dir / "tokenizer.json") as f:
        tok = json.load(f)

    vocab    = tok["model"]["vocab"]
    merges   = tok["model"]["merges"]

    merge_ranks = {tuple(m.split()): i for i, m in enumerate(merges)}

    def bpe(word):
        chars = list(word)
        while len(chars) > 1:
            pairs = [(chars[i], chars[i + 1]) for i in range(len(chars) - 1)]
            best  = min(pairs, key=lambda p: merge_ranks.get(p, float("inf")))
            if best not in merge_ranks:
                break
            first, second = best
            new, i = [], 0
            while i < len(chars):
                if i < len(chars) - 1 and chars[i] == first and chars[i + 1] == second:
                    new.append(first + second)
                    i += 2
                else:
                    new.append(chars[i])
                    i += 1
            chars = new
        return chars

    sot = vocab.get("<|startoftext|>", 49406)
    eot = vocab.get("<|endoftext|>",   49407)
    unk = eot

    def tokenize(text: str, max_len: int = 77) -> np.ndarray:
        text   = text.lower().strip()
        tokens = [sot]
        for word in text.split():
            for t in bpe(word + "</w>"):
                tokens.append(vocab.get(t, unk))
        tokens.append(eot)
        if len(tokens) > max_len:
            tokens = tokens[:max_len - 1] + [eot]
        ids = tokens + [0] * (max_len - len(tokens))
        return np.array([ids], dtype=np.int64)

    return tokenize


# ── Image preprocessor ───────────────────────────────────────────────────────

def _preprocess_image(image, size: int = 224) -> np.ndarray:
    """Resize + center-crop + CLIP normalize → [1, 3, H, W] float32."""
    from PIL import Image as PILImage

    if not isinstance(image, PILImage.Image):
        image = PILImage.open(io.BytesIO(image))
    image = image.convert("RGB")

    w, h  = image.size
    scale = size / min(w, h)
    image = image.resize((round(w * scale), round(h * scale)), PILImage.BICUBIC)

    w, h  = image.size
    left  = (w - size) // 2
    top   = (h - size) // 2
    image = image.crop((left, top, left + size, top + size))

    arr  = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275,  0.40821073], dtype=np.float32)
    std  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    arr  = (arr - mean) / std
    return arr.transpose(2, 0, 1)[np.newaxis]  # [1, 3, 224, 224]


# ── Classifier ────────────────────────────────────────────────────────────────

class ClipZeroShotClassifier:
    """
    CLIP ViT-B/32 ONNX zero-shot document complexity classifier.

    Text labels are encoded ONCE at startup and cached.
    Only the vision encoder runs per image (~20ms on CPU).
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        labels: Optional[Dict[str, str]] = None,
        n_threads: int = 4,
    ):
        import onnxruntime as ort

        self.model_dir = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
        self.labels    = labels or LABELS

        vision_onnx = self.model_dir / "onnx/vision_model_uint8.onnx"
        text_onnx   = self.model_dir / "onnx/text_model_uint8.onnx"

        for p in (vision_onnx, text_onnx):
            if not p.exists():
                raise FileNotFoundError(f"CLIP ONNX model not found: {p}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads       = n_threads
        opts.graph_optimization_level   = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_cpu_mem_arena       = False

        logger.info(f"Loading CLIP vision model ({vision_onnx.stat().st_size / 1e6:.0f} MB)...")
        self._vision_sess = ort.InferenceSession(str(vision_onnx), opts)

        logger.info(f"Loading CLIP text model ({text_onnx.stat().st_size / 1e6:.0f} MB)...")
        self._text_sess   = ort.InferenceSession(str(text_onnx), opts)

        self._tokenize = _load_tokenizer(self.model_dir)

        # Pre-compute and cache text embeddings (done ONCE)
        logger.info(f"Encoding {len(self.labels)} text labels (cached)...")
        self._text_embs: Dict[str, np.ndarray] = {}
        for name, text in self.labels.items():
            ids = self._tokenize(text)
            out = self._text_sess.run(None, {"input_ids": ids})
            emb = out[0][0]  # [512]
            self._text_embs[name] = emb / np.linalg.norm(emb)

        logger.info("✅ CLIP zero-shot classifier ready")

    def predict(self, image) -> Dict:
        """
        Classify a document image as SIMPLE or COMPLEX.

        Args:
            image: PIL.Image, bytes/bytearray, or base64 str (with or without data: prefix)

        Returns:
            {
                'class_name':    'SIMPLE' | 'COMPLEX',
                'confidence':    float,
                'probabilities': {'SIMPLE': float, 'COMPLEX': float},
                'raw_scores':    {'SIMPLE': float, 'COMPLEX': float},
                'latency_ms':    float,
            }
        """
        from PIL import Image as PILImage

        if isinstance(image, str):
            if image.startswith("data:"):
                image = image.split(",", 1)[1]
            image = PILImage.open(io.BytesIO(base64.b64decode(image)))
        elif isinstance(image, (bytes, bytearray)):
            image = PILImage.open(io.BytesIO(image))

        t0 = time.perf_counter()

        pixel_values = _preprocess_image(image)
        vis_out      = self._vision_sess.run(None, {"pixel_values": pixel_values})
        img_emb      = vis_out[0][0]
        img_emb      = img_emb / np.linalg.norm(img_emb)

        raw_scores: Dict[str, float] = {
            name: float(np.dot(img_emb, text_emb))
            for name, text_emb in self._text_embs.items()
        }

        latency_ms = (time.perf_counter() - t0) * 1000

        # Softmax (scale ×100 like CLIP logit_scale)
        logits = np.array(list(raw_scores.values())) * 100
        probs  = np.exp(logits - logits.max())
        probs /= probs.sum()
        probabilities = {k: float(p) for k, p in zip(raw_scores, probs)}

        best = max(probabilities, key=probabilities.get)

        return {
            "class_name":    best,
            "confidence":    probabilities[best],
            "probabilities": probabilities,
            "raw_scores":    raw_scores,
            "latency_ms":    latency_ms,
        }
