#!/usr/bin/env python3
"""
VLM Benchmark for Document OCR
Tests: InternVL2-2B, Moondream2, LFM2.5-VL-1.6B

Usage:
    python vlm_benchmark.py [--models MODEL1,MODEL2] [--images-dir PATH]

Requirements:
    pip install torch torchvision transformers accelerate bitsandbytes pillow
    pip install flash-attn --no-build-isolation  # Optional, for InternVL2

Benchmark Results:
    - InternVL2-2B: 86.9% DocVQA, 784 OCRBench (GPU required, 4-5GB VRAM)
    - Moondream2: 79.3% DocVQA, 61.2 OCRBench (~3GB VRAM)
    - LFM2.5-VL-1.6B: 41.4% OCRBench v2, CPU-friendly (edge-first design)
"""
import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

# Constants for InternVL2 image processing
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    """Build transform for InternVL2 image preprocessing."""
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find closest aspect ratio for dynamic preprocessing."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Dynamic preprocess for InternVL2 - handles aspect ratio and creates tiles."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate target ratios
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find closest aspect ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # Calculate target dimensions
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize and split
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def load_image_internvl(image_path: str, input_size: int = 448, max_num: int = 12):
    """Load and preprocess image for InternVL2."""
    image = Image.open(image_path).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# Prompts for testing different capabilities
PROMPTS = {
    "ocr_simple": "Extract all text from this document.",
    "ocr_structured": "Extract the text and preserve the structure (tables, lists, formatting).",
    "classify": "Is this document SIMPLE (plain text, single column) or COMPLEX (tables, forms, multi-column)? Answer with just SIMPLE or COMPLEX.",
    "summary": "Summarize the main content of this document in 2-3 sentences.",
}


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    model_name: str
    image_name: str
    prompt_name: str
    response: str
    latency: float
    error: Optional[str] = None


@dataclass
class ModelBenchmark:
    """Benchmark results for a model."""
    model_name: str
    results: list = field(default_factory=list)
    avg_latency: float = 0.0
    load_time: float = 0.0
    vram_used_mb: float = 0.0
    error: Optional[str] = None

    def add_result(self, result: BenchmarkResult):
        self.results.append(result)

    def compute_avg_latency(self):
        valid_results = [r for r in self.results if r.error is None]
        if valid_results:
            self.avg_latency = sum(r.latency for r in valid_results) / len(valid_results)


class VLMBenchmark:
    """Benchmark multiple Vision-Language Models for document OCR."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.results: dict[str, ModelBenchmark] = {}

        print(f"Device: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    def _get_vram_used(self) -> float:
        """Get current VRAM usage in MB."""
        if self.device == "cuda":
            return torch.cuda.memory_allocated() / 1e6
        return 0.0

    def _cleanup(self):
        """Cleanup GPU memory."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # =========================================================================
    # Model Loaders
    # =========================================================================

    def load_internvl2(self, quantize_4bit: bool = False) -> tuple:
        """Load InternVL2-2B model."""
        print("\n[InternVL2-2B] Loading...")
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }

        if quantize_4bit and self.device == "cuda":
            # Use BitsAndBytesConfig for 4-bit quantization (transformers >= 4.50)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = {"": self.device}

        # Try to use flash attention
        try:
            import flash_attn
            model_kwargs["use_flash_attn"] = True
            print("  Using Flash Attention")
        except ImportError:
            print("  Flash Attention not available")

        model = AutoModel.from_pretrained(
            "OpenGVLab/InternVL2-2B",
            **model_kwargs
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            "OpenGVLab/InternVL2-2B",
            trust_remote_code=True,
            use_fast=False
        )

        print(f"  Loaded! VRAM: {self._get_vram_used():.0f} MB")
        return model, tokenizer

    def load_moondream2(self) -> tuple:
        """Load Moondream2 model."""
        print("\n[Moondream2] Loading...")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Try different revisions - some require pyvips, some don't
        revisions_to_try = ["2024-08-26", "2024-07-23", "2024-05-20"]

        for revision in revisions_to_try:
            try:
                print(f"  Trying revision {revision}...")
                model = AutoModelForCausalLM.from_pretrained(
                    "vikhyatk/moondream2",
                    revision=revision,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map={"": self.device}
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    "vikhyatk/moondream2",
                    revision=revision
                )
                print(f"  Loaded! VRAM: {self._get_vram_used():.0f} MB")
                return model, tokenizer
            except Exception as e:
                if "pyvips" in str(e) or "libvips" in str(e):
                    print(f"  Revision {revision} requires pyvips, trying next...")
                    continue
                raise

        raise RuntimeError("All Moondream2 revisions require pyvips. Install with: sudo apt install libvips-dev")

    def load_lfm25_vl(self) -> tuple:
        """Load LFM2.5-VL-1.6B model."""
        print("\n[LFM2.5-VL-1.6B] Loading...")
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model = AutoModelForImageTextToText.from_pretrained(
            "LiquidAI/LFM2.5-VL-1.6B",
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map={"": self.device} if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        if self.device == "cpu":
            model = model.to(self.device)

        processor = AutoProcessor.from_pretrained(
            "LiquidAI/LFM2.5-VL-1.6B",
            trust_remote_code=True,
        )

        print(f"  Loaded! VRAM: {self._get_vram_used():.0f} MB")
        return model, processor

    # =========================================================================
    # Benchmark Functions
    # =========================================================================

    def benchmark_internvl2(
        self, model: Any, tokenizer: Any, image_path: str, prompt: str
    ) -> tuple[str, float]:
        """Benchmark InternVL2-2B on a single image."""
        # Load and preprocess image using InternVL2 preprocessing
        pixel_values = load_image_internvl(image_path, max_num=6)  # Limit tiles for speed
        pixel_values = pixel_values.to(model.device, dtype=model.dtype)

        question = f'<image>\n{prompt}'

        start_time = time.time()
        generation_config = dict(max_new_tokens=512, do_sample=False)
        response = model.chat(tokenizer, pixel_values, question, generation_config)
        latency = time.time() - start_time

        return response, latency

    def benchmark_moondream2(
        self, model: Any, tokenizer: Any, image_path: str, prompt: str
    ) -> tuple[str, float]:
        """Benchmark Moondream2 on a single image."""
        image = Image.open(image_path).convert('RGB')

        start_time = time.time()

        # Try different API methods depending on model revision
        if hasattr(model, 'query'):
            result = model.query(image, prompt)
            response = result.get("answer", str(result)) if isinstance(result, dict) else str(result)
        elif hasattr(model, 'answer_question'):
            # Older revisions use answer_question with separate tokenizer
            enc_image = model.encode_image(image)
            response = model.answer_question(enc_image, prompt, tokenizer)
        else:
            raise AttributeError("Moondream2 model has no known query method")

        latency = time.time() - start_time
        return response, latency

    def benchmark_lfm25_vl(
        self, model: Any, processor: Any, image_path: str, prompt: str
    ) -> tuple[str, float]:
        """Benchmark LFM2.5-VL-1.6B on a single image."""
        image = Image.open(image_path).convert('RGB')

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

        start_time = time.time()
        outputs = model.generate(**inputs, max_new_tokens=512)
        latency = time.time() - start_time

        response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return response, latency

    # =========================================================================
    # Main Benchmark Runner
    # =========================================================================

    def run_single_model(
        self,
        model_name: str,
        test_images: list[str],
        load_fn: Callable,
        benchmark_fn: Callable,
    ) -> ModelBenchmark:
        """Run benchmark for a single model."""
        benchmark = ModelBenchmark(model_name=model_name)

        try:
            # Load model
            load_start = time.time()
            model, tokenizer = load_fn()
            benchmark.load_time = time.time() - load_start
            benchmark.vram_used_mb = self._get_vram_used()

            # Run benchmarks
            for img_path in test_images:
                img_name = Path(img_path).name

                for prompt_name, prompt in PROMPTS.items():
                    print(f"  [{img_name}] {prompt_name}...", end=" ", flush=True)

                    try:
                        response, latency = benchmark_fn(model, tokenizer, img_path, prompt)
                        result = BenchmarkResult(
                            model_name=model_name,
                            image_name=img_name,
                            prompt_name=prompt_name,
                            response=response[:1000] if len(response) > 1000 else response,
                            latency=latency,
                        )
                        print(f"{latency:.2f}s")
                    except Exception as e:
                        result = BenchmarkResult(
                            model_name=model_name,
                            image_name=img_name,
                            prompt_name=prompt_name,
                            response="",
                            latency=0.0,
                            error=str(e),
                        )
                        print(f"ERROR: {e}")

                    benchmark.add_result(result)

            benchmark.compute_avg_latency()

            # Cleanup
            del model
            if tokenizer:
                del tokenizer
            self._cleanup()

        except Exception as e:
            benchmark.error = str(e)
            print(f"  ERROR loading model: {e}")

        return benchmark

    def run_benchmark(
        self,
        test_images: list[str],
        models_to_test: Optional[list[str]] = None,
    ) -> dict[str, ModelBenchmark]:
        """Run full benchmark on all specified models."""
        if models_to_test is None:
            models_to_test = ["internvl2", "moondream2", "lfm25_vl"]

        model_configs = {
            "internvl2": (self.load_internvl2, self.benchmark_internvl2),
            "moondream2": (self.load_moondream2, self.benchmark_moondream2),
            "lfm25_vl": (self.load_lfm25_vl, self.benchmark_lfm25_vl),
        }

        for model_name in models_to_test:
            if model_name not in model_configs:
                print(f"Unknown model: {model_name}")
                continue

            print(f"\n{'='*60}")
            print(f"Testing: {model_name}")
            print(f"{'='*60}")

            load_fn, benchmark_fn = model_configs[model_name]
            benchmark = self.run_single_model(
                model_name=model_name,
                test_images=test_images,
                load_fn=load_fn,
                benchmark_fn=benchmark_fn,
            )
            self.results[model_name] = benchmark

        return self.results

    def save_results(self, output_path: str):
        """Save benchmark results to JSON."""
        output = {}
        for name, benchmark in self.results.items():
            output[name] = {
                "load_time": benchmark.load_time,
                "avg_latency": benchmark.avg_latency,
                "vram_used_mb": benchmark.vram_used_mb,
                "error": benchmark.error,
                "results": [
                    {
                        "image": r.image_name,
                        "prompt": r.prompt_name,
                        "response": r.response[:500] + "..." if len(r.response) > 500 else r.response,
                        "latency": r.latency,
                        "error": r.error,
                    }
                    for r in benchmark.results
                ],
            }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")

    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print(f"{'Model':<15} {'Load Time':<12} {'Avg Latency':<12} {'VRAM (MB)':<12} {'Status'}")
        print("-"*60)

        for name, benchmark in self.results.items():
            if benchmark.error:
                status = f"ERROR: {benchmark.error[:30]}"
            else:
                status = "OK"

            print(
                f"{name:<15} "
                f"{benchmark.load_time:>8.2f}s    "
                f"{benchmark.avg_latency:>8.2f}s    "
                f"{benchmark.vram_used_mb:>8.0f}    "
                f"{status}"
            )


def main():
    parser = argparse.ArgumentParser(description="Benchmark VLMs for Document OCR")
    parser.add_argument(
        "--models",
        type=str,
        default="internvl2,moondream2,lfm25_vl",
        help="Comma-separated list of models to test (default: all)",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="./test_documents",
        help="Directory containing test images (default: ./test_documents)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="vlm_benchmark_results.json",
        help="Output JSON file (default: vlm_benchmark_results.json)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)",
    )
    args = parser.parse_args()

    # Find test images
    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        images_dir.mkdir(parents=True)
        print(f"Created {images_dir}")
        print("Please add test document images (.png, .jpg, .jpeg)")
        print("\nExample documents to add:")
        print("  - facture.png (invoice)")
        print("  - formulaire.png (form with fields)")
        print("  - ticket.jpg (receipt)")
        print("  - document_tableau.png (document with tables)")
        return

    test_images = (
        list(images_dir.glob("*.png"))
        + list(images_dir.glob("*.jpg"))
        + list(images_dir.glob("*.jpeg"))
    )

    if not test_images:
        print(f"No images found in {images_dir}")
        print("Supported formats: .png, .jpg, .jpeg")
        return

    print(f"Found {len(test_images)} test images:")
    for img in test_images:
        print(f"  - {img.name}")

    # Parse models
    models = [m.strip() for m in args.models.split(",")]

    # Run benchmark
    benchmark = VLMBenchmark(device=args.device)
    benchmark.run_benchmark([str(p) for p in test_images], models_to_test=models)

    # Save and print results
    benchmark.save_results(args.output)
    benchmark.print_summary()


if __name__ == "__main__":
    main()
