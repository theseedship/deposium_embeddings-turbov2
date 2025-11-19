import time
import torch
import numpy as np
from src.model_manager import get_model_manager

def benchmark_model(model_name, batch_size=1, num_batches=10):
    print(f"\nBenchmarking {model_name} (BS={batch_size})...")
    mm = get_model_manager()
    try:
        model = mm.get_model(model_name)
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return

    # Dummy data
    sentences = ["This is a test sentence for benchmarking purposes." * 5] * batch_size
    
    # Warmup
    print("Warming up...")
    if hasattr(model, 'encode'):
        model.encode(sentences[:1])
    
    # Benchmark
    print("Running inference...")
    start_time = time.time()
    for _ in range(num_batches):
        if hasattr(model, 'encode'):
            model.encode(sentences)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_batches
    throughput = (batch_size * num_batches) / total_time
    
    print(f"Results for {model_name}:")
    print(f"  Avg Latency (batch): {avg_time*1000:.2f} ms")
    print(f"  Throughput: {throughput:.2f} sentences/sec")
    
    # Memory
    used, free = mm.get_vram_usage_mb()
    print(f"  VRAM Used: {used} MB")

if __name__ == "__main__":
    # Test Qwen3 (Transformer)
    benchmark_model("qwen3-embed", batch_size=1)
    benchmark_model("qwen3-embed", batch_size=32)
    
    # Test Model2Vec (Static)
    benchmark_model("qwen25-1024d", batch_size=1)
    benchmark_model("qwen25-1024d", batch_size=32)
