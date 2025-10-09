#!/usr/bin/env python3

import requests
import time
import json

# Colors
GREEN = '\033[0;32m'
BLUE = '\033[0;34m'
YELLOW = '\033[1;33m'
NC = '\033[0m'

print("🔬 Embedding Models Benchmark")
print("=" * 50)
print()

# Test TurboX.v2
print(f"{BLUE}📊 Test 1: Single Embedding Latency{NC}")
print("=" * 50)
print()

print(f"{YELLOW}🚀 TurboX.v2 (localhost:11435){NC}")
start = time.time()
response = requests.post(
    "http://localhost:11435/api/embed",
    json={"model": "turbov2", "input": "semantic search test"}
)
turbov2_latency = int((time.time() - start) * 1000)
turbov2_data = response.json()
turbov2_dim = len(turbov2_data['embeddings'][0])
print(f"  ✅ Latency: {turbov2_latency}ms")
print(f"  ✅ Dimensions: {turbov2_dim}")
print()

print(f"{YELLOW}🐳 Qwen3-Embedding:0.6b (172.20.0.11:11434){NC}")
start = time.time()
response = requests.post(
    "http://172.20.0.11:11434/api/embed",
    json={"model": "qwen3-embedding:0.6b", "input": "semantic search test"}
)
qwen_latency = int((time.time() - start) * 1000)
qwen_data = response.json()
qwen_dim = len(qwen_data['embeddings'][0])
print(f"  ✅ Latency: {qwen_latency}ms")
print(f"  ✅ Dimensions: {qwen_dim}")
print()

speedup = qwen_latency / turbov2_latency if turbov2_latency > 0 else 0
print(f"{GREEN}🏆 TurboX.v2 is {speedup:.1f}x faster for single embeddings{NC}")
print()

# Test batch embeddings
print("=" * 50)
print(f"{BLUE}📊 Test 2: Batch Embeddings (5 texts){NC}")
print("=" * 50)
print()

batch_texts = [
    "semantic search",
    "vector database",
    "machine learning",
    "neural networks",
    "deep learning"
]

print(f"{YELLOW}🚀 TurboX.v2 (batch){NC}")
start = time.time()
response = requests.post(
    "http://localhost:11435/api/embed",
    json={"model": "turbov2", "input": batch_texts}
)
turbov2_batch_latency = int((time.time() - start) * 1000)
turbov2_batch_count = len(response.json()['embeddings'])
print(f"  ✅ Total latency: {turbov2_batch_latency}ms")
print(f"  ✅ Embeddings: {turbov2_batch_count}")
print(f"  ✅ Per embedding: {turbov2_batch_latency // turbov2_batch_count}ms avg")
print()

print(f"{YELLOW}🐳 Qwen3-Embedding:0.6b (sequential){NC}")
start = time.time()
for text in batch_texts:
    requests.post(
        "http://172.20.0.11:11434/api/embed",
        json={"model": "qwen3-embedding:0.6b", "input": text}
    )
qwen_batch_latency = int((time.time() - start) * 1000)
print(f"  ✅ Total latency: {qwen_batch_latency}ms")
print(f"  ✅ Embeddings: 5")
print(f"  ✅ Per embedding: {qwen_batch_latency // 5}ms avg")
print()

batch_speedup = qwen_batch_latency / turbov2_batch_latency if turbov2_batch_latency > 0 else 0
print(f"{GREEN}🏆 TurboX.v2 is {batch_speedup:.1f}x faster for batch embeddings{NC}")
print()

# Summary
print("=" * 50)
print(f"{GREEN}📋 Summary{NC}")
print("=" * 50)
print()
print("Single embedding:")
print(f"  - TurboX.v2:    {turbov2_latency}ms ({turbov2_dim}D)")
print(f"  - Qwen3:        {qwen_latency}ms ({qwen_dim}D)")
print()
print("Batch (5 texts):")
print(f"  - TurboX.v2:    {turbov2_batch_latency}ms total (~{turbov2_batch_latency // turbov2_batch_count}ms/embedding)")
print(f"  - Qwen3:        {qwen_batch_latency}ms total (~{qwen_batch_latency // 5}ms/embedding)")
print()

print(f"{GREEN}✅ Benchmark complete!{NC}")
print()

if speedup > 10:
    print(f"💡 TurboX.v2 is {speedup:.1f}x faster - highly recommended for CPU-only Railway")
elif speedup > 5:
    print(f"💡 TurboX.v2 shows {speedup:.1f}x improvement - good for CPU workloads")
else:
    print(f"💡 TurboX.v2 provides {speedup:.1f}x speedup - evaluate based on use case")

print()
print("Next steps:")
print("1. Test embedding quality for your specific use case")
print("2. Configure N8N to use TurboX.v2:")
print("   - Base URL: http://deposium-embeddings-turbov2-test:11435")
print("   - Model: turbov2")
print("3. Deploy to Railway if performance is satisfactory")
