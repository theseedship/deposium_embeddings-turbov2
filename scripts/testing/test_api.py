#!/usr/bin/env python3
"""Test l'API avec les modèles actuels (v11.0.0 architecture)"""
import requests
import json

API_URL = "http://localhost:11436"

print("="*60)
print("Testing Deposium Embeddings API (v11.0.0)")
print("="*60 + "\n")

# Test 1: GET /
print("1. Testing GET /")
response = requests.get(f"{API_URL}/")
print(json.dumps(response.json(), indent=2))
print()

# Test 2: GET /api/tags
print("2. Testing GET /api/tags")
response = requests.get(f"{API_URL}/api/tags")
print(json.dumps(response.json(), indent=2))
print()

# Test 3: POST /api/embed with m2v-bge-m3-1024d (PRIMARY)
print("3. Testing POST /api/embed with m2v-bge-m3-1024d (PRIMARY)")
response = requests.post(
    f"{API_URL}/api/embed",
    json={"model": "m2v-bge-m3-1024d", "input": "Hello world"}
)
result = response.json()
print(f"Model: {result['model']}")
print(f"Embeddings shape: {len(result['embeddings'])} x {len(result['embeddings'][0])}")
print()

# Test 4: POST /api/embed with bge-m3-onnx (CPU fallback)
print("4. Testing POST /api/embed with bge-m3-onnx (CPU)")
response = requests.post(
    f"{API_URL}/api/embed",
    json={"model": "bge-m3-onnx", "input": "Hello world"}
)
result = response.json()
print(f"Model: {result['model']}")
print(f"Embeddings shape: {len(result['embeddings'])} x {len(result['embeddings'][0])}")
print()

# Test 5: POST /api/embed with gemma-768d (legacy)
print("5. Testing POST /api/embed with gemma-768d (legacy)")
response = requests.post(
    f"{API_URL}/api/embed",
    json={"model": "gemma-768d", "input": "Hello world"}
)
result = response.json()
print(f"Model: {result['model']}")
print(f"Embeddings shape: {len(result['embeddings'])} x {len(result['embeddings'][0])}")
print()

# Test 6: Multiple texts with primary model
print("6. Testing multiple texts with m2v-bge-m3-1024d")
response = requests.post(
    f"{API_URL}/api/embed",
    json={"model": "m2v-bge-m3-1024d", "input": ["Machine learning", "Deep learning", "AI"]}
)
result = response.json()
print(f"Model: {result['model']}")
print(f"Embeddings shape: {len(result['embeddings'])} x {len(result['embeddings'][0])}")
print()

# Test 7: Reranking with qwen3-rerank
print("7. Testing POST /api/rerank with qwen3-rerank")
response = requests.post(
    f"{API_URL}/api/rerank",
    json={
        "model": "qwen3-rerank",
        "query": "What is machine learning?",
        "documents": [
            "Machine learning is a subset of AI",
            "The weather is nice today",
            "Deep learning uses neural networks"
        ]
    }
)
result = response.json()
print(f"Query: {result.get('query', 'N/A')}")
print(f"Results: {len(result.get('results', []))} documents ranked")
print()

print("="*60)
print("ALL TESTS COMPLETED!")
print("="*60)
