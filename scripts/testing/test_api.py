#!/usr/bin/env python3
"""Test l'API avec les 3 modèles"""
import requests
import json

API_URL = "http://localhost:11436"

print("="*60)
print("Testing Deposium Embeddings API")
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

# Test 3: POST /api/embed with turbov2
print("3. Testing POST /api/embed with turbov2")
response = requests.post(
    f"{API_URL}/api/embed",
    json={"model": "turbov2", "input": "Hello world"}
)
result = response.json()
print(f"Model: {result['model']}")
print(f"Embeddings shape: {len(result['embeddings'])} x {len(result['embeddings'][0])}")
print()

# Test 4: POST /api/embed with int8
print("4. Testing POST /api/embed with int8")
response = requests.post(
    f"{API_URL}/api/embed",
    json={"model": "int8", "input": "Hello world"}
)
result = response.json()
print(f"Model: {result['model']}")
print(f"Embeddings shape: {len(result['embeddings'])} x {len(result['embeddings'][0])}")
print()

# Test 5: POST /api/embed with LEAF
print("5. Testing POST /api/embed with LEAF")
response = requests.post(
    f"{API_URL}/api/embed",
    json={"model": "leaf", "input": "Hello world"}
)
result = response.json()
print(f"Model: {result['model']}")
print(f"Embeddings shape: {len(result['embeddings'])} x {len(result['embeddings'][0])}")
print()

# Test 6: Multiple texts with LEAF
print("6. Testing multiple texts with LEAF")
response = requests.post(
    f"{API_URL}/api/embed",
    json={"model": "leaf", "input": ["Machine learning", "Deep learning", "AI"]}
)
result = response.json()
print(f"Model: {result['model']}")
print(f"Embeddings shape: {len(result['embeddings'])} x {len(result['embeddings'][0])}")
print()

print("="*60)
print("✅ ALL TESTS PASSED!")
print("="*60)
