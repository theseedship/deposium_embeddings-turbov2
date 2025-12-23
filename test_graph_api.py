"""
Test graph.png via Railway API
"""
import requests
import base64
import os

API_URL = "https://deposiumembeddings-turbov2-production.up.railway.app"
API_KEY = os.getenv("EMBEDDINGS_API_KEY", "sYt3Q7t5XL-mSMXgdxG5lMKYrCvHPUPuzNQ0hpafN-o")

# Read and encode image
with open("graph.png", "rb") as f:
    img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')

print("Testing graph.png via Railway API...")
print(f"Image size: {len(img_bytes)} bytes")

# Call API
response = requests.post(
    f"{API_URL}/api/classify/base64",
    json={"image": img_b64},
    headers={"X-API-Key": API_KEY}
)

if response.status_code == 200:
    result = response.json()
    print(f"\n{'='*60}")
    print(f"Prediction: {result['class_name']} ({result['confidence']*100:.1f}%)")
    print(f"Probabilities: LOW={result['probabilities']['LOW']:.3f}, HIGH={result['probabilities']['HIGH']:.3f}")
    print(f"Latency: {result['latency_ms']:.1f}ms")
    print(f"Routing: {result['routing_decision']}")
    print(f"{'='*60}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
