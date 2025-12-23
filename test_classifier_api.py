"""
Test VL Classifier API on Railway
==================================

Tests the deployed classifier with sample images.
"""

import requests
import base64
import os
from PIL import Image, ImageDraw
import io

# API configuration
API_URL = "https://deposiumembeddings-turbov2-production.up.railway.app"
API_KEY = os.getenv("EMBEDDINGS_API_KEY", "sYt3Q7t5XL-mSMXgdxG5lMKYrCvHPUPuzNQ0hpafN-o")

def create_test_images():
    """Create simple test images."""

    # 1. LOW complexity: Pure text
    low_img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(low_img)

    # Draw text lines
    for i in range(20):
        y = 50 + i * 25
        draw.rectangle([50, y, 750, y + 8], fill='black')

    # 2. HIGH complexity: Graph with axes
    high_img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(high_img)

    # Draw axes
    draw.line([100, 500, 700, 500], fill='black', width=3)  # X-axis
    draw.line([100, 500, 100, 100], fill='black', width=3)  # Y-axis

    # Draw a simple line graph
    points = [(100 + i*30, 500 - i*20) for i in range(20)]
    draw.line(points, fill='blue', width=2)

    # Add some markers
    for x, y in points[::2]:
        draw.ellipse([x-3, y-3, x+3, y+3], fill='red')

    return low_img, high_img

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64."""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')

def test_classifier(image: Image.Image, expected_class: str) -> dict:
    """Test classifier with an image."""

    # Convert to base64
    img_b64 = image_to_base64(image)

    # Make request
    response = requests.post(
        f"{API_URL}/api/classify/base64",
        json={"image": img_b64},
        headers={"X-API-Key": API_KEY}
    )

    if response.status_code == 200:
        result = response.json()
        print(f"\n{'='*60}")
        print(f"Expected: {expected_class}")
        print(f"Predicted: {result['class_name']} ({result['confidence']*100:.1f}%)")
        print(f"Probabilities: LOW={result['probabilities']['LOW']:.3f}, HIGH={result['probabilities']['HIGH']:.3f}")
        print(f"Latency: {result['latency_ms']:.1f}ms")
        print(f"Routing: {result['routing_decision']}")
        print(f"✅ PASS" if result['class_name'] == expected_class else "❌ FAIL")
        return result
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return None

def main():
    """Run classifier tests."""
    print("="*60)
    print("Testing VL Classifier on Railway")
    print("="*60)
    print(f"API URL: {API_URL}")

    # Create test images
    print("\nCreating test images...")
    low_img, high_img = create_test_images()
    print("✅ Test images created")

    # Test LOW complexity (pure text)
    print("\n--- Test 1: LOW Complexity (Pure Text) ---")
    test_classifier(low_img, "LOW")

    # Test HIGH complexity (graph)
    print("\n--- Test 2: HIGH Complexity (Graph with Axes) ---")
    test_classifier(high_img, "HIGH")

    print("\n" + "="*60)
    print("✅ Tests completed!")
    print("="*60)

if __name__ == "__main__":
    main()
