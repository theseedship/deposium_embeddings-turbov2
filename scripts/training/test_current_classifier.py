"""
Test Current VL Complexity Classifier
======================================

Script to diagnose why the current model classifies everything as LOW.

Tests the model with various image types:
- Plain text documents (expected: LOW)
- Charts and graphs (expected: HIGH)
- Tables (expected: HIGH)
- Maps (expected: HIGH)
- Mixed documents (expected: HIGH)

Analyzes:
- Prediction probabilities (LOW vs HIGH)
- Confidence scores
- Potential bias in the model
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.classifier import ComplexityClassifier
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_test_images():
    """
    Generate synthetic test images for different document types.

    Returns:
        dict: {image_name: (PIL.Image, expected_class)}
    """
    images = {}

    # 1. Plain text document (LOW)
    text_img = Image.new('RGB', (800, 1000), color='white')
    draw = ImageDraw.Draw(text_img)

    # Simulate text lines
    for i in range(40):
        y = 50 + i * 20
        draw.rectangle([100, y, 700, y + 10], fill='black')

    images['plain_text.png'] = (text_img, 'LOW', 'Plain text document - only text lines')

    # 2. Bar chart (HIGH)
    chart_img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(chart_img)

    # Draw bars
    bar_colors = ['red', 'blue', 'green', 'orange']
    for i, color in enumerate(bar_colors):
        x = 150 + i * 150
        height = np.random.randint(100, 400)
        draw.rectangle([x, 500 - height, x + 80, 500], fill=color)

    # Draw axes
    draw.line([100, 500, 750, 500], fill='black', width=3)  # X-axis
    draw.line([100, 100, 100, 500], fill='black', width=3)  # Y-axis

    images['bar_chart.png'] = (chart_img, 'HIGH', 'Bar chart with colored bars')

    # 3. Line graph (HIGH)
    graph_img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(graph_img)

    # Draw axes
    draw.line([100, 500, 750, 500], fill='black', width=2)
    draw.line([100, 100, 100, 500], fill='black', width=2)

    # Draw line graph
    points = [(100 + i * 50, 500 - np.random.randint(50, 350)) for i in range(13)]
    draw.line(points, fill='blue', width=3)

    # Draw points
    for x, y in points:
        draw.ellipse([x-5, y-5, x+5, y+5], fill='red')

    images['line_graph.png'] = (graph_img, 'HIGH', 'Line graph with data points')

    # 4. Pie chart (HIGH)
    pie_img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(pie_img)

    # Draw pie slices (approximation with polygons)
    center = (400, 300)
    radius = 200

    # Draw circle segments
    colors = ['red', 'blue', 'green', 'yellow', 'purple']
    angles = [0, 72, 144, 216, 288, 360]

    for i, color in enumerate(colors):
        start_angle = angles[i]
        end_angle = angles[i + 1]

        # Simple pie slice simulation
        points = [center]
        for angle in range(start_angle, end_angle + 1, 5):
            rad = np.radians(angle)
            x = center[0] + radius * np.cos(rad)
            y = center[1] + radius * np.sin(rad)
            points.append((x, y))
        points.append(center)

        draw.polygon(points, fill=color, outline='black')

    images['pie_chart.png'] = (pie_img, 'HIGH', 'Pie chart with colored segments')

    # 5. Table (HIGH)
    table_img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(table_img)

    # Draw table grid
    rows, cols = 10, 5
    cell_width, cell_height = 120, 50
    start_x, start_y = 100, 100

    # Horizontal lines
    for i in range(rows + 1):
        y = start_y + i * cell_height
        draw.line([start_x, y, start_x + cols * cell_width, y], fill='black', width=2)

    # Vertical lines
    for j in range(cols + 1):
        x = start_x + j * cell_width
        draw.line([x, start_y, x, start_y + rows * cell_height], fill='black', width=2)

    # Fill header row
    draw.rectangle([start_x, start_y, start_x + cols * cell_width, start_y + cell_height],
                   fill='lightgray')

    images['table.png'] = (table_img, 'HIGH', 'Data table with grid')

    # 6. Map (HIGH) - simplified
    map_img = Image.new('RGB', (800, 600), color='lightblue')
    draw = ImageDraw.Draw(map_img)

    # Draw some "land masses"
    land_color = 'green'
    draw.ellipse([100, 150, 350, 400], fill=land_color, outline='darkgreen')
    draw.ellipse([450, 200, 700, 500], fill=land_color, outline='darkgreen')
    draw.polygon([(200, 50), (350, 100), (300, 200), (150, 150)],
                 fill=land_color, outline='darkgreen')

    # Add grid lines
    for i in range(10):
        x = 100 + i * 60
        draw.line([x, 50, x, 550], fill='gray', width=1)
    for i in range(8):
        y = 50 + i * 62.5
        draw.line([100, y, 700, y], fill='gray', width=1)

    images['map.png'] = (map_img, 'HIGH', 'Geographic map with landmasses and grid')

    # 7. Complex diagram (HIGH)
    diagram_img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(diagram_img)

    # Draw flowchart-like boxes
    boxes = [
        (150, 100, 300, 150, 'Start'),
        (150, 200, 300, 250, 'Process 1'),
        (500, 200, 650, 250, 'Process 2'),
        (325, 350, 475, 400, 'End'),
    ]

    for x1, y1, x2, y2, label in boxes:
        draw.rectangle([x1, y1, x2, y2], fill='lightblue', outline='black', width=2)

    # Draw arrows
    arrows = [
        ((225, 150), (225, 200)),  # Start to Process 1
        ((300, 225), (500, 225)),  # Process 1 to Process 2
        ((225, 250), (350, 350)),  # Process 1 to End
        ((575, 250), (425, 350)),  # Process 2 to End
    ]

    for start, end in arrows:
        draw.line([start, end], fill='black', width=2)
        # Simple arrowhead
        draw.polygon([
            (end[0], end[1]),
            (end[0] - 10, end[1] - 10),
            (end[0] + 10, end[1] - 10)
        ], fill='black')

    images['diagram.png'] = (diagram_img, 'HIGH', 'Flowchart diagram with boxes and arrows')

    # 8. Simple form (LOW)
    form_img = Image.new('RGB', (800, 1000), color='white')
    draw = ImageDraw.Draw(form_img)

    # Draw form fields (text lines and boxes)
    fields = [
        (100, 100, "Name:"),
        (100, 200, "Email:"),
        (100, 300, "Address:"),
        (100, 400, "Phone:"),
    ]

    for x, y, label in fields:
        # Draw field box
        draw.rectangle([x + 150, y - 10, x + 650, y + 30], outline='black', width=2)

    images['simple_form.png'] = (form_img, 'LOW', 'Simple form with text fields')

    return images


def test_classifier(classifier, test_images):
    """
    Test classifier on all images and analyze results.

    Args:
        classifier: ComplexityClassifier instance
        test_images: dict of {name: (image, expected_class, description)}

    Returns:
        dict: Test results with statistics
    """
    results = []

    logger.info("=" * 80)
    logger.info("TESTING CURRENT VL COMPLEXITY CLASSIFIER")
    logger.info("=" * 80)

    for img_name, (img, expected, description) in test_images.items():
        logger.info(f"\nTesting: {img_name}")
        logger.info(f"  Description: {description}")
        logger.info(f"  Expected: {expected}")

        # Predict
        prediction = classifier.predict(img)

        # Extract results
        predicted_class = prediction['class_name']
        confidence = prediction['confidence']
        probs = prediction['probabilities']

        # Check if correct
        correct = (predicted_class == expected)

        logger.info(f"  Predicted: {predicted_class} ({confidence*100:.1f}% confidence)")
        logger.info(f"  Probabilities: LOW={probs['LOW']:.4f}, HIGH={probs['HIGH']:.4f}")
        logger.info(f"  Result: {'✅ CORRECT' if correct else '❌ WRONG'}")

        results.append({
            'image': img_name,
            'description': description,
            'expected': expected,
            'predicted': predicted_class,
            'confidence': confidence,
            'prob_low': probs['LOW'],
            'prob_high': probs['HIGH'],
            'correct': correct
        })

    return results


def analyze_results(results):
    """
    Analyze test results and diagnose bias.

    Args:
        results: list of test results

    Returns:
        dict: Analysis statistics
    """
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS & DIAGNOSIS")
    logger.info("=" * 80)

    # Calculate statistics
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    accuracy = correct / total * 100

    # Count predictions
    low_predictions = sum(1 for r in results if r['predicted'] == 'LOW')
    high_predictions = sum(1 for r in results if r['predicted'] == 'HIGH')

    # Count expected
    expected_low = sum(1 for r in results if r['expected'] == 'LOW')
    expected_high = sum(1 for r in results if r['expected'] == 'HIGH')

    # Calculate average probabilities
    avg_prob_low = np.mean([r['prob_low'] for r in results])
    avg_prob_high = np.mean([r['prob_high'] for r in results])

    # Find HIGH class performance
    high_results = [r for r in results if r['expected'] == 'HIGH']
    high_correct = sum(1 for r in high_results if r['correct'])
    high_recall = (high_correct / len(high_results) * 100) if high_results else 0

    # Find LOW class performance
    low_results = [r for r in results if r['expected'] == 'LOW']
    low_correct = sum(1 for r in low_results if r['correct'])
    low_recall = (low_correct / len(low_results) * 100) if low_results else 0

    logger.info(f"\n📊 OVERALL STATISTICS:")
    logger.info(f"  Total images tested: {total}")
    logger.info(f"  Accuracy: {accuracy:.1f}% ({correct}/{total})")
    logger.info(f"  Expected LOW: {expected_low}, Expected HIGH: {expected_high}")
    logger.info(f"  Predicted LOW: {low_predictions}, Predicted HIGH: {high_predictions}")

    logger.info(f"\n📈 PROBABILITY ANALYSIS:")
    logger.info(f"  Average P(LOW): {avg_prob_low:.4f}")
    logger.info(f"  Average P(HIGH): {avg_prob_high:.4f}")

    logger.info(f"\n🎯 PER-CLASS PERFORMANCE:")
    logger.info(f"  LOW recall: {low_recall:.1f}% ({low_correct}/{len(low_results)})")
    logger.info(f"  HIGH recall: {high_recall:.1f}% ({high_correct}/{len(high_results)})")

    # Diagnose bias
    logger.info(f"\n🔍 BIAS DIAGNOSIS:")

    if low_predictions == total:
        logger.info("  ⚠️  SEVERE BIAS: Model predicts LOW for ALL images!")
        logger.info("  ⚠️  This confirms the reported issue.")
    elif low_predictions > total * 0.8:
        logger.info(f"  ⚠️  STRONG BIAS: Model predicts LOW {low_predictions/total*100:.0f}% of the time")
    elif high_predictions == total:
        logger.info("  ⚠️  SEVERE BIAS: Model predicts HIGH for ALL images!")
    elif high_predictions > total * 0.8:
        logger.info(f"  ⚠️  STRONG BIAS: Model predicts HIGH {high_predictions/total*100:.0f}% of the time")
    else:
        logger.info("  ✅ No severe bias detected (predictions are varied)")

    # Check if probabilities are too close (model uncertain)
    if avg_prob_low > 0.45 and avg_prob_low < 0.55:
        logger.info("  ⚠️  Model is UNCERTAIN: Probabilities close to 50/50")
        logger.info("  ⚠️  Model may need threshold tuning or retraining")

    # Check for HIGH recall issue (most important metric)
    if high_recall < 50:
        logger.info(f"  🚨 CRITICAL: HIGH recall is only {high_recall:.1f}%!")
        logger.info("  🚨 Model is missing complex documents (charts, graphs, etc.)")
        logger.info("  🚨 This is the PRIMARY issue that needs fixing")
    elif high_recall < 100:
        logger.info(f"  ⚠️  WARNING: HIGH recall is {high_recall:.1f}% (target: 100%)")
    else:
        logger.info(f"  ✅ HIGH recall: {high_recall:.1f}% (perfect!)")

    # Recommendations
    logger.info(f"\n💡 RECOMMENDATIONS:")

    if high_recall < 100:
        logger.info("  1. ⚠️  Dataset was likely imbalanced (too many LOW examples)")
        logger.info("  2. ⚠️  Need to retrain with balanced dataset (50/50 or 40/60 LOW/HIGH)")
        logger.info("  3. ⚠️  Prioritize HIGH recall = 100% (critical for VLM routing)")
        logger.info("  4. ⚠️  Consider threshold adjustment (lower threshold for HIGH prediction)")

    if avg_prob_low > avg_prob_high:
        logger.info("  5. ⚠️  Model is biased towards LOW class")
        logger.info("  6. ⚠️  Use class weights during training to fix this")

    logger.info("  7. ✅ Create new balanced dataset with clear LOW/HIGH criteria")
    logger.info("  8. ✅ Retrain model with CLIP features + balanced dataset")

    # Return statistics
    return {
        'accuracy': accuracy,
        'low_recall': low_recall,
        'high_recall': high_recall,
        'avg_prob_low': avg_prob_low,
        'avg_prob_high': avg_prob_high,
        'low_predictions': low_predictions,
        'high_predictions': high_predictions,
        'total': total
    }


def save_results(results, stats, output_file='classifier_test_results.json'):
    """Save test results to JSON file."""
    output_path = Path(__file__).parent / output_file

    output_data = {
        'test_date': '2025-10-23',
        'model': 'model_quantized.onnx (current)',
        'statistics': stats,
        'detailed_results': results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\n💾 Results saved to: {output_path}")


def main():
    """Main test function."""
    # Initialize classifier
    logger.info("Loading current VL classifier...")
    classifier = ComplexityClassifier()

    # Generate test images
    logger.info("Generating synthetic test images...")
    test_images = generate_test_images()
    logger.info(f"Generated {len(test_images)} test images")

    # Test classifier
    results = test_classifier(classifier, test_images)

    # Analyze results
    stats = analyze_results(results)

    # Save results
    save_results(results, stats)

    logger.info("\n" + "=" * 80)
    logger.info("TESTING COMPLETE!")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("  1. Review the analysis above")
    logger.info("  2. Create balanced dataset (Phase 2)")
    logger.info("  3. Retrain model with CLIP (Phase 3)")


if __name__ == "__main__":
    main()
