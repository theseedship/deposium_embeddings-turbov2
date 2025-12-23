"""
Create Balanced Dataset for VL Complexity Classifier
====================================================

Generates synthetic images for training a balanced LOW/HIGH classifier.

Dataset composition:
- 40% LOW (text documents, simple forms)
- 60% HIGH (charts, graphs, tables, maps, diagrams)

Split:
- 70% train
- 15% validation
- 15% test

Total target: 1000 images (400 LOW / 600 HIGH)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import logging
from typing import Tuple, List
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dataset paths
DATASET_ROOT = project_root / "data" / "complexity_classification"
IMAGES_DIR = DATASET_ROOT / "images"
TRAIN_DIR = IMAGES_DIR / "train"
VAL_DIR = IMAGES_DIR / "val"
TEST_DIR = IMAGES_DIR / "test"

# Create directories
for dir_path in [DATASET_ROOT, IMAGES_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Random seed for reproducibility
random.seed(42)
np.random.seed(42)


def get_arial_font(size=20):
    """Get Arial font or fallback to default."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except:
        return ImageFont.load_default()


class LowComplexityGenerator:
    """Generator for LOW complexity images (text only)."""

    @staticmethod
    def generate_text_page(width=800, height=1000):
        """Generate simple text page."""
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)

        # Draw text lines
        num_lines = random.randint(30, 50)
        for i in range(num_lines):
            y = 50 + i * (height - 100) // num_lines
            line_width = random.randint(int(width * 0.6), int(width * 0.9))
            x_start = random.randint(50, 100)

            # Simulate text with black rectangles
            draw.rectangle([x_start, y, x_start + line_width, y + 8], fill='black')

        return img, "text", "Page with text paragraphs"

    @staticmethod
    def generate_simple_form(width=800, height=1000):
        """Generate simple form with text fields."""
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        font = get_arial_font(16)

        # Form fields
        fields = ["Name:", "Email:", "Address:", "Phone:", "City:", "Country:"]

        for i, label in enumerate(fields):
            y = 100 + i * 120

            # Label
            draw.text((100, y), label, fill='black', font=font)

            # Field box
            draw.rectangle([250, y - 5, 700, y + 30], outline='black', width=2)

        return img, "form", "Simple form with text fields"

    @staticmethod
    def generate_letter(width=800, height=1000):
        """Generate official letter."""
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        font_title = get_arial_font(24)
        font_text = get_arial_font(14)

        # Header
        draw.text((300, 50), "Official Letter", fill='black', font=font_title)

        # Date
        draw.text((100, 120), "Date: 2025-10-23", fill='black', font=font_text)

        # Body (simulated text lines)
        for i in range(25):
            y = 180 + i * 25
            line_width = random.randint(400, 650)
            draw.rectangle([100, y, 100 + line_width, y + 8], fill='black')

        # Signature line
        draw.line([100, 850, 400, 850], fill='black', width=2)
        draw.text((100, 870), "Signature", fill='black', font=font_text)

        return img, "letter", "Official letter with paragraphs"

    @staticmethod
    def generate_simple_invoice(width=800, height=1000):
        """Generate simple text invoice (no table)."""
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        font_title = get_arial_font(28)
        font_text = get_arial_font(16)

        # Title
        draw.text((320, 50), "INVOICE", fill='black', font=font_title)

        # Invoice details (text only, no table)
        details = [
            "Invoice #: INV-2025-001",
            "Date: 2025-10-23",
            "Customer: John Doe",
            "",
            "Item 1: Product A    $100.00",
            "Item 2: Product B    $150.00",
            "Item 3: Service X    $200.00",
            "",
            "Subtotal:           $450.00",
            "Tax (10%):          $45.00",
            "------------------------",
            "TOTAL:              $495.00"
        ]

        y = 150
        for line in details:
            draw.text((100, y), line, fill='black', font=font_text)
            y += 40

        return img, "invoice_simple", "Simple invoice with text lines"


class HighComplexityGenerator:
    """Generator for HIGH complexity images (charts, graphs, tables, etc.)."""

    @staticmethod
    def generate_bar_chart(width=800, height=600):
        """Generate bar chart using matplotlib."""
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)

        # Random data
        categories = [f'Cat {i}' for i in range(random.randint(4, 8))]
        values = np.random.randint(10, 100, size=len(categories))
        colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))

        ax.bar(categories, values, color=colors)
        ax.set_xlabel('Categories')
        ax.set_ylabel('Values')
        ax.set_title('Bar Chart Example')
        ax.grid(True, alpha=0.3)

        # Convert to PIL Image
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = Image.fromarray(img_array[:, :, :3])  # Remove alpha channel

        plt.close(fig)
        return img, "bar_chart", "Bar chart with colored bars"

    @staticmethod
    def generate_line_graph(width=800, height=600):
        """Generate line graph using matplotlib."""
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)

        # Random data
        x = np.linspace(0, 10, 50)
        num_lines = random.randint(2, 4)

        for i in range(num_lines):
            y = np.sin(x + i) + np.random.normal(0, 0.1, len(x))
            ax.plot(x, y, marker='o', label=f'Series {i+1}', linewidth=2)

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_title('Line Graph Example')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Convert to PIL Image
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = Image.fromarray(img_array[:, :, :3])  # Remove alpha channel

        plt.close(fig)
        return img, "line_graph", "Line graph with multiple series"

    @staticmethod
    def generate_pie_chart(width=800, height=600):
        """Generate pie chart using matplotlib."""
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)

        # Random data
        num_slices = random.randint(3, 6)
        sizes = np.random.randint(10, 100, size=num_slices)
        labels = [f'Segment {i+1}' for i in range(num_slices)]
        colors = plt.cm.tab10(np.linspace(0, 1, num_slices))

        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Pie Chart Example')

        # Convert to PIL Image
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = Image.fromarray(img_array[:, :, :3])  # Remove alpha channel

        plt.close(fig)
        return img, "pie_chart", "Pie chart with segments"

    @staticmethod
    def generate_scatter_plot(width=800, height=600):
        """Generate scatter plot using matplotlib."""
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)

        # Random data
        num_groups = random.randint(2, 4)
        for i in range(num_groups):
            x = np.random.normal(i * 3, 1, 100)
            y = np.random.normal(i * 2, 1, 100)
            ax.scatter(x, y, alpha=0.6, s=50, label=f'Group {i+1}')

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_title('Scatter Plot Example')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Convert to PIL Image
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = Image.fromarray(img_array[:, :, :3])  # Remove alpha channel

        plt.close(fig)
        return img, "scatter_plot", "Scatter plot with clusters"

    @staticmethod
    def generate_table(width=800, height=600):
        """Generate data table."""
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)

        # Table dimensions
        rows = random.randint(6, 12)
        cols = random.randint(4, 7)
        cell_width = (width - 200) // cols
        cell_height = (height - 200) // rows

        start_x, start_y = 100, 100

        # Draw grid
        for i in range(rows + 1):
            y = start_y + i * cell_height
            draw.line([start_x, y, start_x + cols * cell_width, y], fill='black', width=2)

        for j in range(cols + 1):
            x = start_x + j * cell_width
            draw.line([x, start_y, x, start_y + rows * cell_height], fill='black', width=2)

        # Fill header row
        draw.rectangle([start_x, start_y, start_x + cols * cell_width, start_y + cell_height],
                       fill='lightgray')

        return img, "table", f"Data table {rows}x{cols}"

    @staticmethod
    def generate_flowchart(width=800, height=600):
        """Generate simple flowchart."""
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)

        # Boxes
        boxes = [
            (300, 50, 500, 110, 'Start'),
            (300, 150, 500, 210, 'Process 1'),
            (100, 270, 300, 330, 'Decision A'),
            (500, 270, 700, 330, 'Decision B'),
            (300, 400, 500, 460, 'End'),
        ]

        for x1, y1, x2, y2, label in boxes:
            draw.rectangle([x1, y1, x2, y2], fill='lightblue', outline='black', width=3)

        # Arrows
        arrows = [
            ((400, 110), (400, 150)),
            ((350, 210), (200, 270)),
            ((450, 210), (600, 270)),
            ((200, 330), (400, 400)),
            ((600, 330), (400, 400)),
        ]

        for start, end in arrows:
            draw.line([start, end], fill='black', width=2)

        return img, "flowchart", "Flowchart diagram with boxes"

    @staticmethod
    def generate_map(width=800, height=600):
        """Generate simple map."""
        img = Image.new('RGB', (width, height), color='lightblue')
        draw = ImageDraw.Draw(img)

        # Land masses
        land_color = 'green'

        num_lands = random.randint(3, 5)
        for _ in range(num_lands):
            x = random.randint(50, width - 150)
            y = random.randint(50, height - 150)
            w = random.randint(100, 200)
            h = random.randint(80, 180)
            draw.ellipse([x, y, x + w, y + h], fill=land_color, outline='darkgreen', width=2)

        # Grid lines
        for i in range(10):
            x = 50 + i * (width - 100) / 10
            draw.line([x, 50, x, height - 50], fill='gray', width=1)
        for i in range(8):
            y = 50 + i * (height - 100) / 8
            draw.line([50, y, width - 50, y], fill='gray', width=1)

        return img, "map", "Geographic map with grid"


class DatasetGenerator:
    """Main dataset generator."""

    def __init__(self, total_images=1000, low_ratio=0.4):
        """
        Initialize dataset generator.

        Args:
            total_images: Total number of images to generate
            low_ratio: Ratio of LOW complexity images (e.g., 0.4 = 40%)
        """
        self.total_images = total_images
        self.low_ratio = low_ratio
        self.high_ratio = 1.0 - low_ratio

        # Calculate splits
        self.num_low = int(total_images * low_ratio)
        self.num_high = total_images - self.num_low

        logger.info(f"Dataset configuration:")
        logger.info(f"  Total images: {total_images}")
        logger.info(f"  LOW images: {self.num_low} ({low_ratio*100:.0f}%)")
        logger.info(f"  HIGH images: {self.num_high} ({self.high_ratio*100:.0f}%)")

        # LOW generators
        self.low_generators = [
            LowComplexityGenerator.generate_text_page,
            LowComplexityGenerator.generate_simple_form,
            LowComplexityGenerator.generate_letter,
            LowComplexityGenerator.generate_simple_invoice,
        ]

        # HIGH generators
        self.high_generators = [
            HighComplexityGenerator.generate_bar_chart,
            HighComplexityGenerator.generate_line_graph,
            HighComplexityGenerator.generate_pie_chart,
            HighComplexityGenerator.generate_scatter_plot,
            HighComplexityGenerator.generate_table,
            HighComplexityGenerator.generate_flowchart,
            HighComplexityGenerator.generate_map,
        ]

        self.annotations = []

    def generate_image(self, label, idx, split):
        """
        Generate single image.

        Args:
            label: 0 (LOW) or 1 (HIGH)
            idx: Image index
            split: 'train', 'val', or 'test'

        Returns:
            dict: Annotation entry
        """
        # Select generator
        if label == 0:
            generator = random.choice(self.low_generators)
        else:
            generator = random.choice(self.high_generators)

        # Generate image
        img, category, description = generator()

        # Save image
        label_name = 'low' if label == 0 else 'high'
        filename = f"{label_name}_{idx:04d}.png"
        filepath = IMAGES_DIR / split / filename

        img.save(filepath)

        # Create annotation
        annotation = {
            'image_path': f"{split}/{filename}",
            'label': label,
            'category': category,
            'description': description
        }

        return annotation

    def generate_split(self, split_name, num_low, num_high):
        """Generate images for a split (train/val/test)."""
        logger.info(f"\nGenerating {split_name} split:")
        logger.info(f"  LOW: {num_low}, HIGH: {num_high}")

        # Generate LOW images
        for i in range(num_low):
            annotation = self.generate_image(label=0, idx=i, split=split_name)
            self.annotations.append(annotation)

            if (i + 1) % 50 == 0:
                logger.info(f"  Generated {i + 1}/{num_low} LOW images")

        # Generate HIGH images
        for i in range(num_high):
            annotation = self.generate_image(label=1, idx=i, split=split_name)
            self.annotations.append(annotation)

            if (i + 1) % 50 == 0:
                logger.info(f"  Generated {i + 1}/{num_high} HIGH images")

        logger.info(f"  ✅ {split_name} split complete ({num_low + num_high} images)")

    def generate(self):
        """Generate complete dataset."""
        logger.info("=" * 80)
        logger.info("GENERATING VL COMPLEXITY CLASSIFICATION DATASET")
        logger.info("=" * 80)

        # Calculate split sizes (70/15/15)
        train_low = int(self.num_low * 0.7)
        val_low = int(self.num_low * 0.15)
        test_low = self.num_low - train_low - val_low

        train_high = int(self.num_high * 0.7)
        val_high = int(self.num_high * 0.15)
        test_high = self.num_high - train_high - val_high

        # Generate splits
        self.generate_split('train', train_low, train_high)
        self.generate_split('val', val_low, val_high)
        self.generate_split('test', test_low, test_high)

        # Save annotations
        self.save_annotations()

        logger.info("\n" + "=" * 80)
        logger.info("DATASET GENERATION COMPLETE!")
        logger.info("=" * 80)

    def save_annotations(self):
        """Save annotations to CSV."""
        df = pd.DataFrame(self.annotations)
        csv_path = DATASET_ROOT / "annotations.csv"
        df.to_csv(csv_path, index=False)

        logger.info(f"\n💾 Annotations saved to: {csv_path}")
        logger.info(f"  Total annotations: {len(df)}")

        # Statistics
        logger.info("\n📊 Dataset statistics:")
        logger.info(df.groupby('label').size())
        logger.info("\n📋 Categories:")
        logger.info(df.groupby(['label', 'category']).size())


def main():
    """Main function."""
    # Generate dataset: 1000 images, 40% LOW / 60% HIGH
    generator = DatasetGenerator(total_images=1000, low_ratio=0.4)
    generator.generate()

    logger.info("\n✅ Dataset ready for training!")
    logger.info(f"📁 Location: {DATASET_ROOT}")
    logger.info("\nNext steps:")
    logger.info("  1. Review generated images manually (sample check)")
    logger.info("  2. Proceed to Phase 3: Train VL classifier with CLIP")


if __name__ == "__main__":
    main()
