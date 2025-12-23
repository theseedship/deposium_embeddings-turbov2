"""
Create STRICT 500-image Dataset for VL Complexity Classifier
=============================================================

Generates 500 synthetic images with STRICT criteria:
- 200 LOW (40%): Pure text only, no graphics
- 300 HIGH (60%): Any visual element (graphs, charts, maps, diagrams, tables)

Key focus for HIGH:
- 90 images: Graphs with axes (with/without exact values)
- 60 images: Technical diagrams
- 45 images: Maps
- 45 images: Tables
- 30 images: Drawings/infographics
- 30 images: Complex forms

Split: 70% train / 15% val / 15% test
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dataset paths
DATASET_ROOT = project_root / "data" / "complexity_classification"
IMAGES_DIR = DATASET_ROOT / "images_500"
TRAIN_DIR = IMAGES_DIR / "train"
VAL_DIR = IMAGES_DIR / "val"
TEST_DIR = IMAGES_DIR / "test"

for dir_path in [DATASET_ROOT, IMAGES_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)


def get_font(size=16):
    """Get font or fallback."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except:
        return ImageFont.load_default()


class StrictLOWGenerator:
    """Generate STRICT LOW images: text only, NO graphics."""

    @staticmethod
    def generate_pure_text(width=800, height=1000):
        """Pure text paragraphs."""
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)

        num_lines = random.randint(40, 60)
        for i in range(num_lines):
            y = 50 + i * 18
            line_width = random.randint(int(width * 0.6), int(width * 0.85))
            x_start = random.randint(60, 100)
            draw.rectangle([x_start, y, x_start + line_width, y + 6], fill='black')

        return img, "pure_text", "Pure text paragraphs - NO graphics"

    @staticmethod
    def generate_simple_letter(width=800, height=1000):
        """Letter with text only (no logo)."""
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)

        # Title
        for i in range(3):
            y = 80 + i * 20
            draw.rectangle([300, y, 500, y + 6], fill='black')

        # Body text
        for i in range(30):
            y = 200 + i * 25
            line_width = random.randint(500, 700)
            draw.rectangle([80, y, 80 + line_width, y + 6], fill='black')

        return img, "letter_text", "Letter with text only - no logo"

    @staticmethod
    def generate_text_list(width=800, height=1000):
        """Text list with simple bullets."""
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        font = get_font(14)

        for i in range(25):
            y = 100 + i * 35
            # Bullet (simple dot)
            draw.ellipse([100, y + 5, 110, y + 15], fill='black')
            # Text line
            line_width = random.randint(400, 650)
            draw.rectangle([130, y + 5, 130 + line_width, y + 11], fill='black')

        return img, "text_list", "Text list with simple bullets - no graphics"


class StrictHIGHGenerator:
    """Generate STRICT HIGH images: graphs, diagrams, maps, tables."""

    @staticmethod
    def generate_line_graph_with_axes(width=800, height=600):
        """Line graph with clear X/Y axes, possibly without exact values."""
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)

        # Data
        x = np.linspace(0, 10, 30)
        y = np.sin(x) + np.random.normal(0, 0.1, len(x))

        # Plot line
        ax.plot(x, y, 'b-', linewidth=2, marker='o', markersize=4)

        # Axes with labels (but maybe without exact tick values)
        if random.random() > 0.3:  # 70% have tick labels
            ax.set_xlabel('Time (hours)', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
        else:
            # Axes WITHOUT exact values (only suggested by scale)
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

        ax.set_title('Line Graph Example', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)

        # Convert to image
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = Image.fromarray(img_array[:, :, :3])
        plt.close(fig)

        return img, "line_graph_axes", "Line graph WITH axes (may lack exact values)"

    @staticmethod
    def generate_bar_chart_axes(width=800, height=600):
        """Bar chart with clear axes."""
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)

        categories = [f'Cat{i}' for i in range(random.randint(4, 7))]
        values = np.random.randint(10, 100, size=len(categories))

        ax.bar(categories, values, color=plt.cm.Set3(np.linspace(0, 1, len(categories))))
        ax.set_xlabel('Categories', fontsize=12)
        ax.set_ylabel('Values', fontsize=12)
        ax.set_title('Bar Chart Example', fontsize=14)
        ax.grid(True, axis='y', alpha=0.3)

        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = Image.fromarray(img_array[:, :, :3])
        plt.close(fig)

        return img, "bar_chart_axes", "Bar chart with axes"

    @staticmethod
    def generate_scatter_plot_axes(width=800, height=600):
        """Scatter plot with axes."""
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)

        x = np.random.normal(50, 15, 100)
        y = np.random.normal(50, 15, 100)

        ax.scatter(x, y, alpha=0.6, s=50, c='blue')
        ax.set_xlabel('X Variable', fontsize=12)
        ax.set_ylabel('Y Variable', fontsize=12)
        ax.set_title('Scatter Plot', fontsize=14)
        ax.grid(True, alpha=0.3)

        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = Image.fromarray(img_array[:, :, :3])
        plt.close(fig)

        return img, "scatter_plot_axes", "Scatter plot with axes"

    @staticmethod
    def generate_technical_diagram(width=800, height=600):
        """Technical diagram (circuit, flowchart, etc.)."""
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)

        # Draw boxes (components)
        boxes = [
            (100, 100, 250, 180, 'Component A'),
            (500, 100, 650, 180, 'Component B'),
            (300, 350, 450, 430, 'Component C'),
        ]

        for x1, y1, x2, y2, label in boxes:
            draw.rectangle([x1, y1, x2, y2], outline='black', width=3, fill='lightblue')

        # Draw connections (arrows)
        connections = [
            ((175, 180), (325, 350)),
            ((575, 180), (425, 350)),
        ]

        for start, end in connections:
            draw.line([start, end], fill='black', width=2)
            # Arrowhead
            draw.polygon([
                (end[0], end[1]),
                (end[0] - 8, end[1] - 12),
                (end[0] + 8, end[1] - 12)
            ], fill='black')

        return img, "technical_diagram", "Technical diagram with components"

    @staticmethod
    def generate_map_with_grid(width=800, height=600):
        """Map with coordinate grid."""
        img = Image.new('RGB', (width, height), color='lightblue')
        draw = ImageDraw.Draw(img)

        # Land masses
        for _ in range(random.randint(3, 5)):
            x = random.randint(100, width - 200)
            y = random.randint(100, height - 200)
            w = random.randint(120, 200)
            h = random.randint(100, 180)
            draw.ellipse([x, y, x + w, y + h], fill='green', outline='darkgreen', width=2)

        # Grid lines (coordinates)
        for i in range(10):
            x = 80 + i * (width - 160) / 10
            draw.line([x, 80, x, height - 80], fill='gray', width=1)
        for i in range(8):
            y = 80 + i * (height - 160) / 8
            draw.line([80, y, width - 80, y], fill='gray', width=1)

        return img, "map_grid", "Map with coordinate grid"

    @staticmethod
    def generate_data_table(width=800, height=600):
        """Data table with grid."""
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)

        rows, cols = random.randint(5, 8), random.randint(3, 5)
        cell_w, cell_h = 120, 60
        start_x, start_y = 100, 100

        # Grid
        for i in range(rows + 1):
            y = start_y + i * cell_h
            draw.line([start_x, y, start_x + cols * cell_w, y], fill='black', width=2)
        for j in range(cols + 1):
            x = start_x + j * cell_w
            draw.line([x, start_y, x, start_y + rows * cell_h], fill='black', width=2)

        # Header
        draw.rectangle([start_x, start_y, start_x + cols * cell_w, start_y + cell_h], fill='lightgray')

        return img, "data_table", "Data table with grid"


class Dataset500Generator:
    """Main generator for 500 images."""

    def __init__(self):
        self.total_images = 500
        self.num_low = 200  # 40%
        self.num_high = 300  # 60%

        logger.info(f"Dataset: 500 images (200 LOW / 300 HIGH)")

        # LOW generators (simple)
        self.low_generators = [
            StrictLOWGenerator.generate_pure_text,
            StrictLOWGenerator.generate_simple_letter,
            StrictLOWGenerator.generate_text_list,
        ]

        # HIGH generators with distribution
        self.high_generators = {
            'graphs_with_axes': [  # 90 images (30%)
                StrictHIGHGenerator.generate_line_graph_with_axes,
                StrictHIGHGenerator.generate_bar_chart_axes,
                StrictHIGHGenerator.generate_scatter_plot_axes,
            ],
            'technical_diagrams': [  # 60 images (20%)
                StrictHIGHGenerator.generate_technical_diagram,
            ],
            'maps': [  # 45 images (15%)
                StrictHIGHGenerator.generate_map_with_grid,
            ],
            'tables': [  # 45 images (15%)
                StrictHIGHGenerator.generate_data_table,
            ],
        }

        self.annotations = []

    def generate_image(self, label, idx, split, category_hint=None):
        """Generate single image."""
        if label == 0:  # LOW
            generator = random.choice(self.low_generators)
        else:  # HIGH
            if category_hint and category_hint in self.high_generators:
                generator = random.choice(self.high_generators[category_hint])
            else:
                # Random HIGH
                all_high = [g for gens in self.high_generators.values() for g in gens]
                generator = random.choice(all_high)

        img, category, description = generator()

        # Save
        label_name = 'low' if label == 0 else 'high'
        filename = f"{label_name}_{idx:04d}.png"
        filepath = IMAGES_DIR / split / filename
        img.save(filepath)

        return {
            'image_path': f"{split}/{filename}",
            'label': label,
            'category': category,
            'description': description
        }

    def generate_split(self, split_name, num_low, num_high):
        """Generate split with HIGH category distribution."""
        logger.info(f"\n{split_name.upper()} split: {num_low} LOW, {num_high} HIGH")

        # Generate LOW
        for i in range(num_low):
            ann = self.generate_image(0, i, split_name)
            self.annotations.append(ann)

        # Generate HIGH with distribution
        high_distribution = {
            'graphs_with_axes': int(num_high * 0.30),  # 30%
            'technical_diagrams': int(num_high * 0.20),  # 20%
            'maps': int(num_high * 0.15),  # 15%
            'tables': int(num_high * 0.15),  # 15%
        }

        # Remaining goes to mixed
        remaining = num_high - sum(high_distribution.values())

        idx = 0
        for category, count in high_distribution.items():
            for _ in range(count):
                ann = self.generate_image(1, idx, split_name, category_hint=category)
                self.annotations.append(ann)
                idx += 1

        # Generate remaining
        for _ in range(remaining):
            ann = self.generate_image(1, idx, split_name)
            self.annotations.append(ann)
            idx += 1

        logger.info(f"  ✅ {split_name} complete")

    def generate(self):
        """Generate complete dataset."""
        logger.info("=" * 80)
        logger.info("GENERATING 500-IMAGE STRICT DATASET")
        logger.info("=" * 80)

        # Calculate splits (70/15/15)
        train_low = int(self.num_low * 0.7)
        val_low = int(self.num_low * 0.15)
        test_low = self.num_low - train_low - val_low

        train_high = int(self.num_high * 0.7)
        val_high = int(self.num_high * 0.15)
        test_high = self.num_high - train_high - val_high

        logger.info(f"\nSplits:")
        logger.info(f"  Train: {train_low + train_high} ({train_low} LOW / {train_high} HIGH)")
        logger.info(f"  Val: {val_low + val_high} ({val_low} LOW / {val_high} HIGH)")
        logger.info(f"  Test: {test_low + test_high} ({test_low} LOW / {test_high} HIGH)")

        # Generate
        self.generate_split('train', train_low, train_high)
        self.generate_split('val', val_low, val_high)
        self.generate_split('test', test_low, test_high)

        # Save annotations
        df = pd.DataFrame(self.annotations)
        csv_path = DATASET_ROOT / "annotations_500.csv"
        df.to_csv(csv_path, index=False)

        logger.info(f"\n💾 Annotations: {csv_path}")
        logger.info(f"\n📊 Statistics:")
        logger.info(df.groupby(['label', 'category']).size())

        logger.info("\n" + "=" * 80)
        logger.info("✅ DATASET READY!")
        logger.info("=" * 80)


def main():
    generator = Dataset500Generator()
    generator.generate()


if __name__ == "__main__":
    main()
