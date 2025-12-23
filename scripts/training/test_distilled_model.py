"""
Test Distilled ResNet18 Model
==============================

Loads the trained ResNet18 student model and evaluates on test set.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
DATASET_ROOT = project_root / "data" / "complexity_classification"
IMAGES_DIR = DATASET_ROOT / "images_500"
ANNOTATIONS_CSV = DATASET_ROOT / "annotations_500.csv"
MODEL_PATH = project_root / "models" / "vl_distilled_resnet18" / "best_student.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ComplexityDataset:
    """Simple test dataset."""

    def __init__(self, annotations_df, images_dir, transform):
        self.annotations = annotations_df
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_path = self.images_dir / row['image_path']
        label = int(row['label'])

        image = Image.open(img_path).convert('RGB')
        image_tensor = self.transform(image)

        return image_tensor, label


class ResNet18Student(nn.Module):
    """ResNet18 student model."""

    def __init__(self, pretrained=False, feature_dim=768):
        super().__init__()

        # Load ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)

        # Get feature dimension (before final FC layer)
        self.resnet_feature_dim = self.resnet.fc.in_features  # 512

        # Replace final FC with custom layers
        self.resnet.fc = nn.Identity()  # Remove original FC

        # Feature projection (to match CLIP feature_dim)
        self.feature_projection = nn.Sequential(
            nn.Linear(self.resnet_feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Classifier
        self.classifier = nn.Linear(feature_dim, 2)

    def forward(self, x):
        """Extract features and logits."""
        # ResNet18 backbone
        x = self.resnet(x)  # (batch, 512)

        # Project to CLIP feature space
        features = self.feature_projection(x)  # (batch, feature_dim)

        # Classifier
        logits = self.classifier(features)

        return features, logits


def load_model(model_path, device):
    """Load trained model."""
    logger.info(f"Loading model from {model_path}")

    # Initialize model
    model = ResNet18Student(pretrained=False, feature_dim=768).to(device)

    # Load checkpoint (weights_only=False for checkpoint with optimizer state)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['student_state_dict'])

    model.eval()

    logger.info(f"Model loaded (epoch {checkpoint['epoch']}, val HIGH recall: {checkpoint['val_recall_high']:.4f})")
    return model


def test_model(model, test_loader, device):
    """Test model on test set."""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward
            _, logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average=None)
    report = classification_report(all_labels, all_preds, target_names=['LOW', 'HIGH'], output_dict=False)
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, recall, report, cm, all_probs


def main():
    """Main test function."""
    logger.info("=" * 80)
    logger.info("TESTING DISTILLED RESNET18 MODEL")
    logger.info("=" * 80)

    # Load test data
    df = pd.read_csv(ANNOTATIONS_CSV)
    test_df = df[df['image_path'].str.startswith("test/")]

    logger.info(f"\nTest set: {len(test_df)} images ({(test_df['label']==0).sum()} LOW / {(test_df['label']==1).sum()} HIGH)")

    # Transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset
    test_dataset = ComplexityDataset(test_df, IMAGES_DIR, transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # Load model
    model = load_model(MODEL_PATH, DEVICE)

    # Test
    logger.info("\nTesting on test set...")
    accuracy, recall, report, cm, probs = test_model(model, test_loader, DEVICE)

    # Results
    logger.info("\n" + "=" * 80)
    logger.info("TEST RESULTS")
    logger.info("=" * 80)
    logger.info(f"\nAccuracy: {accuracy:.4f}")
    logger.info(f"Recall LOW: {recall[0]:.4f}")
    logger.info(f"Recall HIGH: {recall[1]:.4f}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"\n{cm}")
    logger.info(f"\nClassification Report:")
    logger.info(f"\n{report}")

    # Check target
    if recall[1] >= 0.999:
        logger.info("\n✅ TARGET ACHIEVED! HIGH recall ≥ 99.9%")
    else:
        logger.warning(f"\n⚠️ HIGH recall below target: {recall[1]:.4f} < 0.999")

    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()
