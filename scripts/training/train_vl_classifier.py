"""
Train VL Complexity Classifier with CLIP
=========================================

Trains a binary classifier for document complexity routing using CLIP vision encoder.

Architecture:
- CLIP Vision Encoder (frozen) → Feature extraction
- Linear Classifier Head → Binary classification [LOW, HIGH]

Training objectives:
1. HIGH recall = 100% (priority: never miss complex documents)
2. Balanced accuracy across LOW/HIGH
3. Model size optimized for ONNX INT8 export

Dataset: 1000 images (400 LOW / 600 HIGH)
- Train: 700 (280 LOW / 420 HIGH)
- Val: 150 (60 LOW / 90 HIGH)
- Test: 150 (60 LOW / 90 HIGH)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score
import logging
from tqdm import tqdm
import json
from transformers import CLIPProcessor, CLIPModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
DATASET_ROOT = project_root / "data" / "complexity_classification"
IMAGES_DIR = DATASET_ROOT / "images"
ANNOTATIONS_CSV = DATASET_ROOT / "annotations.csv"
MODEL_SAVE_DIR = project_root / "models" / "vl_classifier_clip"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


class ComplexityDataset(Dataset):
    """Dataset for document complexity classification."""

    def __init__(self, annotations_df, images_dir, transform=None, clip_processor=None):
        """
        Args:
            annotations_df: DataFrame with image_path, label columns
            images_dir: Root directory for images
            transform: Optional torchvision transforms
            clip_processor: CLIP processor for preprocessing
        """
        self.annotations = annotations_df
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.clip_processor = clip_processor

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get annotation
        row = self.annotations.iloc[idx]
        img_path = self.images_dir / row['image_path']
        label = int(row['label'])

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.clip_processor is not None:
            # CLIP preprocessing
            pixel_values = self.clip_processor(images=image, return_tensors="pt")['pixel_values'][0]
            return pixel_values, label
        elif self.transform is not None:
            # Custom transforms
            image = self.transform(image)
            return image, label
        else:
            # Default: to tensor
            image = transforms.ToTensor()(image)
            return image, label


class CLIPComplexityClassifier(nn.Module):
    """
    CLIP-based binary classifier for document complexity.

    Architecture:
    - CLIP Vision Encoder (frozen) → 512D or 768D features
    - Dropout (0.3)
    - Linear (features → 2 classes)
    """

    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", freeze_clip=True, dropout=0.3):
        """
        Args:
            clip_model_name: HuggingFace CLIP model name
            freeze_clip: Freeze CLIP encoder weights
            dropout: Dropout probability before classifier
        """
        super().__init__()

        # Load CLIP model
        logger.info(f"Loading CLIP model: {clip_model_name}")
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)

        # Freeze CLIP if specified
        if freeze_clip:
            for param in self.clip_model.vision_model.parameters():
                param.requires_grad = False
            logger.info("CLIP vision encoder frozen")

        # Get feature dimension
        # CLIP ViT-B/32: 512D
        # CLIP ViT-L/14: 768D
        # CLIP ResNet50: 1024D
        self.feature_dim = self.clip_model.config.vision_config.hidden_size

        logger.info(f"CLIP feature dimension: {self.feature_dim}D")

        # Classifier head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.feature_dim, 2)

        logger.info(f"Classifier architecture: CLIP ({self.feature_dim}D) → Dropout({dropout}) → Linear(2)")

    def forward(self, pixel_values):
        """
        Forward pass.

        Args:
            pixel_values: (batch_size, 3, 224, 224) preprocessed images

        Returns:
            logits: (batch_size, 2) class logits
        """
        # Extract CLIP features
        with torch.no_grad() if self.clip_model.vision_model.training == False else torch.enable_grad():
            vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
            # pooled_output: (batch_size, feature_dim)
            features = vision_outputs.pooler_output

        # Classifier
        features = self.dropout(features)
        logits = self.classifier(features)

        return logits


def load_dataset(split='train'):
    """
    Load dataset split.

    Args:
        split: 'train', 'val', or 'test'

    Returns:
        DataFrame: Annotations for the split
    """
    df = pd.read_csv(ANNOTATIONS_CSV)

    # Filter by split
    split_df = df[df['image_path'].str.startswith(f"{split}/")]

    logger.info(f"{split.upper()} split: {len(split_df)} images")
    logger.info(f"  LOW: {(split_df['label'] == 0).sum()}")
    logger.info(f"  HIGH: {(split_df['label'] == 1).sum()}")

    return split_df


def calculate_class_weights(train_df):
    """
    Calculate class weights for imbalanced dataset.

    Args:
        train_df: Training DataFrame

    Returns:
        torch.Tensor: Class weights [weight_low, weight_high]
    """
    num_low = (train_df['label'] == 0).sum()
    num_high = (train_df['label'] == 1).sum()
    total = len(train_df)

    # Inverse frequency weighting
    weight_low = total / (2 * num_low)
    weight_high = total / (2 * num_high)

    # Boost HIGH class slightly to ensure recall = 100%
    weight_high *= 1.2  # 20% boost

    weights = torch.tensor([weight_low, weight_high], dtype=torch.float32)

    logger.info(f"Class weights: LOW={weight_low:.3f}, HIGH={weight_high:.3f}")

    return weights


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for pixel_values, labels in tqdm(dataloader, desc="Training"):
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(pixel_values)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    # Metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average=None)  # [recall_low, recall_high]

    return avg_loss, accuracy, recall


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for pixel_values, labels in tqdm(dataloader, desc="Validation"):
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(pixel_values)

            # Compute loss
            loss = criterion(logits, labels)

            # Statistics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average=None)  # [recall_low, recall_high]
    precision = precision_score(all_labels, all_preds, average=None)

    # Detailed report
    report = classification_report(all_labels, all_preds, target_names=['LOW', 'HIGH'], output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, accuracy, recall, precision, report, cm


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, save_dir):
    """
    Main training loop.

    Args:
        model: Model to train
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs
        device: Device (cuda/cpu)
        save_dir: Directory to save checkpoints
    """
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)

    best_high_recall = 0.0
    best_epoch = 0
    training_history = []

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        logger.info("-" * 60)

        # Train
        train_loss, train_acc, train_recall = train_epoch(model, train_loader, criterion, optimizer, device)

        logger.info(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        logger.info(f"Train Recall: LOW={train_recall[0]:.4f}, HIGH={train_recall[1]:.4f}")

        # Validate
        val_loss, val_acc, val_recall, val_precision, val_report, val_cm = validate(model, val_loader, criterion, device)

        logger.info(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        logger.info(f"Val Recall: LOW={val_recall[0]:.4f}, HIGH={val_recall[1]:.4f}")
        logger.info(f"Val Precision: LOW={val_precision[0]:.4f}, HIGH={val_precision[1]:.4f}")
        logger.info(f"\nConfusion Matrix:\n{val_cm}")

        # Learning rate schedule
        scheduler.step(val_loss)

        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_recall_low': float(train_recall[0]),
            'train_recall_high': float(train_recall[1]),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_recall_low': float(val_recall[0]),
            'val_recall_high': float(val_recall[1]),
            'val_precision_low': float(val_precision[0]),
            'val_precision_high': float(val_precision[1])
        })

        # Save best model (based on HIGH recall)
        if val_recall[1] > best_high_recall:
            best_high_recall = val_recall[1]
            best_epoch = epoch + 1

            # Save checkpoint
            checkpoint_path = save_dir / "best_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_recall_high': val_recall[1],
                'val_acc': val_acc,
                'val_report': val_report
            }, checkpoint_path)

            logger.info(f"✅ NEW BEST MODEL! HIGH recall: {val_recall[1]:.4f} (saved to {checkpoint_path})")

        # Early stopping if HIGH recall = 1.0
        if val_recall[1] >= 0.999:  # ~100%
            logger.info(f"\n🎯 TARGET ACHIEVED! HIGH recall = {val_recall[1]:.4f} (≥100%)")
            logger.info("Early stopping...")
            break

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best HIGH recall: {best_high_recall:.4f}")

    # Save training history
    history_path = save_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)

    logger.info(f"\n💾 Training history saved to: {history_path}")

    return training_history


def main():
    """Main training function."""
    # Hyperparameters
    CLIP_MODEL = "openai/clip-vit-base-patch32"  # or "openai/clip-vit-large-patch14", "openai/clip-resnet-50"
    FREEZE_CLIP = True
    DROPOUT = 0.3
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    logger.info("=" * 80)
    logger.info("VL COMPLEXITY CLASSIFIER - TRAINING WITH CLIP")
    logger.info("=" * 80)
    logger.info(f"\nHyperparameters:")
    logger.info(f"  CLIP Model: {CLIP_MODEL}")
    logger.info(f"  Freeze CLIP: {FREEZE_CLIP}")
    logger.info(f"  Dropout: {DROPOUT}")
    logger.info(f"  Batch Size: {BATCH_SIZE}")
    logger.info(f"  Epochs: {NUM_EPOCHS}")
    logger.info(f"  Learning Rate: {LEARNING_RATE}")
    logger.info(f"  Weight Decay: {WEIGHT_DECAY}")

    # Load CLIP processor
    logger.info("\nLoading CLIP processor...")
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

    # Load datasets
    logger.info("\nLoading datasets...")
    train_df = load_dataset('train')
    val_df = load_dataset('val')
    test_df = load_dataset('test')

    # Create datasets
    train_dataset = ComplexityDataset(train_df, IMAGES_DIR, clip_processor=clip_processor)
    val_dataset = ComplexityDataset(val_df, IMAGES_DIR, clip_processor=clip_processor)
    test_dataset = ComplexityDataset(test_df, IMAGES_DIR, clip_processor=clip_processor)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    logger.info(f"\nDataLoaders created:")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")

    # Calculate class weights
    class_weights = calculate_class_weights(train_df).to(DEVICE)

    # Create model
    logger.info("\nCreating model...")
    model = CLIPComplexityClassifier(clip_model_name=CLIP_MODEL, freeze_clip=FREEZE_CLIP, dropout=DROPOUT)
    model = model.to(DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"\nModel parameters:")
    logger.info(f"  Total: {total_params:,}")
    logger.info(f"  Trainable: {trainable_params:,}")
    logger.info(f"  Frozen: {total_params - trainable_params:,}")

    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Train
    history = train(model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS, DEVICE, MODEL_SAVE_DIR)

    # Test best model
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATING BEST MODEL ON TEST SET")
    logger.info("=" * 80)

    # Load best checkpoint
    checkpoint = torch.load(MODEL_SAVE_DIR / "best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test
    test_loss, test_acc, test_recall, test_precision, test_report, test_cm = validate(model, test_loader, criterion, DEVICE)

    logger.info(f"\nTest Results:")
    logger.info(f"  Loss: {test_loss:.4f}")
    logger.info(f"  Accuracy: {test_acc:.4f}")
    logger.info(f"  LOW - Recall: {test_recall[0]:.4f}, Precision: {test_precision[0]:.4f}")
    logger.info(f"  HIGH - Recall: {test_recall[1]:.4f}, Precision: {test_precision[1]:.4f}")
    logger.info(f"\nConfusion Matrix:\n{test_cm}")
    logger.info(f"\nDetailed Report:")
    logger.info(json.dumps(test_report, indent=2))

    # Save test results
    test_results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'test_recall_low': float(test_recall[0]),
        'test_recall_high': float(test_recall[1]),
        'test_precision_low': float(test_precision[0]),
        'test_precision_high': float(test_precision[1]),
        'confusion_matrix': test_cm.tolist(),
        'classification_report': test_report
    }

    test_results_path = MODEL_SAVE_DIR / "test_results.json"
    with open(test_results_path, 'w') as f:
        json.dump(test_results, f, indent=2)

    logger.info(f"\n💾 Test results saved to: {test_results_path}")

    # Check if target achieved
    if test_recall[1] >= 0.99:
        logger.info(f"\n🎉 SUCCESS! HIGH recall = {test_recall[1]:.4f} (≥100% target)")
        logger.info("Model ready for ONNX export (Phase 3.3)")
    else:
        logger.info(f"\n⚠️  WARNING: HIGH recall = {test_recall[1]:.4f} (<100% target)")
        logger.info("Consider:")
        logger.info("  1. Training for more epochs")
        logger.info("  2. Adjusting class weights")
        logger.info("  3. Using a larger CLIP model (ViT-L/14)")

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
