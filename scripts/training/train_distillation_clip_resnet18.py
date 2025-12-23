"""
Knowledge Distillation: CLIP → ResNet18
========================================

Trains a lightweight ResNet18 student model by distilling knowledge from CLIP teacher.

Architecture:
- Teacher: CLIP ViT-B/32 (frozen) - Generates soft labels and features
- Student: ResNet18 - Learns to mimic CLIP

Loss combination:
1. Hard label loss (CrossEntropy with ground truth)
2. Soft label loss (KL divergence with CLIP predictions)
3. Feature matching loss (MSE between CLIP and ResNet18 features)

Target: HIGH recall = 100%, model size ~11MB ONNX INT8
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score
import logging
from tqdm import tqdm
import json
from transformers import CLIPProcessor, CLIPModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
DATASET_ROOT = project_root / "data" / "complexity_classification"
IMAGES_DIR = DATASET_ROOT / "images_500"
ANNOTATIONS_CSV = DATASET_ROOT / "annotations_500.csv"
MODEL_SAVE_DIR = project_root / "models" / "vl_distilled_resnet18"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")


class ComplexityDataset(Dataset):
    """Dataset for distillation."""

    def __init__(self, annotations_df, images_dir, clip_processor, transform_student):
        self.annotations = annotations_df
        self.images_dir = Path(images_dir)
        self.clip_processor = clip_processor
        self.transform_student = transform_student

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_path = self.images_dir / row['image_path']
        label = int(row['label'])

        image = Image.open(img_path).convert('RGB')

        # CLIP preprocessing
        clip_input = self.clip_processor(images=image, return_tensors="pt")['pixel_values'][0]

        # ResNet18 preprocessing
        student_input = self.transform_student(image)

        return clip_input, student_input, label


class CLIPTeacher(nn.Module):
    """CLIP teacher model (frozen)."""

    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        self.clip.eval()  # Always eval

        # Freeze all parameters
        for param in self.clip.parameters():
            param.requires_grad = False

        # Classifier on top of CLIP features
        self.feature_dim = self.clip.config.vision_config.hidden_size  # 512 for ViT-B/32
        self.classifier = nn.Linear(self.feature_dim, 2)

        logger.info(f"CLIP Teacher loaded: {model_name} ({self.feature_dim}D features)")

    def forward(self, pixel_values):
        """Extract features and logits."""
        with torch.no_grad():
            vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
            features = vision_outputs.pooler_output  # (batch, feature_dim)

        logits = self.classifier(features)
        return features, logits


class ResNet18Student(nn.Module):
    """ResNet18 student model."""

    def __init__(self, pretrained=True, feature_dim=512):
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

        logger.info(f"ResNet18 Student: {self.resnet_feature_dim}D → {feature_dim}D → 2 classes")

    def forward(self, x):
        """Extract features and logits."""
        # ResNet18 backbone
        x = self.resnet(x)  # (batch, 512)

        # Project to CLIP feature space
        features = self.feature_projection(x)  # (batch, feature_dim)

        # Classifier
        logits = self.classifier(features)

        return features, logits


class DistillationLoss(nn.Module):
    """Combined distillation loss."""

    def __init__(self, alpha=0.3, beta=0.4, gamma=0.3, temperature=4.0):
        """
        Args:
            alpha: Weight for hard label loss
            beta: Weight for soft label loss (KL divergence)
            gamma: Weight for feature matching loss
            temperature: Temperature for softening probabilities
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature

        logger.info(f"Distillation Loss:")
        logger.info(f"  α={alpha} (hard labels), β={beta} (soft labels), γ={gamma} (features), T={temperature}")

    def forward(self, student_features, student_logits, teacher_features, teacher_logits, labels, class_weights=None):
        """
        Compute combined loss.

        Args:
            student_features: (batch, feature_dim)
            student_logits: (batch, 2)
            teacher_features: (batch, feature_dim)
            teacher_logits: (batch, 2)
            labels: (batch,)
            class_weights: (2,) tensor for weighted CE

        Returns:
            total_loss, losses_dict
        """
        # 1. Hard label loss (CrossEntropy with ground truth)
        if class_weights is not None:
            ce_loss = F.cross_entropy(student_logits, labels, weight=class_weights)
        else:
            ce_loss = F.cross_entropy(student_logits, labels)

        # 2. Soft label loss (KL divergence with CLIP predictions)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (self.temperature ** 2)

        # 3. Feature matching loss (MSE between features)
        feature_loss = F.mse_loss(student_features, teacher_features)

        # Total loss
        total_loss = self.alpha * ce_loss + self.beta * kl_loss + self.gamma * feature_loss

        return total_loss, {
            'ce_loss': ce_loss.item(),
            'kl_loss': kl_loss.item(),
            'feature_loss': feature_loss.item(),
            'total_loss': total_loss.item()
        }


def load_dataset(split='train'):
    """Load dataset split."""
    df = pd.read_csv(ANNOTATIONS_CSV)
    split_df = df[df['image_path'].str.startswith(f"{split}/")]

    logger.info(f"{split.upper()}: {len(split_df)} images ({(split_df['label']==0).sum()} LOW / {(split_df['label']==1).sum()} HIGH)")
    return split_df


def calculate_class_weights(train_df):
    """Calculate class weights with boost for HIGH."""
    num_low = (train_df['label'] == 0).sum()
    num_high = (train_df['label'] == 1).sum()
    total = len(train_df)

    weight_low = total / (2 * num_low)
    weight_high = total / (2 * num_high)

    # Boost HIGH by 30% to ensure recall = 100%
    weight_high *= 1.3

    weights = torch.tensor([weight_low, weight_high], dtype=torch.float32)
    logger.info(f"Class weights: LOW={weight_low:.3f}, HIGH={weight_high:.3f} (HIGH boosted 30%)")

    return weights


def train_epoch(teacher, student, train_loader, criterion, optimizer, class_weights, device):
    """Train for one epoch."""
    teacher.eval()  # Teacher always in eval mode
    student.train()

    total_loss = 0.0
    losses_dict = {'ce_loss': 0, 'kl_loss': 0, 'feature_loss': 0, 'total_loss': 0}
    all_preds, all_labels = [], []

    for clip_input, student_input, labels in tqdm(train_loader, desc="Training"):
        clip_input = clip_input.to(device)
        student_input = student_input.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_features, teacher_logits = teacher(clip_input)

        # Student forward
        student_features, student_logits = student(student_input)

        # Loss
        loss, losses = criterion(student_features, student_logits, teacher_features, teacher_logits, labels, class_weights)

        # Backward
        loss.backward()
        optimizer.step()

        # Stats
        total_loss += loss.item()
        for k, v in losses.items():
            losses_dict[k] += v

        preds = torch.argmax(student_logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    # Average losses
    avg_loss = total_loss / len(train_loader)
    for k in losses_dict:
        losses_dict[k] /= len(train_loader)

    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average=None)

    return avg_loss, losses_dict, accuracy, recall


def validate(teacher, student, val_loader, criterion, class_weights, device):
    """Validate student model."""
    teacher.eval()
    student.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for clip_input, student_input, labels in tqdm(val_loader, desc="Validation"):
            clip_input = clip_input.to(device)
            student_input = student_input.to(device)
            labels = labels.to(device)

            # Teacher
            teacher_features, teacher_logits = teacher(clip_input)

            # Student
            student_features, student_logits = student(student_input)

            # Loss
            loss, _ = criterion(student_features, student_logits, teacher_features, teacher_logits, labels, class_weights)
            total_loss += loss.item()

            preds = torch.argmax(student_logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average=None)
    report = classification_report(all_labels, all_preds, target_names=['LOW', 'HIGH'], output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, accuracy, recall, report, cm


def main():
    """Main training function."""
    # Hyperparameters
    CLIP_MODEL = "openai/clip-vit-base-patch32"
    FEATURE_DIM = 768  # Match CLIP ViT-B/32 (768D)
    ALPHA = 0.3  # Hard labels
    BETA = 0.4   # Soft labels (KL)
    GAMMA = 0.3  # Feature matching
    TEMPERATURE = 4.0
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3

    logger.info("=" * 80)
    logger.info("KNOWLEDGE DISTILLATION: CLIP → ResNet18")
    logger.info("=" * 80)
    logger.info(f"\nHyperparameters:")
    logger.info(f"  Teacher: {CLIP_MODEL}")
    logger.info(f"  Student: ResNet18 (ImageNet pretrained)")
    logger.info(f"  Loss: α={ALPHA}, β={BETA}, γ={GAMMA}, T={TEMPERATURE}")
    logger.info(f"  Batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}, LR: {LEARNING_RATE}")

    # Load CLIP processor
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

    # Student transforms (standard ImageNet)
    student_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_df = load_dataset('train')
    val_df = load_dataset('val')
    test_df = load_dataset('test')

    train_dataset = ComplexityDataset(train_df, IMAGES_DIR, clip_processor, student_transform)
    val_dataset = ComplexityDataset(val_df, IMAGES_DIR, clip_processor, student_transform)
    test_dataset = ComplexityDataset(test_df, IMAGES_DIR, clip_processor, student_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Class weights
    class_weights = calculate_class_weights(train_df).to(DEVICE)

    # Models
    logger.info("\nInitializing models...")
    teacher = CLIPTeacher(CLIP_MODEL).to(DEVICE)
    student = ResNet18Student(pretrained=True, feature_dim=FEATURE_DIM).to(DEVICE)

    # Count parameters
    student_params = sum(p.numel() for p in student.parameters())
    student_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)

    logger.info(f"\nStudent parameters: {student_params:,} (trainable: {student_trainable:,})")

    # Loss & optimizer
    criterion = DistillationLoss(alpha=ALPHA, beta=BETA, gamma=GAMMA, temperature=TEMPERATURE)
    optimizer = optim.Adam(student.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    best_high_recall = 0.0
    history = []

    for epoch in range(NUM_EPOCHS):
        logger.info(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        logger.info("-" * 60)

        # Train
        train_loss, losses_dict, train_acc, train_recall = train_epoch(
            teacher, student, train_loader, criterion, optimizer, class_weights, DEVICE
        )

        logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        logger.info(f"  CE: {losses_dict['ce_loss']:.4f}, KL: {losses_dict['kl_loss']:.4f}, Feature: {losses_dict['feature_loss']:.4f}")
        logger.info(f"  Recall: LOW={train_recall[0]:.4f}, HIGH={train_recall[1]:.4f}")

        # Validate
        val_loss, val_acc, val_recall, val_report, val_cm = validate(
            teacher, student, val_loader, criterion, class_weights, DEVICE
        )

        logger.info(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        logger.info(f"  Recall: LOW={val_recall[0]:.4f}, HIGH={val_recall[1]:.4f}")
        logger.info(f"\n{val_cm}")

        scheduler.step(val_loss)

        # Save best
        if val_recall[1] > best_high_recall:
            best_high_recall = val_recall[1]
            torch.save({
                'epoch': epoch + 1,
                'student_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_recall_high': val_recall[1],
                'val_acc': val_acc
            }, MODEL_SAVE_DIR / "best_student.pth")

            logger.info(f"✅ NEW BEST! HIGH recall: {val_recall[1]:.4f}")

        # Early stop
        if val_recall[1] >= 0.999:
            logger.info(f"\n🎯 TARGET ACHIEVED! HIGH recall = {val_recall[1]:.4f}")
            break

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Best HIGH recall: {best_high_recall:.4f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
