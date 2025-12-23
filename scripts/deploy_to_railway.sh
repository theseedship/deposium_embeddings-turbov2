#!/bin/bash
# Deploy New VL Classifier to Railway
# Usage: ./scripts/deploy_to_railway.sh

set -e

echo "============================================================"
echo "Deploying New VL Classifier to Railway"
echo "============================================================"

# Model paths
NEW_MODEL="models/vl_distilled_resnet18/model_quantized.onnx"
MODEL_SIZE=$(du -h "$NEW_MODEL" | cut -f1)

echo ""
echo "Model to deploy:"
echo "  File: $NEW_MODEL"
echo "  Size: $MODEL_SIZE"
echo ""

# Check if model exists
if [ ! -f "$NEW_MODEL" ]; then
    echo "❌ Error: Model file not found: $NEW_MODEL"
    exit 1
fi

echo "✅ Model file verified"
echo ""

# Railway deployment options
echo "Deployment options:"
echo ""
echo "Option 1: Upload via Railway CLI"
echo "  railway volume cp $NEW_MODEL /data/models/vl_classifier/model_quantized.onnx"
echo ""
echo "Option 2: Upload via Railway API (requires token)"
echo "  Use Railway API to upload file to volume"
echo ""
echo "Option 3: Docker volume mount (local testing)"
echo "  docker cp $NEW_MODEL <container_id>:/data/models/vl_classifier/model_quantized.onnx"
echo ""

# Check for Railway CLI
if command -v railway &> /dev/null; then
    echo "Railway CLI detected!"
    echo ""
    read -p "Deploy now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Deploying to Railway..."

        # List volumes
        echo "Available volumes:"
        railway volume list

        # Copy file
        echo ""
        read -p "Enter volume ID or name: " VOLUME_ID
        railway volume cp "$NEW_MODEL" "$VOLUME_ID:/models/vl_classifier/model_quantized.onnx"

        echo "✅ Model deployed!"
    fi
else
    echo "❌ Railway CLI not installed"
    echo ""
    echo "Install with: npm i -g @railway/cli"
    echo "Then login: railway login"
fi

echo ""
echo "============================================================"
echo "Next steps:"
echo "1. Verify model uploaded to Railway volume"
echo "2. Delete old VL model if needed"
echo "3. Restart service to load new model"
echo "4. Test with sample documents"
echo "============================================================"
