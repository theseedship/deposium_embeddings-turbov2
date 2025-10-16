#!/bin/bash

# Full MTEB Evaluation Script for Qwen25-1024D
# Duration: ~4-8 hours (CPU) or ~1-2 hours (GPU)
# Tasks: 58 complete MTEB tasks

set -e

echo "========================================"
echo "🚀 Full MTEB Evaluation - Qwen25-1024D"
echo "========================================"
echo ""
echo "Duration: ~4-8 hours (CPU) or ~1-2 hours (GPU)"
echo "Tasks: 58 complete MTEB tasks"
echo ""

# Warning
echo "⚠️  WARNING: This is a LONG evaluation!"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Check if venv exists
if [ ! -d "venv_mteb" ]; then
    echo "Creating MTEB virtual environment..."
    python3 -m venv venv_mteb
    echo "✅ Virtual environment created"
fi

# Activate venv
echo "Activating virtual environment..."
source venv_mteb/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -q -r requirements_mteb.txt
echo "✅ Requirements installed"
echo ""

# Check if model exists locally
if [ ! -d "models/qwen25-deposium-1024d" ]; then
    echo "⚠️  Model not found locally at models/qwen25-deposium-1024d"
    echo "The script will download from HuggingFace: tss-deposium/qwen25-deposium-1024d"
    echo ""
    MODEL_PATH="tss-deposium/qwen25-deposium-1024d"
else
    echo "✅ Model found locally at models/qwen25-deposium-1024d"
    MODEL_PATH="models/qwen25-deposium-1024d"
fi

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU detected, evaluation will be faster!"
    export CUDA_VISIBLE_DEVICES=0
else
    echo "ℹ️  No GPU detected, using CPU (slower)"
fi

echo ""
echo "Starting FULL MTEB evaluation..."
echo "This will take approximately 4-8 hours on CPU or 1-2 hours on GPU."
echo "Started at: $(date)"
echo ""

# Run evaluation
START_TIME=$(date +%s)

python3 mteb_evaluation.py \
    --model "$MODEL_PATH" \
    --output mteb_results_full \
    --mode full

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "========================================"
echo "✅ MTEB Evaluation Complete!"
echo "========================================"
echo ""
echo "Duration: ${HOURS}h ${MINUTES}m"
echo "Finished at: $(date)"
echo "Results saved to: mteb_results_full/"
echo ""
echo "To view results:"
echo "  cat mteb_results_full/qwen25-deposium-1024d_results.json | python3 -m json.tool | head -100"
echo ""
echo "To calculate average score:"
echo "  python3 -c \"import json; data=json.load(open('mteb_results_full/qwen25-deposium-1024d_results.json')); scores=[v['test']['main_score'] for v in data.values() if 'test' in v]; print(f'Average MTEB Score: {sum(scores)/len(scores):.4f}')\""
echo ""
