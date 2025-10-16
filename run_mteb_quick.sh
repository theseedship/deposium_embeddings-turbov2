#!/bin/bash

# Quick MTEB Evaluation Script for Qwen25-1024D
# Duration: ~30 minutes
# Tasks: 7 representative tasks

set -e

echo "========================================"
echo "🚀 Quick MTEB Evaluation - Qwen25-1024D"
echo "========================================"
echo ""
echo "Duration: ~30 minutes"
echo "Tasks: 7 representative tasks"
echo ""

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

echo ""
echo "Starting MTEB evaluation..."
echo "This will take approximately 30 minutes."
echo ""

# Run evaluation
python3 mteb_evaluation.py \
    --model "$MODEL_PATH" \
    --output mteb_results_quick \
    --mode quick

echo ""
echo "========================================"
echo "✅ MTEB Evaluation Complete!"
echo "========================================"
echo ""
echo "Results saved to: mteb_results_quick/"
echo ""
echo "To view results:"
echo "  cat mteb_results_quick/qwen25-deposium-1024d_results.json | python3 -m json.tool"
echo ""
echo "To run full MTEB benchmark (4-8 hours):"
echo "  ./run_mteb_full.sh"
echo ""
