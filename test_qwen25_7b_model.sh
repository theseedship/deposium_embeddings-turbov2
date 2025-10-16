#!/bin/bash
set -e

echo "================================================================================"
echo "🧪 Testing Qwen2.5-7B-1024D Model"
echo "================================================================================"
echo ""

# Activate venv
if [ -z "$VIRTUAL_ENV" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# Check if model exists
if [ ! -d "models/qwen25-7b-deposium-1024d" ]; then
    echo "❌ Model not found: models/qwen25-7b-deposium-1024d"
    echo ""
    echo "Please run distillation first:"
    echo "  ./run_qwen25_7b_distillation.sh"
    exit 1
fi

# Run tests
python3 test_qwen25_7b_model.py

echo ""
echo "================================================================================"
echo "✅ Tests complete!"
echo "================================================================================"
echo ""
echo "Next step: Run evaluation"
echo "  ./evaluate_qwen25_7b.sh"
echo ""
