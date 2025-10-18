#!/bin/bash
set -e

echo "================================================================================"
echo "📊 Evaluating Qwen2.5-7B-1024D Model"
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

# Run evaluation
python3 quick_eval_qwen25_7b_1024d.py

echo ""
echo "================================================================================"
echo "✅ Evaluation complete!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Review results above"
echo "  2. If score ≥ 91%, deploy: ./deploy_qwen25_7b.sh"
echo "  3. If score < 91%, consider re-distilling with different parameters"
echo ""
