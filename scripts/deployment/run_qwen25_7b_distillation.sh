#!/bin/bash
set -e

echo "================================================================================"
echo "🚀 Qwen2.5-7B-Instruct → Model2Vec Distillation Pipeline"
echo "================================================================================"
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo ""
    echo "Please create one first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate venv
if [ -z "$VIRTUAL_ENV" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

echo "✅ Virtual environment active"
echo ""

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo "⚠️  No GPU detected - distillation will be SLOW!"
    echo ""
    read -p "Continue without GPU? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Aborted."
        exit 1
    fi
    echo ""
fi

# Check requirements
echo "📦 Checking dependencies..."
python3 -c "import model2vec; import torch; import transformers" 2>/dev/null || {
    echo "❌ Missing dependencies!"
    echo ""
    echo "Please install requirements:"
    echo "  pip install -r requirements.txt"
    exit 1
}
echo "✅ All dependencies installed"
echo ""

# Confirm start
echo "📋 Configuration:"
echo "  Source model: Qwen/Qwen2.5-7B-Instruct"
echo "  Target: Model2Vec 1024D (~65MB)"
echo "  Expected time: 2-4 hours (GPU) / 10-20+ hours (CPU)"
echo "  Expected quality: 91-95%"
echo ""

read -p "Start distillation? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "================================================================================"
echo "🔥 Starting distillation..."
echo "================================================================================"
echo ""

# Run distillation
python3 distill_qwen25_7b.py

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "✅ DISTILLATION SUCCESSFUL!"
    echo "================================================================================"
    echo ""
    echo "📁 Model saved to: models/qwen25-7b-deposium-1024d"
    echo ""
    echo "Next steps:"
    echo "  1. Test: ./test_qwen25_7b_model.sh"
    echo "  2. Evaluate: ./evaluate_qwen25_7b.sh"
    echo "  3. Deploy: ./deploy_qwen25_7b.sh"
    echo ""
else
    echo ""
    echo "================================================================================"
    echo "❌ DISTILLATION FAILED (exit code: $exit_code)"
    echo "================================================================================"
    echo ""
    echo "Check the error messages above for details."
    echo ""
    exit $exit_code
fi
