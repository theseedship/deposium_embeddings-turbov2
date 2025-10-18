#!/bin/bash
#
# Setup script for Qwen2.5-1.5B-Instruct Model2Vec Distillation
# Creates venv and installs dependencies with correct versions
#
# Usage: bash setup_qwen25.sh

set -e  # Exit on error

echo "================================================================================"
echo "🚀 Qwen2.5-1.5B Model2Vec Setup"
echo "================================================================================"

# Check Python version
echo ""
echo "📋 Checking Python version..."
PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "❌ python3 not found! Please install Python 3.9+"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "✅ Found Python: $PYTHON_VERSION"

# Check if Python >= 3.9
PYTHON_MAJOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.minor)')

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo "❌ Python 3.9+ required (found $PYTHON_VERSION)"
    exit 1
fi

# Create venv
VENV_DIR="venv_qwen25"
echo ""
echo "📦 Creating virtual environment: $VENV_DIR"

if [ -d "$VENV_DIR" ]; then
    echo "⚠️  Virtual environment already exists"
    read -p "   Delete and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "   Deleting existing venv..."
        rm -rf "$VENV_DIR"
    else
        echo "   Using existing venv"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo "✅ Virtual environment created"
else
    echo "✅ Using existing virtual environment"
fi

# Activate venv
echo ""
echo "🔌 Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo "✅ Virtual environment activated"

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✅ pip upgraded"

# Check if requirements file exists
if [ ! -f "requirements_qwen25.txt" ]; then
    echo "❌ requirements_qwen25.txt not found!"
    echo "   Make sure you're running this from the project root directory"
    exit 1
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies from requirements_qwen25.txt..."
echo "   This will install:"
echo "   - model2vec >= 0.6.0 (CRITICAL: fixes tokenizer bug)"
echo "   - torch == 2.6.0"
echo "   - transformers >= 4.50.0 (for Qwen2.5)"
echo "   - scikit-learn, numpy, etc."
echo ""
echo "⏳ This may take 5-10 minutes..."

pip install -r requirements_qwen25.txt

echo ""
echo "✅ Dependencies installed successfully!"

# Verify critical versions
echo ""
echo "🔍 Verifying installed versions..."

# Check model2vec version
MODEL2VEC_VERSION=$($PYTHON_CMD -c "import model2vec; print(model2vec.__version__)" 2>/dev/null || echo "NOT_INSTALLED")
if [ "$MODEL2VEC_VERSION" == "NOT_INSTALLED" ]; then
    echo "❌ model2vec not installed!"
    exit 1
fi

# Check if version >= 0.6.0
VERSION_MAJOR=$(echo $MODEL2VEC_VERSION | cut -d. -f1)
VERSION_MINOR=$(echo $MODEL2VEC_VERSION | cut -d. -f2)

if [ "$VERSION_MAJOR" -eq 0 ] && [ "$VERSION_MINOR" -lt 6 ]; then
    echo "❌ model2vec version $MODEL2VEC_VERSION is too old!"
    echo "   Need version >= 0.6.0 to fix tokenizer bug"
    exit 1
fi

echo "✅ model2vec: $MODEL2VEC_VERSION (>= 0.6.0 required)"

# Check torch
TORCH_VERSION=$($PYTHON_CMD -c "import torch; print(torch.__version__)" 2>/dev/null || echo "NOT_INSTALLED")
if [ "$TORCH_VERSION" == "NOT_INSTALLED" ]; then
    echo "❌ torch not installed!"
    exit 1
fi
echo "✅ torch: $TORCH_VERSION"

# Check transformers
TRANSFORMERS_VERSION=$($PYTHON_CMD -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "NOT_INSTALLED")
if [ "$TRANSFORMERS_VERSION" == "NOT_INSTALLED" ]; then
    echo "❌ transformers not installed!"
    exit 1
fi
echo "✅ transformers: $TRANSFORMERS_VERSION"

# Check CUDA availability
echo ""
echo "🔍 Checking CUDA availability..."
CUDA_AVAILABLE=$($PYTHON_CMD -c "import torch; print('Yes' if torch.cuda.is_available() else 'No')")
if [ "$CUDA_AVAILABLE" == "Yes" ]; then
    CUDA_VERSION=$($PYTHON_CMD -c "import torch; print(torch.version.cuda)")
    echo "✅ CUDA available: $CUDA_VERSION"
    echo "   Distillation will use GPU (10-20 minutes expected)"
else
    echo "⚠️  CUDA not available - will use CPU"
    echo "   Distillation will take 45-60 minutes"
fi

# Summary
echo ""
echo "================================================================================"
echo "✅ SETUP COMPLETE"
echo "================================================================================"
echo ""
echo "📊 Environment Summary:"
echo "   Virtual env:  $VENV_DIR"
echo "   Python:       $PYTHON_VERSION"
echo "   model2vec:    $MODEL2VEC_VERSION"
echo "   torch:        $TORCH_VERSION"
echo "   transformers: $TRANSFORMERS_VERSION"
echo "   CUDA:         $CUDA_AVAILABLE"
echo ""
echo "🚀 Next Steps:"
echo ""
echo "   1. Activate the virtual environment:"
echo "      source $VENV_DIR/bin/activate"
echo ""
echo "   2. Run distillation (~45-60 min CPU, ~10-20 min GPU):"
echo "      python3 distill_qwen25_1024d.py"
echo ""
echo "   3. After distillation, run evaluation (~2-3 min):"
echo "      python3 quick_eval_qwen25_1024d.py"
echo ""
echo "   4. Compare with other models (~3-5 min):"
echo "      python3 compare_qwen25_vs_all.py"
echo ""
echo "💡 Tips:"
echo "   - Monitor with: tail -f distill_qwen25_1024d.log"
echo "   - First download: ~3GB (Qwen2.5-1.5B model)"
echo "   - Final model: ~65MB (10x smaller!)"
echo ""
echo "================================================================================"
