#!/bin/bash
set -e

echo "================================================================================"
echo "🔧 Configuration de l'Environnement - Qwen2.5-7B Distillation"
echo "================================================================================"
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "📦 Création de l'environnement virtuel..."
    python3 -m venv venv
    echo "✅ Environnement virtuel créé"
else
    echo "✅ Environnement virtuel existant trouvé"
fi
echo ""

# Activate venv
echo "🔧 Activation de l'environnement virtuel..."
source venv/bin/activate
echo "✅ Environnement virtuel activé"
echo ""

# Upgrade pip
echo "📦 Mise à jour de pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✅ pip mis à jour"
echo ""

# Install dependencies
echo "📦 Installation des dépendances..."
echo ""

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "⚠️  requirements.txt non trouvé, création..."
    cat > requirements.txt << 'EOF'
# Core dependencies for Qwen2.5-7B distillation
model2vec>=0.6.0
torch>=2.0.0
transformers>=4.50.0
numpy>=1.24.0
scikit-learn>=1.0.0

# Optional but recommended
huggingface-hub>=0.20.0
sentencepiece>=0.1.99
protobuf>=3.20.0

# For evaluation
mteb>=1.12.0
datasets>=2.14.0
sentence-transformers>=2.2.0
EOF
    echo "✅ requirements.txt créé"
fi

# Install
echo "Installing packages (this may take a few minutes)..."
pip install -r requirements.txt

echo ""
echo "================================================================================"
echo "✅ Installation terminée!"
echo "================================================================================"
echo ""

# Verify installation
echo "🔍 Vérification de l'installation..."
echo ""

python3 -c "import torch; print(f'✅ PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'✅ CUDA disponible: {torch.cuda.is_available()}')"
python3 -c "import model2vec; print('✅ model2vec: OK')"
python3 -c "import transformers; print(f'✅ transformers: {transformers.__version__}')"

echo ""
echo "================================================================================"
echo "🎉 Environnement prêt!"
echo "================================================================================"
echo ""

# Check hardware limitations
echo "⚠️  LIMITATIONS MATÉRIELLES DÉTECTÉES:"
echo ""

RAM_GB=$(free -g | awk '/^Mem:/ {print $2}')
if [ "$RAM_GB" -lt 32 ]; then
    echo "📊 RAM: ${RAM_GB}GB (recommandé: 32GB+)"
    echo "   → Distillation possible mais sera plus lente"
    echo "   → Considérez fermer d'autres applications"
fi

if command -v nvidia-smi &> /dev/null; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    GPU_MEM_GB=$((GPU_MEM / 1024))
    if [ "$GPU_MEM_GB" -lt 16 ]; then
        echo ""
        echo "🎮 VRAM: ${GPU_MEM_GB}GB (recommandé: 16GB+)"
        echo "   → Risque de Out Of Memory (OOM)"
        echo "   → SOLUTION: Utilisez le mode CPU"
        echo ""
        echo "   Pour forcer le mode CPU, éditez distill_qwen25_7b.py:"
        echo "   CONFIG = {"
        echo "       \"device\": \"cpu\",  # Force CPU mode"
        echo "   }"
        echo ""
        echo "   Temps estimé en mode CPU: 10-20 heures"
    fi
fi

echo ""
echo "================================================================================"
echo "📝 Prochaines étapes:"
echo "================================================================================"
echo ""
echo "1. Restez dans cet environnement virtuel"
echo "   (le prompt devrait montrer (venv))"
echo ""
echo "2. Vérifiez les prérequis complets:"
echo "   bash /tmp/check_prerequisites.sh"
echo ""
echo "3. Si vous avez <16GB VRAM, éditez distill_qwen25_7b.py"
echo "   pour forcer le mode CPU (voir instructions ci-dessus)"
echo ""
echo "4. Lancez la distillation:"
echo "   ./run_qwen25_7b_distillation.sh"
echo ""
echo "================================================================================"
echo ""

# Create a simple activation script for future use
cat > activate_env.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
echo "✅ Environnement virtuel activé"
echo ""
echo "Prêt pour la distillation:"
echo "  ./run_qwen25_7b_distillation.sh"
EOF

chmod +x activate_env.sh

echo "💡 ASTUCE: Pour réactiver l'environnement plus tard, utilisez:"
echo "   source activate_env.sh"
echo "   # ou"
echo "   source venv/bin/activate"
echo ""
