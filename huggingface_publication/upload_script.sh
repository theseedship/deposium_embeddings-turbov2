#!/bin/bash
# Script to upload all files to HuggingFace
# Usage: ./upload_script.sh

set -e

echo "🚀 Uploading to HuggingFace: tss-deposium/qwen25-deposium-1024d"
echo ""

# Check if huggingface_hub is installed
if ! pip show huggingface_hub > /dev/null 2>&1; then
    echo "📦 Installing huggingface_hub..."
    pip install huggingface_hub
fi

# Clone the repo (you'll need to login first if not done)
TEMP_DIR=$(mktemp -d)
echo "📥 Cloning repo to $TEMP_DIR..."
git clone https://huggingface.co/tss-deposium/qwen25-deposium-1024d "$TEMP_DIR"

# Copy files
echo "📋 Copying files..."
cp BENCHMARKS.md "$TEMP_DIR/"
cp QUICK_START.md "$TEMP_DIR/"
cp requirements.txt "$TEMP_DIR/"
cp ../HUGGINGFACE_README.md "$TEMP_DIR/README.md"

# Create examples folder
mkdir -p "$TEMP_DIR/examples"
cp examples/instruction_awareness_demo.py "$TEMP_DIR/examples/"
cp examples/real_world_use_cases.py "$TEMP_DIR/examples/"

# Commit and push
cd "$TEMP_DIR"
git add .
git commit -m "Add comprehensive documentation and examples

- README.md: Updated with YAML metadata (fixes metadata warning)
- BENCHMARKS.md: Detailed comparison with ColBERT, Gemma, Qwen3
- QUICK_START.md: 3-step quick start guide
- examples/instruction_awareness_demo.py: 5 interactive demos
- examples/real_world_use_cases.py: 5 real-world use cases
- requirements.txt: Dependencies

This is the first Model2Vec embedding distilled from an instruction-tuned LLM (Qwen2.5-Instruct).
Instruction-awareness: 94.96% | Code understanding: 84.5% | Conversational: 80%
"

echo "⬆️  Pushing to HuggingFace..."
git push

echo ""
echo "✅ Upload complete!"
echo "🔗 View at: https://huggingface.co/tss-deposium/qwen25-deposium-1024d"

# Cleanup
cd -
rm -rf "$TEMP_DIR"
