#!/bin/bash

# Retrieval-Only MTEB Evaluation for Qwen25-1024D
# Duration: ~1-2 hours
# Tasks: 15 retrieval tasks (most important for RAG/search)

set -e

echo "========================================"
echo "🔍 Retrieval MTEB Evaluation - Qwen25-1024D"
echo "========================================"
echo ""
echo "Duration: ~1-2 hours"
echo "Tasks: 15 retrieval tasks (most important for RAG/search)"
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
    MODEL_PATH="tss-deposium/qwen25-deposium-1024d"
else
    MODEL_PATH="models/qwen25-deposium-1024d"
fi

echo "Starting Retrieval-only MTEB evaluation..."
echo "These tasks are most important for RAG and search applications."
echo ""

# Run evaluation with retrieval tasks only
python3 mteb_evaluation.py \
    --model "$MODEL_PATH" \
    --output mteb_results_retrieval \
    --mode custom \
    --tasks \
        ArguAna \
        ClimateFEVER \
        CQADupstackRetrieval \
        DBPedia \
        FEVER \
        FiQA2018 \
        HotpotQA \
        MSMARCO \
        NFCorpus \
        NQ \
        QuoraRetrieval \
        SCIDOCS \
        SciFact \
        Touche2020 \
        TRECCOVID

echo ""
echo "========================================"
echo "✅ Retrieval MTEB Evaluation Complete!"
echo "========================================"
echo ""
echo "Results saved to: mteb_results_retrieval/"
echo ""
echo "Retrieval tasks are the most important for:"
echo "  - RAG (Retrieval-Augmented Generation)"
echo "  - Semantic search"
echo "  - Document retrieval"
echo "  - Q&A systems"
echo ""
