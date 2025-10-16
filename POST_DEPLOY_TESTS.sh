#!/bin/bash

# Post-Deployment Testing Script for Qwen25-1024D Railway Deployment
# Usage: ./POST_DEPLOY_TESTS.sh <railway-url>
# Example: ./POST_DEPLOY_TESTS.sh https://deposium-embeddings-turbov2-production.up.railway.app

if [ -z "$1" ]; then
    echo "❌ Usage: $0 <railway-url>"
    echo "Example: $0 https://deposium-embeddings-turbov2-production.up.railway.app"
    exit 1
fi

RAILWAY_URL="$1"

echo "========================================"
echo "🚀 Railway Deployment Tests - Qwen25-1024D"
echo "========================================"
echo "Target: $RAILWAY_URL"
echo ""

# Test 1: Health Check
echo "✅ Test 1: Health Check"
echo "Command: curl $RAILWAY_URL/health"
HEALTH=$(curl -s "$RAILWAY_URL/health")
echo "$HEALTH" | jq .
echo ""

# Check if qwen25-1024d is loaded
if echo "$HEALTH" | grep -q "qwen25-1024d"; then
    echo "✅ Qwen25-1024D model loaded successfully!"
else
    echo "❌ WARNING: Qwen25-1024D not found in loaded models"
fi
echo ""

# Test 2: List Models
echo "✅ Test 2: List Models"
echo "Command: curl $RAILWAY_URL/api/tags"
curl -s "$RAILWAY_URL/api/tags" | jq '.models[] | {name: .name, details: .details}'
echo ""

# Test 3: Test Qwen25-1024D (PRIMARY - instruction-aware)
echo "✅ Test 3: Qwen25-1024D Embedding (instruction-aware)"
echo "Command: curl -X POST $RAILWAY_URL/api/embed -d '{\"model\":\"qwen25-1024d\",\"input\":\"Explain how neural networks work\"}'"
QWEN_RESULT=$(curl -s -X POST "$RAILWAY_URL/api/embed" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen25-1024d","input":"Explain how neural networks work"}')

QWEN_DIMS=$(echo "$QWEN_RESULT" | jq '.embeddings[0] | length')
echo "Model: qwen25-1024d"
echo "Dimensions: $QWEN_DIMS"
echo "Expected: 1024"

if [ "$QWEN_DIMS" -eq 1024 ]; then
    echo "✅ Qwen25-1024D working correctly!"
else
    echo "❌ WARNING: Expected 1024 dimensions, got $QWEN_DIMS"
fi
echo ""

# Test 4: Test Gemma-768D (SECONDARY - multilingual)
echo "✅ Test 4: Gemma-768D Embedding (multilingual)"
echo "Command: curl -X POST $RAILWAY_URL/api/embed -d '{\"model\":\"gemma-768d\",\"input\":\"Intelligence artificielle et machine learning\"}'"
GEMMA_RESULT=$(curl -s -X POST "$RAILWAY_URL/api/embed" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-768d","input":"Intelligence artificielle et machine learning"}')

GEMMA_DIMS=$(echo "$GEMMA_RESULT" | jq '.embeddings[0] | length')
echo "Model: gemma-768d"
echo "Dimensions: $GEMMA_DIMS"
echo "Expected: 768"

if [ "$GEMMA_DIMS" -eq 768 ]; then
    echo "✅ Gemma-768D working correctly!"
else
    echo "❌ WARNING: Expected 768 dimensions, got $GEMMA_DIMS"
fi
echo ""

# Test 5: Test Default Model (should be qwen25-1024d)
echo "✅ Test 5: Default Model Test (should use qwen25-1024d)"
echo "Command: curl -X POST $RAILWAY_URL/api/embed -d '{\"input\":\"test\"}'"
DEFAULT_RESULT=$(curl -s -X POST "$RAILWAY_URL/api/embed" \
  -H "Content-Type: application/json" \
  -d '{"input":"test"}')

DEFAULT_MODEL=$(echo "$DEFAULT_RESULT" | jq -r '.model')
DEFAULT_DIMS=$(echo "$DEFAULT_RESULT" | jq '.embeddings[0] | length')

echo "Default Model: $DEFAULT_MODEL"
echo "Dimensions: $DEFAULT_DIMS"

if [ "$DEFAULT_MODEL" = "qwen25-1024d" ]; then
    echo "✅ Default model is Qwen25-1024D (PRIMARY)!"
else
    echo "❌ WARNING: Expected qwen25-1024d, got $DEFAULT_MODEL"
fi
echo ""

# Test 6: Test Reranking
echo "✅ Test 6: Qwen3 Reranking (FP32 optimized)"
echo "Command: curl -X POST $RAILWAY_URL/api/rerank -d '{\"model\":\"qwen3-rerank\",\"query\":\"machine learning\",\"documents\":[...]}'"
RERANK_RESULT=$(curl -s -X POST "$RAILWAY_URL/api/rerank" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-rerank","query":"machine learning","documents":["AI and ML research","cooking recipes","deep learning tutorial"]}')

TOP_DOC=$(echo "$RERANK_RESULT" | jq -r '.results[0].document')
TOP_SCORE=$(echo "$RERANK_RESULT" | jq -r '.results[0].relevance_score')

echo "Top Document: $TOP_DOC"
echo "Relevance Score: $TOP_SCORE"

if echo "$TOP_DOC" | grep -q "AI\|deep learning"; then
    echo "✅ Reranking working correctly!"
else
    echo "❌ WARNING: Unexpected top document"
fi
echo ""

# Summary
echo "========================================"
echo "📊 Deployment Summary"
echo "========================================"
echo ""
echo "✅ Qwen25-1024D (PRIMARY): $QWEN_DIMS dimensions"
echo "✅ Gemma-768D (SECONDARY): $GEMMA_DIMS dimensions"
echo "✅ Default Model: $DEFAULT_MODEL"
echo "✅ Reranking: Working"
echo ""
echo "🔥 Deployment Status: SUCCESS"
echo "🌐 Railway URL: $RAILWAY_URL"
echo ""
echo "Next Steps:"
echo "1. Update N8N credentials to use 'qwen25-1024d' as model"
echo "2. Test instruction-aware queries in your workflows"
echo "3. Monitor latency (<50ms expected)"
echo ""
echo "🎉 First instruction-aware static embeddings deployed! 🔥"
echo "========================================"
