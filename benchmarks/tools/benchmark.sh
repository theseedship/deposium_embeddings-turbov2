#!/bin/bash

# Benchmark TurboX.v2 vs Qwen3-Embedding:0.6b
# Compare latency, throughput, and resource usage

set -e

echo "🔬 Embedding Models Benchmark"
echo "=============================="
echo ""

# Test text samples
SINGLE_TEXT="semantic search test"
BATCH_TEXTS='["semantic search", "vector database", "machine learning", "neural networks", "deep learning"]'

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}📊 Test 1: Single Embedding Latency${NC}"
echo "======================================="
echo ""

# TurboX.v2 single embedding
echo -e "${YELLOW}🚀 TurboX.v2 (localhost:11435)${NC}"
TURBOV2_START=$(date +%s%3N)
TURBOV2_RESULT=$(curl -s -X POST http://localhost:11435/api/embed \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"turbov2\",\"input\":\"$SINGLE_TEXT\"}")
TURBOV2_END=$(date +%s%3N)
TURBOV2_LATENCY=$((TURBOV2_END - TURBOV2_START))

TURBOV2_DIM=$(echo "$TURBOV2_RESULT" | jq -r '.embeddings[0] | length')
echo "  ✅ Latency: ${TURBOV2_LATENCY}ms"
echo "  ✅ Dimensions: ${TURBOV2_DIM}"
echo ""

# Qwen3-Embedding single embedding
echo -e "${YELLOW}🐳 Qwen3-Embedding:0.6b (deposium-ollama:11434)${NC}"
QWEN_START=$(date +%s%3N)
QWEN_RESULT=$(docker exec deposium-ollama curl -s -X POST http://localhost:11434/api/embed \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"qwen3-embedding:0.6b\",\"input\":\"$SINGLE_TEXT\"}")
QWEN_END=$(date +%s%3N)
QWEN_LATENCY=$((QWEN_END - QWEN_START))

QWEN_DIM=$(echo "$QWEN_RESULT" | jq -r '.embeddings[0] | length')
echo "  ✅ Latency: ${QWEN_LATENCY}ms"
echo "  ✅ Dimensions: ${QWEN_DIM}"
echo ""

# Calculate speedup
SPEEDUP=$(echo "scale=2; $QWEN_LATENCY / $TURBOV2_LATENCY" | bc)
echo -e "${GREEN}🏆 TurboX.v2 is ${SPEEDUP}x faster for single embeddings${NC}"
echo ""

echo "======================================"
echo -e "${BLUE}📊 Test 2: Batch Embeddings (5 texts)${NC}"
echo "======================================"
echo ""

# TurboX.v2 batch
echo -e "${YELLOW}🚀 TurboX.v2 (batch)${NC}"
TURBOV2_BATCH_START=$(date +%s%3N)
TURBOV2_BATCH_RESULT=$(curl -s -X POST http://localhost:11435/api/embed \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"turbov2\",\"input\":$BATCH_TEXTS}")
TURBOV2_BATCH_END=$(date +%s%3N)
TURBOV2_BATCH_LATENCY=$((TURBOV2_BATCH_END - TURBOV2_BATCH_START))

TURBOV2_BATCH_COUNT=$(echo "$TURBOV2_BATCH_RESULT" | jq -r '.embeddings | length')
echo "  ✅ Latency: ${TURBOV2_BATCH_LATENCY}ms"
echo "  ✅ Embeddings: ${TURBOV2_BATCH_COUNT}"
echo "  ✅ Per embedding: $((TURBOV2_BATCH_LATENCY / TURBOV2_BATCH_COUNT))ms avg"
echo ""

# Qwen3-Embedding batch (note: might not support batch, test individually)
echo -e "${YELLOW}🐳 Qwen3-Embedding:0.6b (batch simulation)${NC}"
QWEN_BATCH_START=$(date +%s%3N)
QWEN_BATCH_COUNT=0

for text in "semantic search" "vector database" "machine learning" "neural networks" "deep learning"; do
  docker exec deposium-ollama curl -s -X POST http://localhost:11434/api/embed \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"qwen3-embedding:0.6b\",\"input\":\"$text\"}" > /dev/null
  ((QWEN_BATCH_COUNT++))
done

QWEN_BATCH_END=$(date +%s%3N)
QWEN_BATCH_LATENCY=$((QWEN_BATCH_END - QWEN_BATCH_START))

echo "  ✅ Total latency: ${QWEN_BATCH_LATENCY}ms"
echo "  ✅ Embeddings: ${QWEN_BATCH_COUNT}"
echo "  ✅ Per embedding: $((QWEN_BATCH_LATENCY / QWEN_BATCH_COUNT))ms avg"
echo ""

# Calculate batch speedup
BATCH_SPEEDUP=$(echo "scale=2; $QWEN_BATCH_LATENCY / $TURBOV2_BATCH_LATENCY" | bc)
echo -e "${GREEN}🏆 TurboX.v2 is ${BATCH_SPEEDUP}x faster for batch embeddings${NC}"
echo ""

echo "======================================"
echo -e "${BLUE}📊 Test 3: Memory & CPU Usage${NC}"
echo "======================================"
echo ""

# TurboX.v2 memory
TURBOV2_MEM=$(docker stats deposium-embeddings-turbov2-test --no-stream --format "{{.MemUsage}}" | awk '{print $1}')
echo -e "${YELLOW}🚀 TurboX.v2${NC}"
echo "  📦 Memory: ${TURBOV2_MEM}"
echo "  📏 Model size: ~30MB"
echo ""

# Qwen3-Embedding memory
QWEN_MEM=$(docker stats deposium-ollama --no-stream --format "{{.MemUsage}}" | awk '{print $1}')
echo -e "${YELLOW}🐳 Qwen3-Embedding:0.6b${NC}"
echo "  📦 Memory: ${QWEN_MEM} (Ollama + model)"
echo "  📏 Model size: ~639MB"
echo ""

echo "======================================"
echo -e "${GREEN}📋 Summary${NC}"
echo "======================================"
echo ""
echo "┌─────────────────────────┬──────────────┬────────────────────┐"
echo "│ Metric                  │ TurboX.v2    │ Qwen3-Embedding    │"
echo "├─────────────────────────┼──────────────┼────────────────────┤"
echo "│ Single embedding        │ ${TURBOV2_LATENCY}ms        │ ${QWEN_LATENCY}ms             │"
echo "│ Batch (5 texts)         │ ${TURBOV2_BATCH_LATENCY}ms       │ ${QWEN_BATCH_LATENCY}ms            │"
echo "│ Speedup (single)        │              │ ${SPEEDUP}x slower        │"
echo "│ Speedup (batch)         │              │ ${BATCH_SPEEDUP}x slower       │"
echo "│ Embedding dimensions    │ ${TURBOV2_DIM}         │ ${QWEN_DIM}                │"
echo "│ Model size              │ ~30MB        │ ~639MB             │"
echo "│ Memory usage            │ ${TURBOV2_MEM}      │ ${QWEN_MEM}          │"
echo "└─────────────────────────┴──────────────┴────────────────────┘"
echo ""

echo -e "${GREEN}✅ Benchmark complete!${NC}"
echo ""
echo "💡 Recommendations:"
if (( $(echo "$SPEEDUP > 10" | bc -l) )); then
  echo "  🚀 TurboX.v2 is significantly faster - recommended for CPU-only Railway"
elif (( $(echo "$SPEEDUP > 5" | bc -l) )); then
  echo "  ⚡ TurboX.v2 shows good performance improvement for CPU workloads"
else
  echo "  ⚖️  Performance gain is moderate - consider use case requirements"
fi

echo ""
echo "Next steps:"
echo "1. Test embedding quality/similarity for your specific use case"
echo "2. Configure N8N to use TurboX.v2 (http://deposium-embeddings-turbov2-test:11435)"
echo "3. Deploy to Railway if performance is satisfactory"
