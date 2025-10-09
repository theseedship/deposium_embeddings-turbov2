#!/bin/bash

# Simplified Benchmark TurboX.v2 vs Qwen3-Embedding:0.6b
# No jq dependency - uses python for JSON parsing

set -e

echo "🔬 Embedding Models Benchmark"
echo "=============================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}📊 Test 1: Single Embedding Latency${NC}"
echo "======================================="
echo ""

# TurboX.v2 single embedding
echo -e "${YELLOW}🚀 TurboX.v2 (localhost:11435)${NC}"
TURBOV2_START=$(date +%s%3N)
TURBOV2_RESULT=$(curl -s -X POST http://localhost:11435/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model":"turbov2","input":"semantic search test"}')
TURBOV2_END=$(date +%s%3N)
TURBOV2_LATENCY=$((TURBOV2_END - TURBOV2_START))

TURBOV2_DIM=$(echo "$TURBOV2_RESULT" | python3 -c "import sys,json; data=json.load(sys.stdin); print(len(data['embeddings'][0]))")
echo "  ✅ Latency: ${TURBOV2_LATENCY}ms"
echo "  ✅ Dimensions: ${TURBOV2_DIM}"
echo ""

# Qwen3-Embedding single embedding
echo -e "${YELLOW}🐳 Qwen3-Embedding:0.6b (deposium-ollama:11434)${NC}"
QWEN_START=$(date +%s%3N)
QWEN_RESULT=$(docker exec deposium-ollama curl -s -X POST http://localhost:11434/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-embedding:0.6b","input":"semantic search test"}')
QWEN_END=$(date +%s%3N)
QWEN_LATENCY=$((QWEN_END - QWEN_START))

QWEN_DIM=$(echo "$QWEN_RESULT" | python3 -c "import sys,json; data=json.load(sys.stdin); print(len(data['embeddings'][0]))")
echo "  ✅ Latency: ${QWEN_LATENCY}ms"
echo "  ✅ Dimensions: ${QWEN_DIM}"
echo ""

# Calculate speedup
if [ "$TURBOV2_LATENCY" -gt 0 ]; then
    SPEEDUP=$(python3 -c "print(f'{$QWEN_LATENCY / $TURBOV2_LATENCY:.2f}')")
    echo -e "${GREEN}🏆 TurboX.v2 is ${SPEEDUP}x faster for single embeddings${NC}"
else
    echo -e "${YELLOW}⚠️  TurboX.v2 latency too low to measure accurately${NC}"
fi
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
  -d '{"model":"turbov2","input":["semantic search","vector database","machine learning","neural networks","deep learning"]}')
TURBOV2_BATCH_END=$(date +%s%3N)
TURBOV2_BATCH_LATENCY=$((TURBOV2_BATCH_END - TURBOV2_BATCH_START))

TURBOV2_BATCH_COUNT=$(echo "$TURBOV2_BATCH_RESULT" | python3 -c "import sys,json; data=json.load(sys.stdin); print(len(data['embeddings']))")
echo "  ✅ Total latency: ${TURBOV2_BATCH_LATENCY}ms"
echo "  ✅ Embeddings: ${TURBOV2_BATCH_COUNT}"
echo "  ✅ Per embedding: $((TURBOV2_BATCH_LATENCY / TURBOV2_BATCH_COUNT))ms avg"
echo ""

# Qwen3-Embedding batch (sequential)
echo -e "${YELLOW}🐳 Qwen3-Embedding:0.6b (sequential)${NC}"
QWEN_BATCH_START=$(date +%s%3N)

for text in "semantic search" "vector database" "machine learning" "neural networks" "deep learning"; do
  docker exec deposium-ollama curl -s -X POST http://localhost:11434/api/embed \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"qwen3-embedding:0.6b\",\"input\":\"$text\"}" > /dev/null
done

QWEN_BATCH_END=$(date +%s%3N)
QWEN_BATCH_LATENCY=$((QWEN_BATCH_END - QWEN_BATCH_START))

echo "  ✅ Total latency: ${QWEN_BATCH_LATENCY}ms"
echo "  ✅ Embeddings: 5"
echo "  ✅ Per embedding: $((QWEN_BATCH_LATENCY / 5))ms avg"
echo ""

# Calculate batch speedup
if [ "$TURBOV2_BATCH_LATENCY" -gt 0 ]; then
    BATCH_SPEEDUP=$(python3 -c "print(f'{$QWEN_BATCH_LATENCY / $TURBOV2_BATCH_LATENCY:.2f}')")
    echo -e "${GREEN}🏆 TurboX.v2 is ${BATCH_SPEEDUP}x faster for batch embeddings${NC}"
else
    echo -e "${YELLOW}⚠️  TurboX.v2 batch latency too low to measure accurately${NC}"
fi
echo ""

echo "======================================"
echo -e "${BLUE}📊 Test 3: Memory & CPU Usage${NC}"
echo "======================================"
echo ""

# TurboX.v2 memory
TURBOV2_MEM=$(docker stats deposium-embeddings-turbov2-test --no-stream --format "{{.MemUsage}}" | head -1 | awk '{print $1}')
echo -e "${YELLOW}🚀 TurboX.v2${NC}"
echo "  📦 Memory: ${TURBOV2_MEM}"
echo "  📏 Model size: ~30MB"
echo ""

# Qwen3-Embedding memory
QWEN_MEM=$(docker stats deposium-ollama --no-stream --format "{{.MemUsage}}" | head -1 | awk '{print $1}')
echo -e "${YELLOW}🐳 Qwen3-Embedding:0.6b${NC}"
echo "  📦 Memory: ${QWEN_MEM} (Ollama + models)"
echo "  📏 Model size: ~639MB"
echo ""

echo "======================================"
echo -e "${GREEN}📋 Summary${NC}"
echo "======================================"
echo ""
echo "Single embedding:"
echo "  - TurboX.v2:    ${TURBOV2_LATENCY}ms (${TURBOV2_DIM}D)"
echo "  - Qwen3:        ${QWEN_LATENCY}ms (${QWEN_DIM}D)"
echo ""
echo "Batch (5 texts):"
echo "  - TurboX.v2:    ${TURBOV2_BATCH_LATENCY}ms total (~$((TURBOV2_BATCH_LATENCY / TURBOV2_BATCH_COUNT))ms/embedding)"
echo "  - Qwen3:        ${QWEN_BATCH_LATENCY}ms total (~$((QWEN_BATCH_LATENCY / 5))ms/embedding)"
echo ""
echo "Memory usage:"
echo "  - TurboX.v2:    ${TURBOV2_MEM} (~30MB model)"
echo "  - Qwen3:        ${QWEN_MEM} (~639MB model)"
echo ""

if [ "$TURBOV2_LATENCY" -gt 0 ] && [ "$QWEN_LATENCY" -gt 0 ]; then
    SPEEDUP=$(python3 -c "print(f'{$QWEN_LATENCY / $TURBOV2_LATENCY:.1f}')")
    echo -e "${GREEN}✅ Benchmark complete!${NC}"
    echo ""

    if (( $(python3 -c "print(1 if $QWEN_LATENCY / $TURBOV2_LATENCY > 10 else 0)") )); then
        echo "💡 TurboX.v2 is ${SPEEDUP}x faster - highly recommended for CPU-only Railway"
    elif (( $(python3 -c "print(1 if $QWEN_LATENCY / $TURBOV2_LATENCY > 5 else 0)") )); then
        echo "💡 TurboX.v2 shows ${SPEEDUP}x improvement - good for CPU workloads"
    else
        echo "💡 TurboX.v2 provides ${SPEEDUP}x speedup - evaluate based on use case"
    fi
fi

echo ""
echo "Next steps:"
echo "1. Test embedding quality for your specific use case"
echo "2. Configure N8N to use TurboX.v2:"
echo "   - Base URL: http://deposium-embeddings-turbov2-test:11435"
echo "   - Model: turbov2"
echo "3. Deploy to Railway if performance is satisfactory"
