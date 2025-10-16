#!/bin/bash
# Monitor Granite 4.0 Micro distillation progress

echo "=========================================="
echo "📊 Granite 4.0 Micro Distillation Monitor"
echo "=========================================="
echo ""

if [ ! -f granite-4.0-distillation.log ]; then
    echo "❌ Log file not found: granite-4.0-distillation.log"
    echo ""
    echo "Distillation not started yet?"
    echo "Start with: python3 distill_granite_4_0_micro.py > granite-4.0-distillation.log 2>&1 &"
    exit 1
fi

echo "📋 Monitoring: granite-4.0-distillation.log"
echo "   Press Ctrl+C to stop monitoring"
echo ""
echo "=========================================="
echo ""

tail -f granite-4.0-distillation.log
