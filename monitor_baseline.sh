#!/bin/bash

while true; do
    clear
    echo "=========================================="
    echo "📊 Baseline MTEB Progress Monitor"
    echo "=========================================="
    echo ""
    
    # Check if still running
    if pgrep -f test_mteb_baseline.py > /dev/null; then
        echo "✅ Process running (PID: $(pgrep -f test_mteb_baseline.py))"
    else
        echo "❌ Process finished!"
        echo ""
        echo "Final results:"
        python3 compare_baseline_vs_qwen25.py
        exit 0
    fi
    
    echo ""
    echo "Completed tasks:"
    completed=$(find mteb_results_baseline -name "*.json" -type f | grep -v model_meta | wc -l)
    echo "  $completed / 7"
    echo ""
    
    # List completed
    find mteb_results_baseline -name "*.json" -type f | grep -v model_meta | while read f; do
        echo "  ✅ $(basename $f .json)"
    done
    
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    echo "Refreshing in 5 seconds..."
    
    sleep 5
done
