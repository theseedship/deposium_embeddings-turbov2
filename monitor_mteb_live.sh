#!/bin/bash

echo "=========================================="
echo "📊 MTEB Live Monitor"
echo "=========================================="
echo ""

# Check if process is running
PID=$(pgrep -f "mteb_evaluation.py")

if [ -z "$PID" ]; then
    echo "❌ MTEB not running!"
    echo ""
    echo "Check logs: tail -50 mteb_output.log"
    exit 1
fi

echo "✅ MTEB running (PID: $PID)"
echo ""

# Show last 20 lines of output
echo "📝 Last 20 lines of output:"
echo "----------------------------------------"
tail -20 mteb_output.log 2>/dev/null || echo "No log file yet"
echo "----------------------------------------"
echo ""

# Count completed tasks
completed=$(find mteb_results_quick -name "*test_results.json" 2>/dev/null | wc -l)
echo "Progress: $completed / 7 tasks completed"
echo ""

# List completed tasks
if [ $completed -gt 0 ]; then
    echo "✅ Completed tasks:"
    find mteb_results_quick -name "*test_results.json" -type f 2>/dev/null | while read file; do
        task=$(echo $file | sed 's|.*/\([^/]*\)/test_results.json|\1|')
        echo "  - $task"
    done
    echo ""
fi

echo "=========================================="
echo "Commands:"
echo "=========================================="
echo "Follow live output: tail -f mteb_output.log"
echo "Re-run this monitor: ./monitor_mteb_live.sh"
echo "Kill process: kill $PID"
echo ""

