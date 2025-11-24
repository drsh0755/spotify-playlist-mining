#!/bin/bash
# Helper to view logs

echo "Available log files:"
echo ""
ls -lht logs/*.log 2>/dev/null | head -10

echo ""
echo "=========================================="
echo ""

if [ $# -eq 0 ]; then
    # Show most recent log
    LATEST_LOG=$(ls -t logs/*.log 2>/dev/null | head -1)
    if [ -z "$LATEST_LOG" ]; then
        echo "No log files found in logs/"
        exit 1
    fi
    echo "Showing latest log: $LATEST_LOG"
    echo "Use: ./view_logs.sh <log_file> to view specific log"
    echo "Use: ./view_logs.sh -f to follow latest log"
    echo ""
    echo "=========================================="
    echo ""
    cat "$LATEST_LOG"
else
    if [ "$1" == "-f" ]; then
        LATEST_LOG=$(ls -t logs/*.log 2>/dev/null | head -1)
        echo "Following log: $LATEST_LOG"
        echo "Press Ctrl+C to stop"
        echo ""
        tail -f "$LATEST_LOG"
    else
        cat "logs/$1"
    fi
fi
