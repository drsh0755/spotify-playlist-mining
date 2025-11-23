#!/bin/bash
# Helper script to run Python scripts in screen sessions with logging

if [ $# -eq 0 ]; then
    echo "Usage: ./run_with_screen.sh <script_name> [session_name]"
    echo ""
    echo "Examples:"
    echo "  ./run_with_screen.sh 01_verify_data.py"
    echo "  ./run_with_screen.sh 01_verify_data.py verify_session"
    echo ""
    echo "After starting:"
    echo "  - Detach: Ctrl+A then D"
    echo "  - Reattach: screen -r <session_name>"
    echo "  - View logs: tail -f logs/<script>_*.log"
    exit 1
fi

SCRIPT=$1
SESSION_NAME=${2:-$(basename $SCRIPT .py)}

# Check if screen session already exists
if screen -list | grep -q "\.$SESSION_NAME\s"; then
    echo "Screen session '$SESSION_NAME' already exists!"
    echo "Options:"
    echo "  1. Reattach: screen -r $SESSION_NAME"
    echo "  2. Kill existing: screen -X -S $SESSION_NAME quit"
    exit 1
fi

echo "Starting screen session: $SESSION_NAME"
echo "Running script: $SCRIPT"
echo ""
echo "Commands:"
echo "  Detach from session: Ctrl+A then D"
echo "  Reattach to session: screen -r $SESSION_NAME"
echo "  Kill session: screen -X -S $SESSION_NAME quit"
echo "  List sessions: screen -ls"
echo ""
echo "Starting in 2 seconds..."
sleep 2

# Start screen session with the script
screen -dmS $SESSION_NAME bash -c "cd $(pwd) && source venv/bin/activate && python scripts/$SCRIPT; exec bash"

echo "Screen session '$SESSION_NAME' started!"
echo ""
echo "To view output: screen -r $SESSION_NAME"
echo "To view logs: tail -f logs/$(basename $SCRIPT .py)_*.log"
