#!/bin/bash

# Define project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$PROJECT_ROOT/yolo_bot/bin/activate"

# Check if venv exists
if [ ! -f "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    echo "Please create it or update the path."
    exit 1
fi

# Activate venv
source "$VENV_PATH"

# Set PYTHONPATH to include the project root and submodules
# BoT-SORT: Needed for tracker modules
# RT-DETRv4-main: Needed for RT-DETR engine
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/BoT-SORT:$PROJECT_ROOT/RT-DETRv4-main"

# Check if a command was provided
if [ $# -eq 0 ]; then
    echo "Usage: ./run.sh <script_path> [args...]"
    echo "Example: ./run.sh custom_scripts/track_rtdetrv4.py"
    exit 1
fi

# Run python with arguments
python "$@"
