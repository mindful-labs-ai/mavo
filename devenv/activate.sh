#!/bin/bash
# Source this file to activate the virtual environment
# Usage: source devenv/activate.sh

# Get the absolute path of the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Get the absolute path of the project root (the directory containing the script directory)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"

# Check if the virtual environment exists
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
    echo "Virtual environment activated. Use 'deactivate' to exit."
else
    echo "Error: Virtual environment not found at $PROJECT_ROOT/venv/bin/activate"
    echo "Please run ./devenv/setup.sh first to create the virtual environment."
    return 1
fi 