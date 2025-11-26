#!/bin/bash
# Run the FastAPI server
# Usage: ./devenv/run_server.sh

# Get the absolute path of the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Get the absolute path of the project root (the directory containing the script directory)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"

# Check if the virtual environment exists
if [ ! -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    echo "Error: Virtual environment not found at $PROJECT_ROOT/venv/bin/activate"
    echo "Please run ./devenv/setup.sh first to create the virtual environment."
    exit 1
fi

# Activate virtual environment if not already activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    source "$PROJECT_ROOT/venv/bin/activate"
    echo "Virtual environment activated."
fi

# Load PORT from .env file if it exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
    echo "Loaded environment variables from .env file."
    echo "Using PORT: $PORT"
else
    # Default port if .env file doesn't exist
    export PORT=25500
    echo "No .env file found. Using default PORT: $PORT"
fi

# Set PYTHONPATH to include the project root
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"

# Change to backend directory and run server
# cd "$PROJECT_ROOT/backend"

## single thread
# uvicorn main:app --host ${HOST:-0.0.0.0} --port ${PORT:-25500} --reload 
uvicorn backend.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-25500} 

## multi thread - erraneous
# uvicorn main:app --host ${HOST:-0.0.0.0} --port ${PORT:-25500} --workers 8

# hypercorn main:app --bind ${HOST:-0.0.0.0}:${PORT:-25500} --worker-class uvloop --reload --workers 4 --log-level debug

# Note: The virtual environment will remain active after this script exits 