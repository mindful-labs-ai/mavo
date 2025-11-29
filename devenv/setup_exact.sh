#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up development environment with exact package versions...${NC}"

# Get the absolute path of the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Get the absolute path of the project root (the directory containing the script directory)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

echo -e "${GREEN}Script directory: $SCRIPT_DIR${NC}"
echo -e "${GREEN}Project root: $PROJECT_ROOT${NC}"

# Create virtual environment in project root
echo -e "${GREEN}Creating Python virtual environment in project root...${NC}"
cd "$PROJECT_ROOT"
python3 -m venv venv

# Verify the virtual environment was created
if [ ! -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    echo -e "${YELLOW}Error: Failed to create virtual environment at $PROJECT_ROOT/venv${NC}"
    exit 1
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source "$PROJECT_ROOT/venv/bin/activate"

# Install dependencies with exact versions
echo -e "${GREEN}Installing dependencies from requirements.txt with exact versions...${NC}"
# pip install -r requirements.txt
pip install -r ./devenv/requirements.freeze2.txt

# Check if .env file exists, create if it doesn't
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo -e "${GREEN}Creating default .env file...${NC}"
    cat > "$PROJECT_ROOT/.env" << EOF
PORT=25500
HOST=0.0.0.0
DEBUG=True

# Whisper settings
WHISPER_MODEL=tiny
# Set to False to avoid MPS compatibility issues on Mac
USE_GPU=False

# OpenAI API settings
OPENAI_API_KEY=
OPENAI_API_STT_MODEL=whisper-1

# Diarization settings
MAX_SPEAKERS=3
EOF
    echo -e "${GREEN}.env file created with default settings${NC}"
else
    echo -e "${GREEN}.env file already exists.${NC}"
fi

echo -e "${YELLOW}Setup complete with exact package versions!${NC}"
echo -e "${YELLOW}To activate the virtual environment in the future, run:${NC}"
echo -e "    ${GREEN}source devenv/activate.sh${NC}"
echo -e "${YELLOW}To run the server:${NC}"
echo -e "    ${GREEN}./devenv/run_server.sh${NC}"

# Keep the virtual environment active
echo -e "${GREEN}Virtual environment is now active.${NC}"
echo -e "${GREEN}You can proceed with development.${NC}" 