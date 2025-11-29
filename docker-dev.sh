#!/bin/bash

# Mavo Docker Development Script
# This script runs the container in development mode with hot reload

set -e  # Exit on any error

echo "🔧 Starting Mavo in development mode..."

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "📝 Please edit .env file with your configuration."
    echo ""
fi

# Set development environment variables
export DEBUG=true

# Run with docker-compose in development mode
if command -v docker-compose &> /dev/null; then
    echo "📦 Using docker-compose for development..."

    # Override environment for development
    DEBUG=true docker-compose up --build

else
    echo "🐳 Using docker run for development..."

    # GPU support detection
    GPU_ARGS=""
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        echo "🎮 GPU detected, enabling GPU support..."
        GPU_ARGS="--gpus all"
    fi

    # Run in development mode (foreground, with logs)
    docker run \
        --name mavo-dev \
        -p 25500:25500 \
        --env-file .env \
        -e DEBUG=true \
        -v "$(pwd)/backend:/app/backend" \
        -v "$(pwd)/uploads:/app/uploads" \
        -v "$(pwd)/temp:/app/temp" \
        -v "$(pwd)/persistent:/app/persistent" \
        $GPU_ARGS \
        --rm \
        mavo:latest
fi
