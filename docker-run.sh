#!/bin/bash

# Mavo Docker Run Script
# This script runs the Mavo container with optimal settings

set -e  # Exit on any error

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "📝 Please edit .env file with your configuration before running."
    echo ""
fi

# Check if image exists, build if not
if ! docker images mavo | grep -q latest; then
    echo "🛠️  Mavo image not found. Building..."
    ./docker-build.sh
fi

echo "🚀 Starting Mavo server..."

# Run with docker-compose (recommended)
if command -v docker-compose &> /dev/null; then
    echo "📦 Using docker-compose..."
    docker-compose up -d

    echo ""
    echo "✅ Server started successfully!"
    echo ""
    echo "🌐 Access the application at:"
    echo "   http://localhost:25500"
    echo "   http://localhost:25500/docs (API documentation)"
    echo "   http://localhost:25500/mavo (Web interface)"
    echo ""
    echo "📊 Check logs with: docker-compose logs -f"
    echo "🛑 Stop server with: docker-compose down"

else
    echo "🐳 Using docker run..."

    # GPU support detection
    GPU_ARGS=""
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        echo "🎮 GPU detected, enabling GPU support..."
        GPU_ARGS="--gpus all"
    fi

    # Run the container
    docker run -d \
        --name mavo-server \
        -p 25500:25500 \
        --env-file .env \
        -v "$(pwd)/uploads:/app/uploads" \
        -v "$(pwd)/temp:/app/temp" \
        -v "$(pwd)/persistent:/app/persistent" \
        $GPU_ARGS \
        --restart unless-stopped \
        mavo:latest

    echo ""
    echo "✅ Server started successfully!"
    echo ""
    echo "🌐 Access the application at:"
    echo "   http://localhost:25500"
    echo "   http://localhost:25500/docs (API documentation)"
    echo "   http://localhost:25500/mavo (Web interface)"
    echo ""
    echo "📊 Check logs with: docker logs -f mavo-server"
    echo "🛑 Stop server with: docker stop mavo-server && docker rm mavo-server"
fi
