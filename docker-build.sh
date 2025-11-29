#!/bin/bash

# Mavo Docker Build Script
# This script builds the Docker image with optimizations

set -e  # Exit on any error

echo "🏗️  Building Mavo Docker image..."

# Enable BuildKit for better performance
export DOCKER_BUILDKIT=1

# Build the image with multi-platform support (optional)
# docker buildx create --use --name mavo-builder 2>/dev/null || true

# Build the image
docker build \
    --target runtime \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --tag mavo:latest \
    --tag mavo:$(date +%Y%m%d-%H%M%S) \
    .

echo "✅ Build completed successfully!"
echo ""
echo "📋 Available images:"
docker images mavo --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

echo ""
echo "🚀 To run the container:"
echo "   docker compose up -d"
echo "   # or"
echo "   docker run -p 25500:25500 mavo:latest"
