#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

# Ensure build directory exists and is configured
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    echo "Build directory not configured — running cmake..."
    cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
fi

# Rebuild using all available cores; stop on first error
echo "Building..."
make -C "$BUILD_DIR" -j"$(nproc)"

# Run from the project root so relative paths (config/, data/) resolve correctly
echo "Running..."
cd "$SCRIPT_DIR"
exec ./build/bev_warp
