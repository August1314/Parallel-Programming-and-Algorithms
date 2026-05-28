#!/bin/bash
# Build script for lab9 CUDA programs
# Usage: ./build.sh [target]
#   all        - build everything (default)
#   hello      - build hello_world only
#   transpose  - build transpose only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAB_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$LAB_DIR/src"
BIN_DIR="$LAB_DIR/bin"

mkdir -p "$BIN_DIR"

# Detect nvcc
NVCC="${NVCC:-nvcc}"
if ! command -v "$NVCC" &>/dev/null; then
    echo "Error: nvcc not found. Is CUDA installed?"
    echo "Set NVCC environment variable to specify nvcc path."
    exit 1
fi

# CUDA architecture for RTX 3090 (Ampere SM 8.6)
# Also include older archs for broader compatibility
NVCC_FLAGS="-std=c++17 -O3 -arch=sm_86 --ptxas-options=-v -lineinfo"
NVCC_FLAGS="${NVCC_FLAGS} -D_FORTIFY_SOURCE=2"

echo "CUDA Compiler: $NVCC"
echo "Flags: $NVCC_FLAGS"
echo ""

build_hello() {
    echo "=== Building hello_world ==="
    $NVCC $NVCC_FLAGS -o "$BIN_DIR/hello_world" "$SRC_DIR/hello_world.cu"
    echo "  -> $BIN_DIR/hello_world"
}

build_transpose() {
    echo "=== Building transpose ==="
    $NVCC $NVCC_FLAGS -o "$BIN_DIR/transpose" "$SRC_DIR/transpose.cu"
    echo "  -> $BIN_DIR/transpose"
}

target="${1:-all}"
case "$target" in
    all)
        build_hello
        build_transpose
        ;;
    hello)
        build_hello
        ;;
    transpose)
        build_transpose
        ;;
    *)
        echo "Unknown target: $target"
        echo "Usage: $0 [all|hello|transpose]"
        exit 1
        ;;
esac

echo ""
echo "Build complete."
