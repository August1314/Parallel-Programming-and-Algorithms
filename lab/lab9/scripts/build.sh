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

# Auto-detect GPU architecture, fallback to sm_80 (A100) or sm_86 (RTX 3090)
GPU_ARCH="sm_80"
if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -qi "3090\|3080\|3070\|3060"; then
    GPU_ARCH="sm_86"
elif nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -qi "A100"; then
    GPU_ARCH="sm_80"
elif nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -qi "H100\|H800"; then
    GPU_ARCH="sm_90"
fi

# Use C++14 for broader CUDA toolkit compatibility (works with CUDA 10.x+)
# Use C++17 only if available (CUDA 11.x+)
NVCC_FLAGS="-std=c++14 -O3 -arch=${GPU_ARCH} --ptxas-options=-v -lineinfo"
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
