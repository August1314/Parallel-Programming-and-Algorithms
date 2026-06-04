#!/bin/bash
# Build script for lab10 CUDA GEMM
# Usage: ./build.sh [target]
#   all   - build everything (default)
#   gemm  - build gemm only
#
# Target platform: 智算习堂 (NVIDIA RTX 3090 / A100, CUDA)

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

# GPU architecture: sm_37 for platform CUDA toolkit compatibility
NVCC_FLAGS="-std=c++14 -O3 -arch=sm_37 --ptxas-options=-v -lineinfo"
NVCC_FLAGS="${NVCC_FLAGS} -D_FORTIFY_SOURCE=2"

echo "CUDA Compiler: $NVCC"
echo "Flags: $NVCC_FLAGS"
echo ""

build_gemm() {
    echo "=== Building gemm ==="
    $NVCC $NVCC_FLAGS -o "$BIN_DIR/gemm" "$SRC_DIR/gemm.cu"
    echo "  -> $BIN_DIR/gemm"
}

target="${1:-all}"
case "$target" in
    all|gemm)
        build_gemm
        ;;
    *)
        echo "Unknown target: $target"
        echo "Usage: $0 [all|gemm]"
        exit 1
        ;;
esac

echo ""
echo "Build complete."
