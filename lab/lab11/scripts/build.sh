#!/bin/bash
# Build script for lab11 CUDA Convolution
# Usage: ./build.sh [target]
#   conv   - build kernels 1-3 only (default)
#   cudnn  - build with cuDNN support (kernel 4)
#   all    - build both
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
echo ""

build_conv() {
    echo "=== Building convolution (kernels 1-3) ==="
    $NVCC $NVCC_FLAGS -o "$BIN_DIR/conv" "$SRC_DIR/convolution.cu"
    echo "  -> $BIN_DIR/conv"
}

build_cudnn() {
    echo "=== Building convolution with cuDNN ==="
    # Try to find cuDNN
    CUDNN_ROOT="${CUDNN_ROOT:-/usr/local/cuda}"
    CUDNN_INCLUDE="${CUDNN_ROOT}/include"
    CUDNN_LIB="${CUDNN_ROOT}/lib64"

    if [ ! -f "${CUDNN_INCLUDE}/cudnn.h" ]; then
        echo "Warning: cudnn.h not found at ${CUDNN_INCLUDE}/cudnn.h"
        echo "  Set CUDNN_ROOT environment variable to cuDNN installation path."
        echo "  Skipping cuDNN binary."
        return 0
    fi

    # cuDNN requires newer arch (sm_50 minimum); use sm_86 for Ampere
    $NVCC $NVCC_FLAGS -arch=sm_86 \
        -I"${CUDNN_INCLUDE}" -L"${CUDNN_LIB}" -lcudnn \
        -DUSE_CUDNN \
        -o "$BIN_DIR/conv_cudnn" "$SRC_DIR/convolution.cu"
    echo "  -> $BIN_DIR/conv_cudnn"
}

target="${1:-conv}"
case "$target" in
    conv)
        build_conv
        ;;
    cudnn)
        build_cudnn
        ;;
    all)
        build_conv
        build_cudnn
        ;;
    *)
        echo "Unknown target: $target"
        echo "Usage: $0 [conv|cudnn|all]"
        exit 1
        ;;
esac

echo ""
echo "Build complete."
