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

# Base flags (arch is set per target)
NVCC_BASE_FLAGS="-std=c++14 -O3 --ptxas-options=-v -lineinfo"
NVCC_BASE_FLAGS="${NVCC_BASE_FLAGS} -D_FORTIFY_SOURCE=2"

echo "CUDA Compiler: $NVCC"
echo ""

build_conv() {
    echo "=== Building convolution (kernels 1-3) ==="
    $NVCC $NVCC_BASE_FLAGS -arch=sm_37 -o "$BIN_DIR/conv" "$SRC_DIR/convolution.cu"
    echo "  -> $BIN_DIR/conv"
}

build_cudnn() {
    echo "=== Building convolution with cuDNN ==="
    # Auto-find cuDNN from common paths
    if [ -z "${CUDNN_ROOT:-}" ]; then
        for d in /usr/local/cuda /usr/local/cuda-*; do
            if [ -f "$d/include/cudnn.h" ]; then
                CUDNN_ROOT="$d"
                break
            fi
        done
    fi
    CUDNN_INCLUDE="${CUDNN_ROOT:-}/include"
    CUDNN_LIB="${CUDNN_ROOT:-}/lib64"

    if [ ! -f "${CUDNN_INCLUDE}/cudnn.h" ]; then
        echo "Warning: cudnn.h not found."
        echo "  Searched: /usr/local/cuda /usr/local/cuda-*"
        echo "  Set CUDNN_ROOT environment variable."
        echo "  Skipping cuDNN binary."
        return 0
    fi
    echo "  Found cuDNN at: $CUDNN_ROOT"

    # Detect GPU compute capability, fallback to sm_75
    GPU_ARCH="${GPU_ARCH:-}"
    if [ -z "$GPU_ARCH" ]; then
        GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
            | head -1 | tr -d '.')
        GPU_ARCH="${GPU_ARCH:-75}"
        GPU_ARCH="sm_${GPU_ARCH}"
    fi
    echo "  Target arch: $GPU_ARCH"

    $NVCC $NVCC_BASE_FLAGS -arch="$GPU_ARCH" \
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
