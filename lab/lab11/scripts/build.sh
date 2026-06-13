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

    CUDNN_INCLUDE=""
    CUDNN_LIB=""
    CUDART_LIB=""

    # 1) Prefer pip-installed cuDNN (newer, supports RTX 3090 Ampere)
    PIP_CUDNN=$(python3 -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])" 2>/dev/null || echo "")
    if [ -n "$PIP_CUDNN" ] && [ -f "$PIP_CUDNN/include/cudnn.h" ]; then
        echo "  Found pip cuDNN at: $PIP_CUDNN"
        CUDNN_INCLUDE="$PIP_CUDNN/include"
        CUDNN_LIB="$PIP_CUDNN/lib"
        PIP_CUDART=$(python3 -c "import nvidia.cuda_runtime; print(nvidia.cuda_runtime.__path__[0])" 2>/dev/null || echo "")
        if [ -n "$PIP_CUDART" ]; then
            echo "  Found pip CUDA runtime at: $PIP_CUDART"
            CUDART_LIB="$PIP_CUDART/lib"
        fi
    fi

    # 2) Fallback: auto-find system cuDNN from common paths
    if [ -z "$CUDNN_INCLUDE" ]; then
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
    fi

    if [ ! -f "${CUDNN_INCLUDE}/cudnn.h" ]; then
        echo "Warning: cudnn.h not found."
        echo "  Install via: python3 -m pip install --user nvidia-cudnn-cu12==8.9.7.29"
        echo "  Or set CUDNN_ROOT environment variable."
        echo "  Skipping cuDNN binary."
        return 0
    fi
    echo "  cuDNN include: $CUDNN_INCLUDE"
    echo "  cuDNN lib:     $CUDNN_LIB"
    [ -n "$CUDART_LIB" ] && echo "  CUDA runtime:  $CUDART_LIB"

    LINK_FLAGS="-L${CUDNN_LIB} -lcudnn"
    if [ -n "$CUDART_LIB" ]; then
        LINK_FLAGS="-L${CUDART_LIB} ${LINK_FLAGS}"
    fi

    $NVCC $NVCC_BASE_FLAGS -arch=sm_37 \
        -I"${CUDNN_INCLUDE}" \
        ${LINK_FLAGS} \
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
