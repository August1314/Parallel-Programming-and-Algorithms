#!/bin/bash
# Run single convolution test
# Usage: ./run_conv.sh <H|W> <stride> <kernel_id> [block_x] [block_y]
#
# Examples:
#   ./run_conv.sh 256 1 1              # naive_conv, stride=1, default block
#   ./run_conv.sh 256 1 1 8 8          # naive_conv, 8x8 block
#   ./run_conv.sh 256 1 2              # tiled_conv
#   ./run_conv.sh 256 1 3              # im2col_gemm
#   ./run_conv.sh 256 1 4              # cuDNN

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAB_DIR="$(dirname "$SCRIPT_DIR")"

if [ $# -lt 3 ]; then
    echo "Usage: $0 <H|W> <stride> <kernel_id> [block_x] [block_y]"
    echo ""
    echo "  kernel_id: 1=naive_conv  2=tiled_conv  3=im2col_gemm  4=cudnn"
    echo "  block_x, block_y: block dimensions (only for naive_conv)"
    exit 1
fi

KERNEL_ID="$3"

# Detect cuDNN binary for kernel_id=4
if [ "$KERNEL_ID" = "4" ]; then
    BINARY="$LAB_DIR/bin/conv_cudnn"
    if [ ! -f "$BINARY" ]; then
        echo "Error: cuDNN binary not found at $BINARY"
        echo "Run: ./scripts/build.sh cudnn"
        exit 1
    fi
else
    BINARY="$LAB_DIR/bin/conv"
fi

if [ ! -f "$BINARY" ]; then
    echo "Error: binary not found at $BINARY"
    echo "Run ./scripts/build.sh first."
    exit 1
fi

exec "$BINARY" "$@"
