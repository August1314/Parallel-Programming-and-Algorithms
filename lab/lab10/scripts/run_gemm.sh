#!/bin/bash
# Run single GEMM test
# Usage: ./run_gemm.sh <M> <N> <K> <kernel_id> [block_x] [block_y]
#
# Examples:
#   ./run_gemm.sh 512 512 512 1 16 16    # naive kernel, 16×16 block
#   ./run_gemm.sh 1024 1024 1024 2        # tiled_16 kernel
#   ./run_gemm.sh 2048 2048 2048 3        # tiled_32 kernel

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAB_DIR="$(dirname "$SCRIPT_DIR")"
BINARY="$LAB_DIR/bin/gemm"

if [ ! -f "$BINARY" ]; then
    echo "Error: gemm binary not found at $BINARY"
    echo "Run ./scripts/build.sh first."
    exit 1
fi

if [ $# -lt 4 ]; then
    echo "Usage: $0 <M> <N> <K> <kernel_id> [block_x] [block_y]"
    echo ""
    echo "  kernel_id: 1=naive  2=tiled_16  3=tiled_32"
    echo "  block_x, block_y: block dimensions (default: kernel defaults)"
    exit 1
fi

"$BINARY" "$@"
