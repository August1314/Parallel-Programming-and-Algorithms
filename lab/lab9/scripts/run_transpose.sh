#!/bin/bash
# Run CUDA transpose single test
# Usage: ./run_transpose.sh <N> <kernel_id> [block_x] [block_y]
# Defaults: 1024 3 (shared_mem kernel)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAB_DIR="$(dirname "$SCRIPT_DIR")"
BIN="$LAB_DIR/bin/transpose"

if [ ! -f "$BIN" ]; then
    echo "Error: $BIN not found. Run build.sh first."
    exit 1
fi

N="${1:-1024}"
KERNEL_ID="${2:-3}"
BLOCK_X="${3:-32}"
BLOCK_Y="${4:-32}"

echo "Running transpose N=$N kernel=$KERNEL_ID block=${BLOCK_X}x${BLOCK_Y}"
echo ""
$BIN "$N" "$KERNEL_ID" "$BLOCK_X" "$BLOCK_Y"
