#!/bin/bash
# Run CUDA Hello World
# Usage: ./run_hello.sh [grid_x] [block_x] [block_y]
# Defaults: 2 4 4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAB_DIR="$(dirname "$SCRIPT_DIR")"
BIN="$LAB_DIR/bin/hello_world"

if [ ! -f "$BIN" ]; then
    echo "Error: $BIN not found. Run build.sh first."
    exit 1
fi

GRID_X="${1:-2}"
BLOCK_X="${2:-4}"
BLOCK_Y="${3:-4}"

echo "Running hello_world with grid=($GRID_X) block=($BLOCK_X, $BLOCK_Y)"
echo ""
$BIN "$GRID_X" "$BLOCK_X" "$BLOCK_Y"
