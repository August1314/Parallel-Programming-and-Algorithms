#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAB_DIR="$(dirname "$SCRIPT_DIR")"
N="${1:-1024}"
SEED="${2:-331.0}"
DUMP="${3:-}"

if [ ! -f "$LAB_DIR/bin/fft_serial" ]; then
    echo "Building first..."
    make -C "$LAB_DIR" all
fi

"$LAB_DIR/bin/fft_serial" "$N" "$SEED" $DUMP
