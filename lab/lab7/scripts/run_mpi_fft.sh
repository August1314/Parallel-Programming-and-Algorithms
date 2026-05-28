#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAB_DIR="$(dirname "$SCRIPT_DIR")"
N="${1:-1024}"
NP="${2:-4}"
SEED="${3:-331.0}"
DUMP="${4:-}"

if [ ! -f "$LAB_DIR/bin/fft_mpi" ]; then
    echo "Building first..."
    make -C "$LAB_DIR" all
fi

mpirun -np "$NP" --oversubscribe "$LAB_DIR/bin/fft_mpi" "$N" "$SEED" $DUMP
