#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAB_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Building lab7 (native) ==="
make -C "$LAB_DIR" clean
make -C "$LAB_DIR" all
echo "=== Build complete ==="
ls -la "$LAB_DIR/bin/"
