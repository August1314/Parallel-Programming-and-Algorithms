#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "[build] Compiling..."
make -C "${LAB_DIR}" clean
make -C "${LAB_DIR}"

echo "[build] Done."
echo ""
echo "Binaries:"
ls -lh "${LAB_DIR}/bin/"
