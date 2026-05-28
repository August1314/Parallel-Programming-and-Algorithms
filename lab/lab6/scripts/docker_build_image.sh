#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

docker build \
  --platform linux/arm64 \
  -t parallel-programming-lab6:latest \
  -f "${LAB_DIR}/docker/lab6.Dockerfile" \
  "${LAB_DIR}"
