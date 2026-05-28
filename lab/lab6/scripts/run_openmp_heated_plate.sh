#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 ]]; then
  echo "usage: run_openmp_heated_plate.sh <m> <n> <epsilon> <threads> [--dump]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${LAB_DIR}/../.." && pwd)"

docker run \
  --platform linux/arm64 \
  --rm \
  -v "${REPO_DIR}:/workspace" \
  -w /workspace/lab/lab6 \
  parallel-programming-lab6:latest \
  ./bin/heated_plate_openmp "$@"
