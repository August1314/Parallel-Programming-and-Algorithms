#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 5 ]]; then
  echo "usage: run_pthreads_heated_plate.sh <block|cyclic|dynamic> <m> <n> <epsilon> <threads> [chunk_size] [--dump]" >&2
  exit 1
fi

SCHEDULE="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${LAB_DIR}/../.." && pwd)"

docker run \
  --platform linux/arm64 \
  --rm \
  -v "${REPO_DIR}:/workspace" \
  -w /workspace/lab/lab6 \
  parallel-programming-lab6:latest \
  ./bin/heated_plate_pthreads "$1" "$2" "$3" "$4" "${SCHEDULE}" "${@:5}"
