#!/bin/zsh

set -euo pipefail

if [[ $# -lt 5 ]]; then
    echo "usage: run_parallel_for_matmul.sh <m> <n> <k> <seed> <threads> [--dump]" >&2
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
IMAGE_TAG="parallel-programming-lab5:latest"
TARGET="./bin/parallel_for_matmul"

if [[ ! -x "$ROOT_DIR/lab/lab5/${TARGET#./}" ]]; then
    echo "missing executable: $ROOT_DIR/lab/lab5/${TARGET#./}. run ./lab/lab5/scripts/docker_make.sh first." >&2
    exit 1
fi

exec docker run \
    --platform linux/arm64 \
    --rm \
    -v "$ROOT_DIR:/workspace" \
    -w /workspace/lab/lab5 \
    "$IMAGE_TAG" \
    "$TARGET" "$@"
