#!/bin/zsh

set -euo pipefail

if [[ $# -lt 6 ]]; then
    echo "usage: run_openmp_matmul.sh <default|static1|dynamic1> <m> <n> <k> <seed> <threads> [--dump]" >&2
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
IMAGE_TAG="parallel-programming-lab5:latest"
MODE="$1"
shift

case "$MODE" in
    default)
        TARGET="./bin/openmp_matmul_default"
        ;;
    static|static1)
        TARGET="./bin/openmp_matmul_static1"
        ;;
    dynamic|dynamic1)
        TARGET="./bin/openmp_matmul_dynamic1"
        ;;
    *)
        echo "unknown mode: $MODE" >&2
        exit 1
        ;;
esac

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
