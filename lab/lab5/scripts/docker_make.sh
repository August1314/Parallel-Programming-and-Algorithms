#!/bin/zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
IMAGE_TAG="parallel-programming-lab5:latest"

"$ROOT_DIR/lab/lab5/scripts/docker_build_image.sh" >/dev/null

exec docker run \
    --platform linux/arm64 \
    --rm \
    -v "$ROOT_DIR:/workspace" \
    -w /workspace/lab/lab5 \
    "$IMAGE_TAG" \
    make "$@"
