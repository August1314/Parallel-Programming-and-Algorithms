#!/bin/zsh

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: run_v6_docker.sh <m> <n> <k> [seed]" >&2
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DOCKERFILE="$ROOT_DIR/docker/v6-mkl.Dockerfile"
DOCKER_CONTEXT="$ROOT_DIR/docker"
IMAGE_TAG="lab1-v6-mkl:latest"

docker build \
    --platform linux/amd64 \
    -f "$DOCKERFILE" \
    -t "$IMAGE_TAG" \
    "$DOCKER_CONTEXT"

exec docker run \
    --platform linux/amd64 \
    --rm \
    "$IMAGE_TAG" \
    "$@"
