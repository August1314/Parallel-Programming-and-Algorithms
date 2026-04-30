#!/bin/zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
DOCKERFILE="$ROOT_DIR/lab/lab5/docker/lab5.Dockerfile"
DOCKER_CONTEXT="$ROOT_DIR/lab/lab5/docker"
IMAGE_TAG="parallel-programming-lab5:latest"

docker build \
    --platform linux/arm64 \
    -f "$DOCKERFILE" \
    -t "$IMAGE_TAG" \
    "$DOCKER_CONTEXT"
