#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${LAB_DIR}/../.." && pwd)"

"${SCRIPT_DIR}/docker_build_image.sh"

docker run \
  --platform linux/arm64 \
  --rm \
  -v "${REPO_DIR}:/workspace" \
  -w /workspace/lab/lab6 \
  parallel-programming-lab6:latest \
  make
