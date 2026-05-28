#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${LAB_DIR}/../.." && pwd)"

N="${1:-1024}"
NP="${2:-4}"
SEED="${3:-331.0}"

"${SCRIPT_DIR}/docker_build_image.sh"

docker run \
  --platform linux/arm64 \
  --rm \
  -v "${REPO_DIR}:/workspace" \
  -w /workspace/lab/lab7 \
  parallel-programming-lab7:latest \
  mpirun --allow-run-as-root --oversubscribe -np "${NP}" ./bin/fft_mpi "${N}" "${SEED}"
