#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_DIR="${LAB_DIR}/data"
GRAPH="${1:?usage: run_openmp.sh <graph.csv> [queries.csv] [num_threads]}"
QUERIES="${2:-${DATA_DIR}/queries.txt}"
NUM_THREADS="${3:-4}"
OUTPUT="${4:-/dev/stdout}"

export OMP_NUM_THREADS="${NUM_THREADS}"
exec "${LAB_DIR}/bin/apsp_openmp" "${GRAPH}" "${QUERIES}" "${NUM_THREADS}" "${OUTPUT}"
