#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_DIR="${LAB_DIR}/data"
GRAPH="${1:?usage: run_serial.sh <graph.csv> [queries.csv]}"
QUERIES="${2:-${DATA_DIR}/queries.txt}"
OUTPUT="${3:-/dev/stdout}"

exec "${LAB_DIR}/bin/apsp_serial" "${GRAPH}" "${QUERIES}" "${OUTPUT}"
