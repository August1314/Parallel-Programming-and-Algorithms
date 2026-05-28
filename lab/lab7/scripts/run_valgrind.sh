#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB7_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LAB6_DIR="${LAB7_DIR}/../lab6"
REPO_DIR="$(cd "${LAB7_DIR}/../.." && pwd)"
OUTPUT_DIR="${LAB7_DIR}/results/valgrind"

mkdir -p "${OUTPUT_DIR}"

echo "=== Building lab6 Docker image ==="
"${LAB6_DIR}/scripts/docker_build_image.sh"

echo ""
echo "=== Running Valgrind massif on heated_plate ==="

SIZES=("128 128" "256 256")
THREADS=(1 2 4)

# Use a modified Docker image with Valgrind
docker build \
  --platform linux/arm64 \
  -t parallel-programming-lab6-valgrind:latest \
  -f - \
  "${REPO_DIR}" <<'DOCKERFILE'
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        valgrind \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/lab/lab6
DOCKERFILE

for size in "${SIZES[@]}"; do
    read m n <<< "$size"
    for t in "${THREADS[@]}"; do
        echo "  heated_plate_openmp ${m}x${n} threads=${t}"
        massif_out="massif_omp_${m}x${n}_t${t}.out"
        docker run \
          --platform linux/arm64 \
          --rm \
          -v "${REPO_DIR}:/workspace" \
          -w /workspace/lab/lab6 \
          parallel-programming-lab6-valgrind:latest \
          bash -c "make clean && make && \
            valgrind --tool=massif --stacks=yes --massif-out-file=/tmp/${massif_out} \
            ./bin/heated_plate_openmp ${m} ${n} 0.1 ${t} > /dev/null 2>&1 && \
            cat /tmp/${massif_out}" > "${OUTPUT_DIR}/${massif_out}"

        if [ -s "${OUTPUT_DIR}/${massif_out}" ]; then
            echo "    -> $(wc -c < "${OUTPUT_DIR}/${massif_out}") bytes"
        else
            echo "    -> FAILED"
        fi

        echo "  heated_plate_pthreads block ${m}x${n} threads=${t}"
        massif_out="massif_pthreads_block_${m}x${n}_t${t}.out"
        docker run \
          --platform linux/arm64 \
          --rm \
          -v "${REPO_DIR}:/workspace" \
          -w /workspace/lab/lab6 \
          parallel-programming-lab6-valgrind:latest \
          bash -c "make clean && make && \
            valgrind --tool=massif --stacks=yes --massif-out-file=/tmp/${massif_out} \
            ./bin/heated_plate_pthreads block ${m} ${n} 0.1 ${t} 8 > /dev/null 2>&1 && \
            cat /tmp/${massif_out}" > "${OUTPUT_DIR}/${massif_out}"

        if [ -s "${OUTPUT_DIR}/${massif_out}" ]; then
            echo "    -> $(wc -c < "${OUTPUT_DIR}/${massif_out}") bytes"
        else
            echo "    -> FAILED"
        fi
    done
done

echo ""
echo "=== Generating ms_print reports ==="
for f in "${OUTPUT_DIR}"/massif_*.out; do
    base=$(basename "$f" .out)
    echo "  ms_print ${base}"
    docker run \
      --platform linux/arm64 \
      --rm \
      -v "${REPO_DIR}:/workspace" \
      -v "${OUTPUT_DIR}/..:/tmp/results" \
      parallel-programming-lab6-valgrind:latest \
      bash -c "ms_print /tmp/results/valgrind/${base}.out" > "${OUTPUT_DIR}/${base}.txt" 2>&1 || true
done

echo ""
echo "=== Valgrind analysis complete ==="
echo "Results: ${OUTPUT_DIR}"
ls -la "${OUTPUT_DIR}"
