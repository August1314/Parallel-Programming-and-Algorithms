#!/bin/zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$ROOT_DIR/src"
BIN_DIR="$ROOT_DIR/bin"
CXX="${CXX:-clang++}"
CXXFLAGS_COMMON=(-std=c++17 -Wall -Wextra -pedantic)

mkdir -p "$BIN_DIR"

echo "Building v2_cpp_baseline"
"$CXX" "${CXXFLAGS_COMMON[@]}" -O0 "$SRC_DIR/v2_cpp_baseline.cpp" -o "$BIN_DIR/v2_cpp_baseline"

echo "Building v3_cpp_loop_reorder"
"$CXX" "${CXXFLAGS_COMMON[@]}" -O0 "$SRC_DIR/v3_cpp_loop_reorder.cpp" -o "$BIN_DIR/v3_cpp_loop_reorder"

echo "Building v4_cpp_optimized"
"$CXX" "${CXXFLAGS_COMMON[@]}" -O3 "$SRC_DIR/v4_cpp_optimized.cpp" -o "$BIN_DIR/v4_cpp_optimized"

echo "Building v5_cpp_unrolled"
"$CXX" "${CXXFLAGS_COMMON[@]}" -O3 "$SRC_DIR/v5_cpp_unrolled.cpp" -o "$BIN_DIR/v5_cpp_unrolled"

echo "Building v6_mkl"
if pkg-config --exists mkl-static-lp64 2>/dev/null; then
    MKL_FLAGS=("${(@f)$(pkg-config --cflags --libs mkl-static-lp64)}")
    "$CXX" "${CXXFLAGS_COMMON[@]}" -O3 "$SRC_DIR/v6_mkl.cpp" "${MKL_FLAGS[@]}" -o "$BIN_DIR/v6_mkl"
else
    "$CXX" "${CXXFLAGS_COMMON[@]}" -O3 "$SRC_DIR/v6_mkl.cpp" -o "$BIN_DIR/v6_mkl"
fi

chmod +x "$SRC_DIR/v1_python.py"
echo "Build complete"
