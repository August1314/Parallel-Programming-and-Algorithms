# Makefile for lab10 - CUDA Matrix Multiplication (GEMM)
# Target platform: 智算习堂 (NVIDIA RTX 3090 / A100, CUDA)

NVCC ?= nvcc
NVCC_FLAGS := -std=c++14 -O3 -arch=sm_37

SRC_DIR := src
BIN_DIR := bin
SCRIPTS_DIR := scripts

# Default sizes for quick test
TEST_M := 512
TEST_N := 512
TEST_K := 512

.PHONY: all clean gemm run-gemm

all: gemm

gemm: $(BIN_DIR)/gemm

$(BIN_DIR)/gemm: $(SRC_DIR)/gemm.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

run-gemm: $(BIN_DIR)/gemm
	$(BIN_DIR)/gemm $(TEST_M) $(TEST_N) $(TEST_K) 1 16 16

clean:
	rm -rf $(BIN_DIR)

help:
	@echo "Lab10 CUDA GEMM (Matrix Multiplication)"
	@echo "  make             - Build all"
	@echo "  make gemm        - Build gemm"
	@echo "  make run-gemm    - Quick test (512x512x512, naive 16x16)"
	@echo "  make clean       - Remove binaries"
	@echo ""
	@echo "For full benchmark: uv run python scripts/benchmark.py"
	@echo "For single test:   ./scripts/run_gemm.sh <M> <N> <K> <kernel_id> [bx] [by]"
