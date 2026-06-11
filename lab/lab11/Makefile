# Makefile for lab11 - CUDA Convolution
# Target platform: 智算习堂 (NVIDIA RTX 3090 / A100, CUDA)

NVCC ?= nvcc
NVCC_FLAGS := -std=c++14 -O3 -arch=sm_37

SRC_DIR := src
BIN_DIR := bin
SCRIPTS_DIR := scripts

TEST_HW := 256
TEST_STRIDE := 1

.PHONY: all clean conv run-conv help

all: conv

conv: $(BIN_DIR)/conv

$(BIN_DIR)/conv: $(SRC_DIR)/convolution.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

run-conv: $(BIN_DIR)/conv
	$(BIN_DIR)/conv $(TEST_HW) $(TEST_STRIDE) 1

clean:
	rm -rf $(BIN_DIR)

help:
	@echo "Lab11 CUDA Convolution"
	@echo "  make             - Build all"
	@echo "  make conv        - Build convolution (kernels 1-3)"
	@echo "  make run-conv    - Quick test (256x256, stride=1, naive)"
	@echo "  make clean       - Remove binaries"
	@echo ""
	@echo "For single test:   ./scripts/run_conv.sh <H|W> <stride> <kernel_id>"
	@echo "For full benchmark: uv run python scripts/benchmark.py"
