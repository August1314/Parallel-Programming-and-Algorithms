# Makefile for lab9 - CUDA Matrix Transpose
# Target platform: 智算习堂 (NVIDIA RTX 3090, CUDA)

NVCC ?= nvcc
NVCC_FLAGS := -std=c++14 -O3 -arch=sm_37

SRC_DIR := src
BIN_DIR := bin
SCRIPTS_DIR := scripts

# Default block sizes for transpose benchmark
# 8x8, 16x16, 32x32 cover the analysis dimensions
BENCH_SIZES := 512 1024 2048
BENCH_BLOCKS := 8x8 16x16 32x32

.PHONY: all clean hello transpose run-hello run-transpose

all: hello transpose

hello: $(BIN_DIR)/hello_world

transpose: $(BIN_DIR)/transpose

$(BIN_DIR)/hello_world: $(SRC_DIR)/hello_world.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(BIN_DIR)/transpose: $(SRC_DIR)/transpose.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

run-hello: $(BIN_DIR)/hello_world
	$(BIN_DIR)/hello_world 2 4 4

run-transpose: $(BIN_DIR)/transpose
	$(BIN_DIR)/transpose 1024 3 32 32

clean:
	rm -rf $(BIN_DIR)

help:
	@echo "Lab9 CUDA Matrix Transpose"
	@echo "  make             - Build all"
	@echo "  make hello       - Build hello_world"
	@echo "  make transpose   - Build transpose"
	@echo "  make run-hello   - Run hello_world"
	@echo "  make run-transpose - Run transpose"
	@echo "  make clean       - Remove binaries"
	@echo ""
	@echo "For full benchmark: uv run python scripts/benchmark.py"
