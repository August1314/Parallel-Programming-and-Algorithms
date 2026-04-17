CXX := clang++
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -pthread
SRC_DIR := src
BIN_DIR := bin
TARGETS := $(BIN_DIR)/pthreads_matmul_v1 $(BIN_DIR)/pthreads_matmul_v2_partitioned $(BIN_DIR)/pthreads_array_sum_v1 $(BIN_DIR)/pthreads_array_sum_v2_reduction

all: $(TARGETS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/pthreads_matmul_v1: $(SRC_DIR)/pthreads_matmul.cpp $(SRC_DIR)/pthreads_common.hpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -DPTHREADS_MATMUL_VERSION=1 -o $@ $(SRC_DIR)/pthreads_matmul.cpp

$(BIN_DIR)/pthreads_matmul_v2_partitioned: $(SRC_DIR)/pthreads_matmul.cpp $(SRC_DIR)/pthreads_common.hpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -DPTHREADS_MATMUL_VERSION=2 -o $@ $(SRC_DIR)/pthreads_matmul.cpp

$(BIN_DIR)/pthreads_array_sum_v1: $(SRC_DIR)/pthreads_array_sum.cpp $(SRC_DIR)/pthreads_common.hpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -DPTHREADS_ARRAY_SUM_VERSION=1 -o $@ $(SRC_DIR)/pthreads_array_sum.cpp

$(BIN_DIR)/pthreads_array_sum_v2_reduction: $(SRC_DIR)/pthreads_array_sum.cpp $(SRC_DIR)/pthreads_common.hpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -DPTHREADS_ARRAY_SUM_VERSION=2 -o $@ $(SRC_DIR)/pthreads_array_sum.cpp

clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean
