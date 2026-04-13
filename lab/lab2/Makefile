CXX := mpicxx
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra
SRC_DIR := src
BIN_DIR := bin
TARGETS := $(BIN_DIR)/mpi_matmul_v1 $(BIN_DIR)/mpi_matmul_v2_collective

all: $(TARGETS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/mpi_matmul_v1: $(SRC_DIR)/mpi_matmul_v1.cpp $(SRC_DIR)/matmul_mpi.hpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC_DIR)/mpi_matmul_v1.cpp

$(BIN_DIR)/mpi_matmul_v2_collective: $(SRC_DIR)/mpi_matmul_v2_collective.cpp $(SRC_DIR)/matmul_mpi.hpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC_DIR)/mpi_matmul_v2_collective.cpp

clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean
