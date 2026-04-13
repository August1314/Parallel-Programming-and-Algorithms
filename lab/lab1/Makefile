CXX = mpicxx
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra

SRC_DIR = src
BIN_DIR = bin

TARGET = $(BIN_DIR)/mpi_matmul_v1

all: $(BIN_DIR) $(TARGET)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(TARGET): $(SRC_DIR)/mpi_matmul_v1.cpp $(SRC_DIR)/matmul_mpi.hpp
	$(CXX) $(CXXFLAGS) -o $@ $(SRC_DIR)/mpi_matmul_v1.cpp

clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean
