CXX := g++-15
CXX_SERIAL := g++
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra
OMPFLAGS := -fopenmp
BIN_DIR := bin
SRC_DIR := src

TARGETS := \
	$(BIN_DIR)/apsp_serial \
	$(BIN_DIR)/apsp_openmp

all: $(TARGETS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/apsp_serial: $(SRC_DIR)/apsp_serial.cpp $(SRC_DIR)/graph.hpp | $(BIN_DIR)
	$(CXX_SERIAL) $(CXXFLAGS) -o $@ $(SRC_DIR)/apsp_serial.cpp

$(BIN_DIR)/apsp_openmp: $(SRC_DIR)/apsp_openmp.cpp $(SRC_DIR)/graph.hpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $(SRC_DIR)/apsp_openmp.cpp

clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean
