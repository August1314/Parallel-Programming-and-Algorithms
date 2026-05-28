CXX := g++
MPICXX := mpicxx
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra
BIN_DIR := bin
SRC_DIR := src

TARGETS := \
	$(BIN_DIR)/fft_serial \
	$(BIN_DIR)/fft_mpi

all: $(TARGETS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/fft_serial: $(SRC_DIR)/fft_serial.cpp $(SRC_DIR)/fft_common.hpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC_DIR)/fft_serial.cpp

$(BIN_DIR)/fft_mpi: $(SRC_DIR)/fft_mpi.cpp $(SRC_DIR)/fft_common.hpp | $(BIN_DIR)
	$(MPICXX) $(CXXFLAGS) -o $@ $(SRC_DIR)/fft_mpi.cpp

clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean
