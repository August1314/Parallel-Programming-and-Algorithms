CXX := g++
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -pthread
OPENMP_FLAGS := -fopenmp
BIN_DIR := bin
SRC_DIR := src
TARGETS := \
	$(BIN_DIR)/heated_plate_openmp \
	$(BIN_DIR)/libparallel_for.so \
	$(BIN_DIR)/heated_plate_pthreads

all: $(TARGETS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/heated_plate_openmp: $(SRC_DIR)/heated_plate_openmp.cpp $(SRC_DIR)/heated_plate_common.hpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -o $@ $(SRC_DIR)/heated_plate_openmp.cpp

$(BIN_DIR)/libparallel_for.so: $(SRC_DIR)/parallel_for.cpp $(SRC_DIR)/parallel_for.hpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -shared -fPIC -o $@ $(SRC_DIR)/parallel_for.cpp

$(BIN_DIR)/heated_plate_pthreads: $(SRC_DIR)/heated_plate_pthreads.cpp $(SRC_DIR)/heated_plate_common.hpp $(SRC_DIR)/parallel_for.hpp $(BIN_DIR)/libparallel_for.so | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC_DIR)/heated_plate_pthreads.cpp -L$(BIN_DIR) -lparallel_for -Wl,-rpath,'$$ORIGIN'

clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean
