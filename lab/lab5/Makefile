CXX := g++
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -fopenmp -pthread
BIN_DIR := bin
SRC_DIR := src
TARGETS := \
	$(BIN_DIR)/openmp_matmul_default \
	$(BIN_DIR)/openmp_matmul_static1 \
	$(BIN_DIR)/openmp_matmul_dynamic1 \
	$(BIN_DIR)/libparallel_for.so \
	$(BIN_DIR)/parallel_for_matmul

all: $(TARGETS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/openmp_matmul_default: $(SRC_DIR)/openmp_matmul.cpp $(SRC_DIR)/matmul_common.hpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -DOPENMP_SCHEDULE_KIND=0 -o $@ $(SRC_DIR)/openmp_matmul.cpp

$(BIN_DIR)/openmp_matmul_static1: $(SRC_DIR)/openmp_matmul.cpp $(SRC_DIR)/matmul_common.hpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -DOPENMP_SCHEDULE_KIND=1 -o $@ $(SRC_DIR)/openmp_matmul.cpp

$(BIN_DIR)/openmp_matmul_dynamic1: $(SRC_DIR)/openmp_matmul.cpp $(SRC_DIR)/matmul_common.hpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -DOPENMP_SCHEDULE_KIND=2 -o $@ $(SRC_DIR)/openmp_matmul.cpp

$(BIN_DIR)/libparallel_for.so: $(SRC_DIR)/parallel_for.cpp $(SRC_DIR)/parallel_for.hpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -shared -fPIC -o $@ $(SRC_DIR)/parallel_for.cpp

$(BIN_DIR)/parallel_for_matmul: $(SRC_DIR)/parallel_for_matmul.cpp $(SRC_DIR)/matmul_common.hpp $(SRC_DIR)/parallel_for.hpp $(BIN_DIR)/libparallel_for.so | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC_DIR)/parallel_for_matmul.cpp -L$(BIN_DIR) -lparallel_for -Wl,-rpath,'$$ORIGIN'

clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean
