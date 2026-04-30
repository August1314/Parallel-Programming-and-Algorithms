CC := clang
CFLAGS := -O2 -std=c11 -Wall -Wextra -pedantic -pthread
LDFLAGS := -lm
SRC_DIR := src
BIN_DIR := bin
TARGETS := $(BIN_DIR)/quadratic_solver_pthreads $(BIN_DIR)/monte_carlo_pi_pthreads

all: $(TARGETS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/quadratic_solver_pthreads: $(SRC_DIR)/quadratic_solver_pthreads.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BIN_DIR)/monte_carlo_pi_pthreads: $(SRC_DIR)/monte_carlo_pi_pthreads.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean
