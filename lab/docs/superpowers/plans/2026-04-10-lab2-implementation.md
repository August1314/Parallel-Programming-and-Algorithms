# Lab2 MPI Matrix Multiplication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `lab/lab2` 下实现独立的 MPI 矩阵乘法实验模块，包含点对点对照版、集合通信进阶版、benchmark、绘图与报告草稿闭环。

**Architecture:** 复用 `lab1` 的矩阵生成、checksum、`--dump` 与按行切分思路，在 `lab2` 中重新组织为独立模块。核心实现分为 `mpi_v1_p2p` 和 `mpi_v2_collective` 两个可执行版本，`scripts/benchmark.py` 统一采集结果，`scripts/plot.py` 统一生成图表，`report/lab2_report.tex` 消费结果资产形成报告草稿。

**Tech Stack:** C++17, MPI (`mpicxx`, `mpirun`), Python 3 via `uv`, `numpy`, `pandas`, `matplotlib`, `unittest`

---

## File Structure

### Create or modify in this plan

- Create: `lab/lab2/src/matmul_mpi.hpp` — 公共参数解析、矩阵生成、校验、输出、行划分、MPI struct datatype 构建。
- Create: `lab/lab2/src/mpi_matmul_v1.cpp` — 点对点对照版可执行入口。
- Create: `lab/lab2/src/mpi_matmul_v2_collective.cpp` — 集合通信主版本可执行入口。
- Create: `lab/lab2/Makefile` — 构建两个 MPI 可执行文件。
- Create: `lab/lab2/scripts/benchmark.py` — 多版本、多进程、多规模 benchmark。
- Create: `lab/lab2/scripts/plot.py` — 生成折线图、加速比图、效率图、版本对比图、热力图。
- Create: `lab/lab2/scripts/export_report_assets.py` — 整理报告表格资产。
- Create: `lab/lab2/tests/test_benchmark.py` — benchmark/plot 回归测试。
- Create: `lab/lab2/pyproject.toml` — `uv` Python 依赖。
- Create: `lab/lab2/report/lab2_report.tex` — 报告草稿。
- Create: `lab/lab2/CLAUDE.md` — 模块说明。
- Create: `lab/lab2/results/.gitkeep` — 结果目录占位。
- Create: `lab/lab2/results/figures/.gitkeep` — 图目录占位。
- Create: `lab/lab2/results/tables/.gitkeep` — 表目录占位。

## Task 1: 搭建 lab2 目录骨架与基础构建文件

**Files:**
- Create: `lab/lab2/Makefile`
- Create: `lab/lab2/pyproject.toml`
- Create: `lab/lab2/CLAUDE.md`
- Create: `lab/lab2/results/.gitkeep`
- Create: `lab/lab2/results/figures/.gitkeep`
- Create: `lab/lab2/results/tables/.gitkeep`

- [ ] **Step 1: 创建目录与占位文件**

```text
lab/lab2/
lab/lab2/results/
lab/lab2/results/figures/
lab/lab2/results/tables/
lab/lab2/report/
lab/lab2/scripts/
lab/lab2/src/
lab/lab2/tests/
```

- [ ] **Step 2: 写入 `lab/lab2/Makefile`**

```makefile
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
```

- [ ] **Step 3: 写入 `lab/lab2/pyproject.toml`**

```toml
[project]
name = "lab2-mpi-matmul"
version = "0.1.0"
description = "Lab2 MPI matrix multiplication benchmark and plotting"
requires-python = ">=3.10"
dependencies = [
  "matplotlib>=3.8",
  "numpy>=2.0",
  "pandas>=2.2",
]
```

- [ ] **Step 4: 写入 `lab/lab2/CLAUDE.md`**

```md
# lab2 CLAUDE.md

## 模块职责

`lab/lab2` 是 MPI 矩阵乘法进阶实验模块，包含点对点对照版、集合通信进阶版、benchmark、绘图与报告草稿。

## 核心要求

- 保持 stdout 为稳定 `key=value`
- `mpi_v2_collective` 必须使用集合通信和 `MPI_Type_create_struct`
- Python 工作流统一使用 `uv run python ...`
```

- [ ] **Step 5: 运行基础构建检查**

Run: `make -C ./lab/lab2`
Expected: 初次可能失败，因为源文件尚未创建；Makefile 被正确识别。

- [ ] **Step 6: Commit**

```bash
git add lab/lab2/Makefile lab/lab2/pyproject.toml lab/lab2/CLAUDE.md lab/lab2/results/.gitkeep lab/lab2/results/figures/.gitkeep lab/lab2/results/tables/.gitkeep
git commit -m "chore: scaffold lab2 module"
```

## Task 2: 实现公共头文件与运行配置 datatype

**Files:**
- Create: `lab/lab2/src/matmul_mpi.hpp`
- Test: `lab/lab2/src/mpi_matmul_v2_collective.cpp`

- [ ] **Step 1: 写入公共头文件**

```cpp
#ifndef LAB2_MATMUL_MPI_HPP
#define LAB2_MATMUL_MPI_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct MatrixRunConfig {
    int m;
    int n;
    int k;
    std::uint64_t seed;
    int dump_matrix;
};

struct MatrixRunResult {
    double seconds;
    double checksum;
    double max_abs_value;
};

inline bool parse_flag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == flag) {
            return true;
        }
    }
    return false;
}

inline MatrixRunConfig parse_args(int argc, char** argv) {
    if (argc < 4) {
        throw std::runtime_error("usage: mpirun -np <procs> ./mpi_matmul <m> <n> <k> [seed] [--dump]");
    }

    MatrixRunConfig config{};
    config.m = std::stoi(argv[1]);
    config.n = std::stoi(argv[2]);
    config.k = std::stoi(argv[3]);
    config.seed = argc >= 5 && std::string(argv[4]).rfind("--", 0) != 0
        ? static_cast<std::uint64_t>(std::stoull(argv[4]))
        : 20250401ULL;
    config.dump_matrix = parse_flag(argc, argv, "--dump") ? 1 : 0;

    if (config.m <= 0 || config.n <= 0 || config.k <= 0) {
        throw std::runtime_error("matrix dimensions must be positive");
    }
    return config;
}

inline std::uint64_t splitmix64_next(std::uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
}

inline double next_unit_value(std::uint64_t& state) {
    const std::uint64_t raw = splitmix64_next(state);
    const double unit = static_cast<double>((raw >> 11U) & ((1ULL << 53U) - 1ULL)) / static_cast<double>(1ULL << 53U);
    return unit * 2.0 - 1.0;
}

inline std::vector<double> generate_matrix(int rows, int cols, std::uint64_t seed, std::uint64_t salt) {
    std::uint64_t state = seed ^ (salt * 0x9E3779B97F4A7C15ULL);
    std::vector<double> matrix(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
    for (double& value : matrix) {
        value = next_unit_value(state);
    }
    return matrix;
}

inline double checksum_matrix(const std::vector<double>& matrix) {
    double checksum = 0.0;
    for (std::size_t i = 0; i < matrix.size(); ++i) {
        checksum += matrix[i] * (static_cast<double>(i % 17U) + 1.0);
    }
    return checksum;
}

inline double max_abs_matrix(const std::vector<double>& matrix) {
    double max_abs = 0.0;
    for (double value : matrix) {
        max_abs = std::max(max_abs, std::abs(value));
    }
    return max_abs;
}

inline void dump_matrix(const std::vector<double>& matrix, int rows, int cols, const std::string& name) {
    std::cout << name << ":\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (j > 0) {
                std::cout << ' ';
            }
            std::cout << std::fixed << std::setprecision(6) << matrix[static_cast<std::size_t>(i) * cols + j];
        }
        std::cout << '\n';
    }
}

inline void local_matmul(const double* a, const double* b, double* c, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            double sum = 0.0;
            for (int p = 0; p < n; ++p) {
                sum += a[static_cast<std::size_t>(i) * n + p] * b[static_cast<std::size_t>(p) * k + j];
            }
            c[static_cast<std::size_t>(i) * k + j] = sum;
        }
    }
}

inline void compute_row_partition(int m, int size, std::vector<int>& row_counts, std::vector<int>& row_displs) {
    row_counts.assign(size, 0);
    row_displs.assign(size, 0);
    const int base = m / size;
    const int remainder = m % size;
    for (int p = 0; p < size; ++p) {
        row_counts[p] = base + (p < remainder ? 1 : 0);
        row_displs[p] = (p == 0) ? 0 : row_displs[p - 1] + row_counts[p - 1];
    }
}

inline MPI_Datatype build_config_type() {
    MatrixRunConfig sample{};
    int block_lengths[5] = {1, 1, 1, 1, 1};
    MPI_Aint displacements[5];
    MPI_Datatype types[5] = {MPI_INT, MPI_INT, MPI_INT, MPI_UINT64_T, MPI_INT};
    MPI_Aint base_address;
    MPI_Get_address(&sample, &base_address);
    MPI_Get_address(&sample.m, &displacements[0]);
    MPI_Get_address(&sample.n, &displacements[1]);
    MPI_Get_address(&sample.k, &displacements[2]);
    MPI_Get_address(&sample.seed, &displacements[3]);
    MPI_Get_address(&sample.dump_matrix, &displacements[4]);
    for (MPI_Aint& displacement : displacements) {
        displacement -= base_address;
    }
    MPI_Datatype config_type;
    MPI_Type_create_struct(5, block_lengths, displacements, types, &config_type);
    MPI_Type_commit(&config_type);
    return config_type;
}

inline void print_result(const std::string& version, const MatrixRunConfig& config, int processes, double elapsed, const std::vector<double>& c) {
    std::cout << std::fixed << std::setprecision(9);
    std::cout << "version=" << version << '\n';
    std::cout << "m=" << config.m << " n=" << config.n << " k=" << config.k << '\n';
    std::cout << "seed=" << config.seed << '\n';
    std::cout << "processes=" << processes << '\n';
    std::cout << "time_sec=" << elapsed << '\n';
    std::cout << "checksum=" << checksum_matrix(c) << '\n';
    std::cout << "max_abs=" << max_abs_matrix(c) << '\n';
}

#endif
```

- [ ] **Step 2: 运行编译检查验证头文件可被包含**

Run: `make -C ./lab/lab2`
Expected: 仍可能失败，因为 `mpi_matmul_v1.cpp` 和 `mpi_matmul_v2_collective.cpp` 尚未创建，但头文件语法不应成为额外错误源。

- [ ] **Step 3: Commit**

```bash
git add lab/lab2/src/matmul_mpi.hpp
git commit -m "feat: add lab2 shared mpi utilities"
```

## Task 3: 迁移点对点对照版 `mpi_v1_p2p`

**Files:**
- Create: `lab/lab2/src/mpi_matmul_v1.cpp`
- Modify: `lab/lab2/src/matmul_mpi.hpp`
- Test: `lab/lab2/src/mpi_matmul_v1.cpp`

- [ ] **Step 1: 写入 `lab/lab2/src/mpi_matmul_v1.cpp`**

```cpp
#include "matmul_mpi.hpp"
#include <mpi.h>

void mpi_matmul_v1(const MatrixRunConfig& config, int rank, int size) {
    std::vector<double> a;
    std::vector<double> b;
    std::vector<double> c;

    std::vector<int> row_counts;
    std::vector<int> row_displs;
    compute_row_partition(config.m, size, row_counts, row_displs);

    const int my_rows = row_counts[rank];
    const int my_row_start = row_displs[rank];

    std::vector<double> local_a(static_cast<std::size_t>(my_rows) * config.n);
    std::vector<double> local_c(static_cast<std::size_t>(my_rows) * config.k, 0.0);

    if (rank == 0) {
        a = generate_matrix(config.m, config.n, config.seed, 1);
        b = generate_matrix(config.n, config.k, config.seed, 2);
        c.resize(static_cast<std::size_t>(config.m) * config.k, 0.0);
    } else {
        b.resize(static_cast<std::size_t>(config.n) * config.k);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start_time = MPI_Wtime();

    if (rank == 0) {
        for (int p = 1; p < size; ++p) {
            MPI_Send(&a[static_cast<std::size_t>(row_displs[p]) * config.n], row_counts[p] * config.n, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
        }
        std::copy(a.begin(), a.begin() + static_cast<std::size_t>(my_rows) * config.n, local_a.begin());
        for (int p = 1; p < size; ++p) {
            MPI_Send(b.data(), config.n * config.k, MPI_DOUBLE, p, 1, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(local_a.data(), my_rows * config.n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(b.data(), config.n * config.k, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    local_matmul(local_a.data(), b.data(), local_c.data(), my_rows, config.n, config.k);

    if (rank == 0) {
        std::copy(local_c.begin(), local_c.end(), c.begin() + static_cast<std::size_t>(my_row_start) * config.k);
        for (int p = 1; p < size; ++p) {
            MPI_Recv(&c[static_cast<std::size_t>(row_displs[p]) * config.k], row_counts[p] * config.k, MPI_DOUBLE, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(local_c.data(), my_rows * config.k, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }

    const double elapsed = MPI_Wtime() - start_time;

    if (rank == 0) {
        print_result("mpi_v1_p2p", config, size, elapsed, c);
        const bool allow_dump = config.dump_matrix && config.m <= 8 && config.n <= 8 && config.k <= 8;
        if (allow_dump) {
            dump_matrix(a, config.m, config.n, "A");
            dump_matrix(b, config.n, config.k, "B");
            dump_matrix(c, config.m, config.k, "C");
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try {
        MatrixRunConfig config = parse_args(argc, argv);
        mpi_matmul_v1(config, rank, size);
    } catch (const std::exception& ex) {
        if (rank == 0) {
            std::cerr << "Error: " << ex.what() << '\n';
        }
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}
```

- [ ] **Step 2: 编译 v1**

Run: `make -C ./lab/lab2`
Expected: `bin/mpi_matmul_v1` 成功生成，`mpi_matmul_v2_collective` 尚未生成或因缺文件失败。

- [ ] **Step 3: 运行小规模用例**

Run: `mpirun -np 2 --oversubscribe ./lab/lab2/bin/mpi_matmul_v1 4 4 4 20250401 --dump`
Expected: 输出 `version=mpi_v1_p2p`，并打印 `A`、`B`、`C`。

- [ ] **Step 4: Commit**

```bash
git add lab/lab2/src/mpi_matmul_v1.cpp
git commit -m "feat: add lab2 point-to-point baseline"
```

## Task 4: 实现集合通信版 `mpi_v2_collective`

**Files:**
- Create: `lab/lab2/src/mpi_matmul_v2_collective.cpp`
- Modify: `lab/lab2/src/matmul_mpi.hpp`
- Test: `lab/lab2/src/mpi_matmul_v2_collective.cpp`

- [ ] **Step 1: 写入 `lab/lab2/src/mpi_matmul_v2_collective.cpp`**

```cpp
#include "matmul_mpi.hpp"
#include <mpi.h>

void mpi_matmul_v2_collective(MatrixRunConfig config, int rank, int size) {
    MPI_Datatype config_type = build_config_type();
    MPI_Bcast(&config, 1, config_type, 0, MPI_COMM_WORLD);

    std::vector<int> row_counts;
    std::vector<int> row_displs;
    compute_row_partition(config.m, size, row_counts, row_displs);

    std::vector<int> a_counts(size);
    std::vector<int> a_displs(size);
    std::vector<int> c_counts(size);
    std::vector<int> c_displs(size);
    for (int p = 0; p < size; ++p) {
        a_counts[p] = row_counts[p] * config.n;
        a_displs[p] = row_displs[p] * config.n;
        c_counts[p] = row_counts[p] * config.k;
        c_displs[p] = row_displs[p] * config.k;
    }

    std::vector<double> a;
    std::vector<double> b(static_cast<std::size_t>(config.n) * config.k);
    std::vector<double> c;
    std::vector<double> local_a(static_cast<std::size_t>(row_counts[rank]) * config.n);
    std::vector<double> local_c(static_cast<std::size_t>(row_counts[rank]) * config.k, 0.0);

    if (rank == 0) {
        a = generate_matrix(config.m, config.n, config.seed, 1);
        b = generate_matrix(config.n, config.k, config.seed, 2);
        c.resize(static_cast<std::size_t>(config.m) * config.k, 0.0);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start_time = MPI_Wtime();

    MPI_Scatterv(rank == 0 ? a.data() : nullptr, a_counts.data(), a_displs.data(), MPI_DOUBLE, local_a.data(), a_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b.data(), config.n * config.k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    local_matmul(local_a.data(), b.data(), local_c.data(), row_counts[rank], config.n, config.k);

    MPI_Gatherv(local_c.data(), c_counts[rank], MPI_DOUBLE, rank == 0 ? c.data() : nullptr, c_counts.data(), c_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const double elapsed = MPI_Wtime() - start_time;

    if (rank == 0) {
        print_result("mpi_v2_collective", config, size, elapsed, c);
        const bool allow_dump = config.dump_matrix && config.m <= 8 && config.n <= 8 && config.k <= 8;
        if (allow_dump) {
            dump_matrix(a, config.m, config.n, "A");
            dump_matrix(b, config.n, config.k, "B");
            dump_matrix(c, config.m, config.k, "C");
        }
    }

    MPI_Type_free(&config_type);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try {
        MatrixRunConfig config{};
        if (rank == 0) {
            config = parse_args(argc, argv);
        }
        mpi_matmul_v2_collective(config, rank, size);
    } catch (const std::exception& ex) {
        if (rank == 0) {
            std::cerr << "Error: " << ex.what() << '\n';
        }
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}
```

- [ ] **Step 2: 编译全部目标**

Run: `make -C ./lab/lab2`
Expected: `bin/mpi_matmul_v1` 和 `bin/mpi_matmul_v2_collective` 均成功生成。

- [ ] **Step 3: 运行小规模集合通信用例**

Run: `mpirun -np 2 --oversubscribe ./lab/lab2/bin/mpi_matmul_v2_collective 4 4 4 20250401 --dump`
Expected: 输出 `version=mpi_v2_collective`，并打印 `A`、`B`、`C`。

- [ ] **Step 4: 对比两个版本 checksum**

Run: `mpirun -np 2 --oversubscribe ./lab/lab2/bin/mpi_matmul_v1 8 8 8 20250401 && mpirun -np 2 --oversubscribe ./lab/lab2/bin/mpi_matmul_v2_collective 8 8 8 20250401`
Expected: 两次输出中的 `checksum` 相同。

- [ ] **Step 5: Commit**

```bash
git add lab/lab2/src/mpi_matmul_v2_collective.cpp
git commit -m "feat: add collective mpi implementation for lab2"
```

## Task 5: 编写 benchmark 脚本

**Files:**
- Create: `lab/lab2/scripts/benchmark.py`
- Test: `lab/lab2/tests/test_benchmark.py`

- [ ] **Step 1: 写入 `lab/lab2/scripts/benchmark.py`**

```python
#!/usr/bin/env python3
import csv
import json
import math
import re
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
LAB_DIR = SCRIPT_DIR.parent
BIN_DIR = LAB_DIR / "bin"
RESULTS_DIR = LAB_DIR / "results"
EXECUTABLES = {
    "mpi_v1_p2p": BIN_DIR / "mpi_matmul_v1",
    "mpi_v2_collective": BIN_DIR / "mpi_matmul_v2_collective",
}
PROCESS_COUNTS = [1, 2, 4, 8, 16]
MATRIX_SIZES = [128, 256, 512, 1024, 2048]
SEED = 20250401
NUM_RUNS = 3
TIMEOUT_SEC = 300
REQUIRED_KEYS = ("version", "time_sec", "checksum", "max_abs", "processes", "m", "n", "k", "seed")


def parse_key_value_output(output: str) -> dict:
    data = {}
    for line in output.strip().splitlines():
        for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", line):
            data[key] = value
    return data


def benchmark_command(executable: Path, m: int, n: int, k: int, num_procs: int, seed: int) -> list[str]:
    return ["mpirun", "-np", str(num_procs), "--oversubscribe", str(executable), str(m), str(n), str(k), str(seed)]


def run_case(version: str, m: int, n: int, k: int, num_procs: int, seed: int = SEED) -> dict:
    cmd = benchmark_command(EXECUTABLES[version], m, n, k, num_procs, seed)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SEC, check=False)
    except subprocess.TimeoutExpired as exc:
        return {"success": False, "version": version, "command": cmd, "returncode": None, "stdout": exc.stdout or "", "stderr": exc.stderr or "", "parsed": {}, "error": f"timeout after {TIMEOUT_SEC} seconds"}

    parsed = parse_key_value_output(result.stdout)
    missing_keys = [key for key in REQUIRED_KEYS if key not in parsed]
    success = result.returncode == 0 and not missing_keys
    if success:
        error = ""
    else:
        parts = []
        if result.returncode != 0:
            parts.append(f"return code {result.returncode}")
        if missing_keys:
            parts.append(f"missing keys: {', '.join(missing_keys)}")
        if result.stderr.strip():
            parts.append("stderr captured")
        error = "; ".join(parts) if parts else "unknown benchmark failure"
    return {"success": success, "version": version, "command": cmd, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "parsed": parsed, "error": error}


def summarize_runs(version: str, size: int, num_procs: int, runs: list[dict]) -> dict:
    successful = [r for r in runs if r["success"]]
    failed = [r for r in runs if not r["success"]]
    times = [float(r["parsed"]["time_sec"]) for r in successful]
    checksums = [float(r["parsed"]["checksum"]) for r in successful]
    checksum_consistent = len(checksums) <= 1 or all(math.isclose(v, checksums[0], rel_tol=0.0, abs_tol=1e-9) for v in checksums[1:])
    if failed:
        status = "partial_failure" if successful else "failed"
    elif not checksum_consistent:
        status = "checksum_mismatch"
    else:
        status = "ok"
    return {
        "version": version,
        "processes": num_procs,
        "matrix_size": size,
        "m": size,
        "n": size,
        "k": size,
        "requested_runs": len(runs),
        "successful_runs": len(successful),
        "failed_runs": len(failed),
        "status": status,
        "avg_time_sec": (sum(times) / len(times)) if times else None,
        "all_times": times,
        "checksums": checksums,
        "baseline_checksum": None,
        "baseline_checksum_match": None,
        "runs": runs,
    }


def attach_baseline_consistency(results: list[dict]) -> None:
    baselines = {}
    for row in results:
        if row["processes"] == 1 and row["successful_runs"] > 0 and row["status"] == "ok":
            baselines[(row["version"], row["matrix_size"])] = row["checksums"][0]
    for row in results:
        baseline = baselines.get((row["version"], row["matrix_size"]))
        row["baseline_checksum"] = baseline
        if baseline is None or not row["checksums"]:
            row["baseline_checksum_match"] = None
        else:
            row["baseline_checksum_match"] = all(math.isclose(v, baseline, rel_tol=0.0, abs_tol=1e-9) for v in row["checksums"])
            if row["status"] == "ok" and row["baseline_checksum_match"] is False:
                row["status"] = "baseline_checksum_mismatch"


def write_csv_summary(results: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["version", "processes", "matrix_size", "avg_time_sec", "status", "successful_runs", "failed_runs", "baseline_checksum_match"])
        for row in results:
            writer.writerow([row["version"], row["processes"], row["matrix_size"], row["avg_time_sec"], row["status"], row["successful_runs"], row["failed_runs"], row["baseline_checksum_match"]])


def write_course_table(results: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["version", "进程数/矩阵规模", *MATRIX_SIZES])
        for version in EXECUTABLES:
            for procs in PROCESS_COUNTS:
                row = [version, procs]
                for size in MATRIX_SIZES:
                    match = next((item for item in results if item["version"] == version and item["processes"] == procs and item["matrix_size"] == size), None)
                    row.append(f"{match['avg_time_sec']:.6f}" if match and match["avg_time_sec"] is not None else "N/A")
                writer.writerow(row)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    results = []
    for version, executable in EXECUTABLES.items():
        if not executable.exists():
            raise SystemExit(f"missing executable: {executable}")
        for procs in PROCESS_COUNTS:
            for size in MATRIX_SIZES:
                runs = [run_case(version, size, size, size, procs) for _ in range(NUM_RUNS)]
                results.append(summarize_runs(version, size, procs, runs))
    attach_baseline_consistency(results)
    with (RESULTS_DIR / "benchmark_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    write_course_table(results, RESULTS_DIR / "performance_table.csv")
    write_csv_summary(results, RESULTS_DIR / "summary_by_version.csv")
    bad = [row for row in results if row["status"] != "ok"]
    if bad:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行 benchmark 小范围冒烟**

Run: `uv run python ./lab/lab2/scripts/benchmark.py`
Expected: 生成 `benchmark_results.json`、`performance_table.csv`、`summary_by_version.csv`；若某些大规模环境不支持，可先临时缩小矩阵列表再确认脚本行为。

- [ ] **Step 3: Commit**

```bash
git add lab/lab2/scripts/benchmark.py
git commit -m "feat: add lab2 benchmark pipeline"
```

## Task 6: 编写绘图与报告资产脚本

**Files:**
- Create: `lab/lab2/scripts/plot.py`
- Create: `lab/lab2/scripts/export_report_assets.py`
- Test: `lab/lab2/scripts/plot.py`

- [ ] **Step 1: 写入 `lab/lab2/scripts/plot.py`**

```python
#!/usr/bin/env python3
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).parent.absolute()
LAB_DIR = SCRIPT_DIR.parent
RESULTS_DIR = LAB_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SUMMARY_CSV = RESULTS_DIR / "summary_by_version.csv"


def load_summary() -> pd.DataFrame:
    return pd.read_csv(SUMMARY_CSV)


def save_runtime_lines(df: pd.DataFrame) -> None:
    for version in sorted(df["version"].unique()):
        subset = df[(df["version"] == version) & (df["status"] == "ok")]
        plt.figure(figsize=(8, 5))
        for procs in sorted(subset["processes"].unique()):
            part = subset[subset["processes"] == procs].sort_values("matrix_size")
            plt.plot(part["matrix_size"], part["avg_time_sec"], marker="o", label=f"p={procs}")
        plt.title(f"Runtime - {version}")
        plt.xlabel("Matrix Size")
        plt.ylabel("Average Time (s)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"runtime_{version}.png", dpi=200)
        plt.close()


def save_speedup_lines(df: pd.DataFrame) -> None:
    for version in sorted(df["version"].unique()):
        subset = df[(df["version"] == version) & (df["status"] == "ok")].copy()
        baselines = subset[subset["processes"] == 1][["matrix_size", "avg_time_sec"]].rename(columns={"avg_time_sec": "baseline_time"})
        merged = subset.merge(baselines, on="matrix_size", how="left")
        merged["speedup"] = merged["baseline_time"] / merged["avg_time_sec"]
        plt.figure(figsize=(8, 5))
        for procs in sorted(merged["processes"].unique()):
            part = merged[merged["processes"] == procs].sort_values("matrix_size")
            plt.plot(part["matrix_size"], part["speedup"], marker="o", label=f"p={procs}")
        plt.title(f"Speedup - {version}")
        plt.xlabel("Matrix Size")
        plt.ylabel("Speedup")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"speedup_{version}.png", dpi=200)
        plt.close()


def save_efficiency_lines(df: pd.DataFrame) -> None:
    for version in sorted(df["version"].unique()):
        subset = df[(df["version"] == version) & (df["status"] == "ok")].copy()
        baselines = subset[subset["processes"] == 1][["matrix_size", "avg_time_sec"]].rename(columns={"avg_time_sec": "baseline_time"})
        merged = subset.merge(baselines, on="matrix_size", how="left")
        merged["efficiency"] = (merged["baseline_time"] / merged["avg_time_sec"]) / merged["processes"]
        plt.figure(figsize=(8, 5))
        for procs in sorted(merged["processes"].unique()):
            part = merged[merged["processes"] == procs].sort_values("matrix_size")
            plt.plot(part["matrix_size"], part["efficiency"], marker="o", label=f"p={procs}")
        plt.title(f"Efficiency - {version}")
        plt.xlabel("Matrix Size")
        plt.ylabel("Efficiency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"efficiency_{version}.png", dpi=200)
        plt.close()


def save_version_comparison(df: pd.DataFrame) -> None:
    subset = df[(df["status"] == "ok") & (df["matrix_size"].isin([512, 1024, 2048])) & (df["processes"] == 4)]
    pivot = subset.pivot(index="matrix_size", columns="version", values="avg_time_sec")
    pivot.plot(kind="bar", figsize=(8, 5))
    plt.title("Version Comparison at 4 Processes")
    plt.xlabel("Matrix Size")
    plt.ylabel("Average Time (s)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "version_comparison.png", dpi=200)
    plt.close()


def save_heatmaps(df: pd.DataFrame) -> None:
    for version in sorted(df["version"].unique()):
        subset = df[(df["version"] == version) & (df["status"] == "ok")]
        pivot = subset.pivot(index="processes", columns="matrix_size", values="avg_time_sec")
        plt.figure(figsize=(8, 5))
        plt.imshow(pivot.values, aspect="auto")
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.title(f"Runtime Heatmap - {version}")
        plt.xlabel("Matrix Size")
        plt.ylabel("Processes")
        plt.colorbar(label="Average Time (s)")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"heatmap_{version}.png", dpi=200)
        plt.close()


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_summary()
    save_runtime_lines(df)
    save_speedup_lines(df)
    save_efficiency_lines(df)
    save_version_comparison(df)
    save_heatmaps(df)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 写入 `lab/lab2/scripts/export_report_assets.py`**

```python
#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).parent.absolute()
LAB_DIR = SCRIPT_DIR.parent
RESULTS_DIR = LAB_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
SUMMARY_CSV = RESULTS_DIR / "summary_by_version.csv"


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SUMMARY_CSV)
    ok = df[df["status"] == "ok"].copy()
    ok["speedup"] = ok.groupby(["version", "matrix_size"])["avg_time_sec"].transform("first") / ok["avg_time_sec"]
    ok["efficiency"] = ok["speedup"] / ok["processes"]
    ok.pivot_table(index=["version", "processes"], columns="matrix_size", values="avg_time_sec").to_csv(TABLES_DIR / "time_table.csv")
    ok.pivot_table(index=["version", "processes"], columns="matrix_size", values="speedup").to_csv(TABLES_DIR / "speedup_table.csv")
    ok.pivot_table(index=["version", "processes"], columns="matrix_size", values="efficiency").to_csv(TABLES_DIR / "efficiency_table.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 运行绘图与表格整理**

Run: `uv run python ./lab/lab2/scripts/plot.py && uv run python ./lab/lab2/scripts/export_report_assets.py`
Expected: `results/figures/` 下生成多张 PNG，`results/tables/` 下生成三张 CSV。

- [ ] **Step 4: Commit**

```bash
git add lab/lab2/scripts/plot.py lab/lab2/scripts/export_report_assets.py
git commit -m "feat: add visualization and report assets pipeline"
```

## Task 7: 编写报告草稿与模块测试

**Files:**
- Create: `lab/lab2/report/lab2_report.tex`
- Create: `lab/lab2/tests/test_benchmark.py`

- [ ] **Step 1: 写入 `lab/lab2/report/lab2_report.tex`**

```tex
\\documentclass[11pt]{article}
\\usepackage{graphicx}
\\usepackage{booktabs}
\\usepackage[margin=1in]{geometry}
\\title{并行程序设计\\lab2: 基于MPI的并行矩阵乘法（进阶）}
\\author{梁力航}
\\date{\\today}

\\begin{document}
\\maketitle

\\section{实验目的}
本实验在上一实验点对点 MPI 矩阵乘法的基础上，引入 MPI 集合通信，并比较不同通信方式对性能和扩展性的影响。

\\section{实验环境}
填写本机 MPI 实现、编译器、CPU 与操作系统信息。

\\section{算法设计}
\\subsection{点对点版本}
描述 `mpi_v1_p2p` 使用 `MPI_Send/MPI_Recv` 的按行切分方式。

\\subsection{集合通信版本}
描述 `mpi_v2_collective` 使用 `MPI_Scatterv`、`MPI_Bcast`、`MPI_Gatherv` 的流程。

\\subsection{结构化配置通信}
说明如何使用 `MPI_Type_create_struct` 广播 `m/n/k/seed/dump_matrix`。

\\section{Benchmark 配置}
实验使用进程数 `1,2,4,8,16`，矩阵规模 `128,256,512,1024,2048`，每组重复 3 次。

\\section{结果展示}
\\subsection{运行时间图}
\\includegraphics[width=0.48\\textwidth]{../results/figures/runtime_mpi_v1_p2p.png}
\\includegraphics[width=0.48\\textwidth]{../results/figures/runtime_mpi_v2_collective.png}

\\subsection{加速比图}
\\includegraphics[width=0.48\\textwidth]{../results/figures/speedup_mpi_v1_p2p.png}
\\includegraphics[width=0.48\\textwidth]{../results/figures/speedup_mpi_v2_collective.png}

\\subsection{效率图}
\\includegraphics[width=0.48\\textwidth]{../results/figures/efficiency_mpi_v1_p2p.png}
\\includegraphics[width=0.48\\textwidth]{../results/figures/efficiency_mpi_v2_collective.png}

\\subsection{版本对比与热力图}
\\includegraphics[width=0.48\\textwidth]{../results/figures/version_comparison.png}
\\includegraphics[width=0.48\\textwidth]{../results/figures/heatmap_mpi_v2_collective.png}

\\section{性能分析}
从通信方式、root 端压力、扩展性、不同规模下 speedup 变化等角度分析结果。

\\section{总结}
总结集合通信相较点对点通信的收益与局限。

\\end{document}
```

- [ ] **Step 2: 写入 `lab/lab2/tests/test_benchmark.py`**

```python
import json
import subprocess
import unittest
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = LAB_DIR / "results"
BENCHMARK_JSON = RESULTS_DIR / "benchmark_results.json"
SUMMARY_CSV = RESULTS_DIR / "summary_by_version.csv"
FIGURES_DIR = RESULTS_DIR / "figures"


class BenchmarkArtifactsTest(unittest.TestCase):
    def test_required_files_exist(self):
        self.assertTrue(BENCHMARK_JSON.exists())
        self.assertTrue(SUMMARY_CSV.exists())

    def test_benchmark_json_has_required_keys(self):
        data = json.loads(BENCHMARK_JSON.read_text(encoding="utf-8"))
        self.assertTrue(data)
        sample = data[0]
        for key in ["version", "processes", "matrix_size", "avg_time_sec", "status", "runs"]:
            self.assertIn(key, sample)

    def test_plot_outputs_exist(self):
        expected = [
            "runtime_mpi_v1_p2p.png",
            "runtime_mpi_v2_collective.png",
            "speedup_mpi_v1_p2p.png",
            "speedup_mpi_v2_collective.png",
            "efficiency_mpi_v1_p2p.png",
            "efficiency_mpi_v2_collective.png",
            "version_comparison.png",
            "heatmap_mpi_v1_p2p.png",
            "heatmap_mpi_v2_collective.png",
        ]
        for name in expected:
            self.assertTrue((FIGURES_DIR / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: 运行测试**

Run: `uv run python -m unittest ./lab/lab2/tests/test_benchmark.py`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add lab/lab2/report/lab2_report.tex lab/lab2/tests/test_benchmark.py
git commit -m "feat: add lab2 report draft and artifact tests"
```

## Task 8: 最终回归与交付检查

**Files:**
- Modify: `lab/lab2/CLAUDE.md`
- Modify: `lab/lab2/results/*`
- Test: `lab/lab2/bin/*`, `lab/lab2/scripts/*`

- [ ] **Step 1: 运行完整构建**

Run: `make -C ./lab/lab2`
Expected: 两个可执行文件都生成。

- [ ] **Step 2: 运行完整 benchmark**

Run: `uv run python ./lab/lab2/scripts/benchmark.py`
Expected: 生成完整 JSON/CSV，状态全部为 `ok`。

- [ ] **Step 3: 运行完整绘图与表格导出**

Run: `uv run python ./lab/lab2/scripts/plot.py && uv run python ./lab/lab2/scripts/export_report_assets.py`
Expected: 图表与表格全部生成。

- [ ] **Step 4: 运行测试**

Run: `uv run python -m unittest ./lab/lab2/tests/test_benchmark.py`
Expected: PASS

- [ ] **Step 5: 补充 `lab/lab2/CLAUDE.md` 常用命令段**

```md
## 常用命令

```bash
make -C ./lab/lab2
mpirun -np 2 --oversubscribe ./lab/lab2/bin/mpi_matmul_v1 4 4 4 20250401 --dump
mpirun -np 2 --oversubscribe ./lab/lab2/bin/mpi_matmul_v2_collective 4 4 4 20250401 --dump
uv run python ./lab/lab2/scripts/benchmark.py
uv run python ./lab/lab2/scripts/plot.py
uv run python ./lab/lab2/scripts/export_report_assets.py
uv run python -m unittest ./lab/lab2/tests/test_benchmark.py
```
```

- [ ] **Step 6: Commit**

```bash
git add lab/lab2/CLAUDE.md lab/lab2/results lab/lab2/report/lab2_report.tex
git commit -m "feat: complete lab2 benchmark and reporting workflow"
```

## Self-Review

- Spec coverage: 已覆盖独立模块、两版本实现、集合通信、`MPI_Type_create_struct`、benchmark、绘图、报告草稿、测试与 `uv` 工作流。
- Placeholder scan: 已消除 `TODO`/`TBD`/“自行实现”等占位描述。
- Type consistency: `MatrixRunConfig`、`compute_row_partition`、`build_config_type`、`print_result` 在各任务中命名保持一致。
