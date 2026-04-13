[根目录](../../CLAUDE.md) > [lab](../) > **lab0**

# lab0 CLAUDE.md

## 模块职责

`lab/lab0` 是串行矩阵乘法实验模块，围绕同一输入/输出协议逐步比较不同实现路径：

- `v1_python`：Python 三重循环基线。
- `v2_cpp_baseline`：C++ 直接移植版。
- `v3_cpp_loop_reorder`：通过 `i-p-j` 循环顺序改善缓存访问。
- `v4_cpp_optimized`：在 `v3` 基础上依赖编译器 `-O3`。
- `v5_cpp_unrolled`：在 `v4` 上增加小幅循环展开。
- `v6_mkl`：调用 `cblas_dgemm`，优先用于 MKL 接口验证。

它还承担统一运行入口、统一 benchmark、兼容旧脚本名、Docker oneMKL 路由和提交收尾测试的职责。

## 入口与启动

### 主入口

- 统一路由入口：`lab/lab0/scripts/run_matrix_mul_unified`
- 兼容运行入口：`lab/lab0/scripts/run_matrix_mul`
- 构建入口：`lab/lab0/scripts/build_lab1.sh`
- 统一 benchmark：`lab/lab0/scripts/benchmark_lab1_unified.py`
- 兼容 benchmark 入口：`lab/lab0/scripts/benchmark_lab1.py`

### 代码入口

- 公共运行框架：`lab/lab0/src/matmul_common.hpp`
- Python 基线：`lab/lab0/src/v1_python.py`
- C++ 各版本：`lab/lab0/src/v2_cpp_baseline.cpp` 到 `lab/lab0/src/v6_mkl.cpp`

### 启动流程

1. 执行 `run_matrix_mul_unified <version> <m> <n> <k> [seed] [--dump]`。
2. 路由脚本按版本决定：
   - `v1_python` -> `scripts/v1_python_embedded.py`
   - `v2-v5` -> 若二进制不存在则自动执行 `build_lab1.sh`，随后运行 `bin/<version>`
   - `v6_mkl` -> 默认执行 `scripts/run_v6_docker.sh`；若 `LAB1_V6_MODE=local`，则改走本机 `bin/v6_mkl`
3. 实现程序输出统一字段：`version`、`m/n/k`、`seed`、`time_sec`、`checksum`、`max_abs`。
4. benchmark 脚本按 stdout 键值对收集并生成 JSON/CSV。

### 常用命令

```bash
# 构建本机版本
./lab/lab0/scripts/build_lab1.sh

# 跑 Python 基线
./lab/lab0/scripts/run_matrix_mul_unified v1_python 4 4 4 20250401 --dump

# 跑某个 C++ 版本
./lab/lab0/scripts/run_matrix_mul_unified v4_cpp_optimized 512 512 512 20250401

# 兼容旧入口
./lab/lab0/scripts/run_matrix_mul v2_cpp_baseline 512 512 512 20250401

# 默认 Docker 方式运行 v6
./lab/lab0/scripts/run_matrix_mul_unified v6_mkl 512 512 512 20250401

# 本机 MKL 方式运行 v6
LAB1_V6_MODE=local ./lab/lab0/scripts/run_matrix_mul_unified v6_mkl 512 512 512 20250401

# 跑 benchmark
python3 ./lab/lab0/scripts/benchmark_lab1_unified.py --sizes 512x512x512 1024x1024x1024 --repeat 3 --include-mkl

# 单独运行提交测试
python3 -m unittest ./lab/lab0/tests/test_submission_finalize.py
```

## 对外接口

### 命令行接口

#### `run_matrix_mul_unified`

参数格式：

```bash
run_matrix_mul_unified <version> <m> <n> <k> [seed] [--dump]
```

支持版本：

- `v1_python`
- `v2_cpp_baseline`
- `v3_cpp_loop_reorder`
- `v4_cpp_optimized`
- `v5_cpp_unrolled`
- `v6_mkl`

#### `benchmark_lab1_unified.py`

关键参数：

- `--sizes 512x512x512 1024x1024x1024`
- `--repeat 3`
- `--seed 20250401`
- `--peak-gflops <float>`
- `--include-mkl`
- `--v6-mode {docker,local}`
- `--output results/benchmark_unified_latest.json`

### 输出契约

程序和 benchmark 都依赖稳定 stdout 键值对：

- `version`
- `m`, `n`, `k`
- `seed`
- `time_sec`
- `checksum`
- `max_abs`

这套契约是测试脚本与 benchmark 的共同基础，修改输出格式前必须同步检查调用方。

## 关键依赖与配置

### 构建依赖

- Python 3
- `clang++` 或通过环境变量覆盖的 `CXX`
- C++17
- 可选 `pkg-config` + Intel MKL
- Docker（运行 `v6_mkl` 默认路径时必需）

### 构建参数与环境

`build_lab1.sh` 使用：

- 通用编译参数：`-std=c++17 -Wall -Wextra -pedantic`
- `v2/v3`：`-O0`
- `v4/v5/v6`：`-O3`

`v6_mkl.cpp` 会先尝试 `#include <mkl_cblas.h>`，再尝试 `<cblas.h>`；若缺头文件，会在运行时报错而不是静默降级。

### Docker 路径

- Dockerfile：`lab/lab0/docker/v6-mkl.Dockerfile`
- 镜像基于 `intel/oneapi-basekit:latest`
- 容器内用 `icpx -O3 -std=c++17 -qmkl` 编译 `v6_mkl`
- 适合功能验证，不适合与宿主机原生版本做严格公平性能对比

## 数据模型

这里没有数据库，但有统一的数据与结果模型。

### 输入数据模型

`MatrixRunConfig` 由 `matmul_common.hpp` 定义，包含：

- `m`
- `n`
- `k`
- `seed`
- `dump_matrix`

矩阵生成使用固定伪随机流程：`splitmix64` + `seed ^ salt`，其中 A 与 B 使用不同 salt。这意味着跨版本 checksum 应可对齐。

### 结果数据模型

`run_program` 统一输出：

- `time_sec`
- `checksum`
- `max_abs`

`benchmark_lab1_unified.py` 在 JSON 中还会追加：

- `avg_time_sec`
- `median_time_sec`
- `relative_speedup`
- `absolute_speedup`
- `gflops`
- `peak_percent`

### 当前结果文件

- `lab/lab0/results/report_benchmark.json`
- `lab/lab0/results/report_benchmark.csv`
- `lab/lab0/results/benchmark_unified_test.json`
- `lab/lab0/results/test_benchmark.json`
- `lab/lab0/results/report_scaling.csv`

从现有 `report_benchmark.json` 看，当前机器上 `v4/v5` 的原生性能明显优于 Docker 中的 `v6_mkl`；应将 `v6` 视为接口完成验证，而非当前平台上的公平性能上界。

## 测试与质量

### 自动化测试

- 测试文件：`lab/lab0/tests/test_submission_finalize.py`
- 类型：收尾验证/提交验证

覆盖点：

- `report_benchmark.json` 中必须包含 `v6_mkl`
- `report_benchmark.csv` 中必须包含 `v6_mkl`
- 兼容运行入口 `scripts/run_matrix_mul` 可执行
- 兼容 benchmark 入口 `scripts/benchmark_lab1.py --help` 可执行

### 质量策略

- 使用统一 seed 和 checksum 做跨版本一致性校验
- benchmark 脚本会检查同一版本重复运行时 checksum 是否一致
- benchmark 也会检查不同版本对同一规模的 checksum 是否一致

### 当前缺口

- 没有针对各个 C++ 内核的细粒度单元测试
- 没有自动测试 Docker 路径的环境可用性
- 没有专门验证 `--dump` 输出格式的测试
- README 与脚本名中的 `lab1` 历史命名会增加理解成本

## 常见问题 (FAQ)

### 为什么 `lab0` 里很多脚本名叫 `lab1`？

这是历史命名残留。当前实际串行实验目录是 `lab/lab0`，命令执行时请以真实路径为准。

### 为什么 `v6_mkl` 在当前结果里不比 `v4/v5` 快？

因为当前仓库说明与结果都表明 `v6_mkl` 默认通过 Docker 的 `linux/amd64` 仿真环境运行在 Apple Silicon 上，主要用于功能验证，不是公平性能对比。

### 修改实现时最容易破坏什么？

最容易破坏的是 stdout 输出契约。benchmark 和测试脚本都按 `key=value` 解析输出。

### 如何做单次小规模功能检查？

优先运行：

```bash
./lab/lab0/scripts/run_matrix_mul_unified v1_python 4 4 4 20250401 --dump
./lab/lab0/scripts/run_matrix_mul_unified v2_cpp_baseline 4 4 4 20250401 --dump
```

这样可以直接比较 A/B/C 输出与 checksum。

## 相关文件清单

- `lab/lab0/README.md`
- `lab/lab0/src/matmul_common.hpp`
- `lab/lab0/src/v1_python.py`
- `lab/lab0/src/v2_cpp_baseline.cpp`
- `lab/lab0/src/v3_cpp_loop_reorder.cpp`
- `lab/lab0/src/v4_cpp_optimized.cpp`
- `lab/lab0/src/v5_cpp_unrolled.cpp`
- `lab/lab0/src/v6_mkl.cpp`
- `lab/lab0/scripts/build_lab1.sh`
- `lab/lab0/scripts/run_matrix_mul_unified`
- `lab/lab0/scripts/run_matrix_mul`
- `lab/lab0/scripts/benchmark_lab1_unified.py`
- `lab/lab0/scripts/benchmark_lab1.py`
- `lab/lab0/scripts/run_v6_docker.sh`
- `lab/lab0/scripts/USAGE_UNIFIED.md`
- `lab/lab0/docker/README.md`
- `lab/lab0/docker/v6-mkl.Dockerfile`
- `lab/lab0/tests/test_submission_finalize.py`
- `lab/lab0/results/report_benchmark.json`

## 变更记录 (Changelog)

- 2026-04-03T12:40:33: 初始化模块文档，补充统一入口、版本路由、依赖、结果模型、测试策略与 FAQ。
