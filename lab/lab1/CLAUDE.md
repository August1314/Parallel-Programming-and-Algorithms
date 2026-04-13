[根目录](../../CLAUDE.md) > [lab](../) > **lab1**

# lab1 CLAUDE.md

## 模块职责

`lab/lab1` 是 MPI 矩阵乘法实验模块，核心目标是把矩阵 A 按行切分到多个进程，广播或发送完整矩阵 B，在各 rank 上完成本地乘法，再由 rank 0 汇总结果并评估扩展性能。

当前模块聚焦一个实现版本：

- `mpi_matmul_v1`：基于点对点 `MPI_Send/MPI_Recv` 的按行划分方案。

## 入口与启动

### 主入口

- 构建入口：`lab/lab1/Makefile`
- 程序入口：`lab/lab1/src/mpi_matmul_v1.cpp`
- benchmark 入口：`lab/lab1/scripts/benchmark.py`

### 构建与运行命令

```bash
# 构建
make -C ./lab/lab1

# 单次运行
mpirun -np 4 --oversubscribe ./lab/lab1/bin/mpi_matmul_v1 512 512 512 20250401

# 小规模调试并打印矩阵
mpirun -np 2 --oversubscribe ./lab/lab1/bin/mpi_matmul_v1 4 4 4 20250401 --dump

# 跑全量 benchmark
python3 ./lab/lab1/scripts/benchmark.py

# 清理
make -C ./lab/lab1 clean
```

### Makefile 行为

- 编译器：`mpicxx`
- 编译参数：`-O3 -std=c++17 -Wall -Wextra`
- 目标文件：`bin/mpi_matmul_v1`

## 对外接口

### 命令行接口

程序使用格式：

```bash
mpirun -np <procs> ./lab/lab1/bin/mpi_matmul_v1 <m> <n> <k> [seed] [--dump]
```

`parse_args` 支持：

- 必选维度：`m n k`
- 可选种子：`seed`，默认 `20250401`
- 可选开关：`--dump`

### 输出接口

仅 rank 0 输出结构化结果，格式为稳定的 `key=value`：

- `version=mpi_v1_p2p`
- `m`, `n`, `k`
- `seed`
- `processes`
- `time_sec`
- `checksum`
- `max_abs`

benchmark 脚本依赖这套输出契约做解析。

## 关键依赖与配置

### 依赖

- MPI 工具链，尤其是 `mpicxx` 与 `mpirun`
- Python 3（运行 benchmark 脚本）
- C++17

### benchmark 默认配置

`lab/lab1/scripts/benchmark.py` 内置：

- 进程数：`[1, 2, 4, 8, 16]`
- 矩阵规模：`[128, 256, 512, 1024, 2048]`
- 固定 seed：`20250401`
- 每组重复次数：`3`
- 运行参数包含 `--oversubscribe`

这意味着在本机核数不足时也会强制启动更多 MPI 进程，结果更偏向教学实验而不是严格生产环境评测。

## 数据模型

这里没有数据库，但有两类重要数据模型。

### 运行配置

`matmul_mpi.hpp` 中的 `MatrixRunConfig`：

- `m`
- `n`
- `k`
- `seed`
- `dump_matrix`

### 分布式切分模型

在 `mpi_matmul_v1.cpp` 中：

- `row_counts[p]`：第 `p` 个 rank 负责的行数
- `row_displs[p]`：第 `p` 个 rank 在全局矩阵 A/C 中的起始行偏移
- `local_a`：当前 rank 持有的 A 子块
- `b`：每个 rank 都持有完整 B
- `local_c`：当前 rank 计算的 C 子块

切分规则为：

- 每个进程先拿到 `m / size` 行
- 余数 `m % size` 依次分配给前面的 rank

这让非整除场景也能正常工作。

### 结果文件

- `lab/lab1/results/benchmark_results.json`
- `lab/lab1/results/performance_table.csv`

从现有结果看：

- `512` 和 `1024` 规模在 `4` 进程附近表现较好；
- `16` 进程不一定优于 `8` 进程，说明通信和 oversubscribe 开销已开始抵消收益；
- 这与当前实现使用显式点对点发送、且每个 rank 都接收完整 B 的通信模型一致。

## 测试与质量

### 当前验证方式

- 无独立 `tests/` 目录。
- 主要依赖 `scripts/benchmark.py` 对多组参数进行批量运行。
- 程序输出 `checksum` 与 `max_abs` 用于结果稳定性检查。

### 质量关注点

- `MPI_Send/MPI_Recv` 的 tag 与配对必须一致：
  - tag `0`：A 子块
  - tag `1`：完整 B
  - tag `2`：C 子块回传
- 只有 rank 0 持有完整 `A` 与最终 `C`。
- `MPI_Barrier` 与 `MPI_Wtime` 决定计时边界；若改动流程，要重新确认测量区间是否合理。
- `--dump` 只在矩阵规模不超过 `8x8x8` 时真正打印，避免输出爆炸。

### 当前缺口

- 没有自动断言不同进程数下 checksum 一致的测试脚本
- 没有对错误输入、进程数边界、非整除行分配的专项回归测试
- 没有更高效的 collective 通信版本可用于对比，如 `MPI_Bcast` / `MPI_Scatterv` / `MPI_Gatherv`
- benchmark 脚本遇到失败时错误采集较弱，只简单解析 stdout

## 常见问题 (FAQ)

### 为什么 `lab1` 只有一个 MPI 版本？

当前仓库聚焦点对点实现的教学版本，性能分析主要通过进程数和问题规模变化体现，而不是多内核版本横向对比。

### 这个实现是广播 B 吗？

逻辑上是“所有 rank 都收到完整 B”，但实现方式不是 `MPI_Bcast`，而是 rank 0 逐个 `MPI_Send` 给其他进程。

### 为什么大规模下进程数越多不一定越快？

因为当前实现存在通信、收发同步和 oversubscribe 开销；当进程数继续增加时，这些开销可能超过计算收益。

### 最重要的改动风险点是什么？

MPI 通信配对与结果汇总位置。只要 tag、偏移、行数或 rank 0 汇总逻辑错位，就可能出现死锁或错误结果。

## 相关文件清单

- `lab/lab1/Makefile`
- `lab/lab1/src/matmul_mpi.hpp`
- `lab/lab1/src/mpi_matmul_v1.cpp`
- `lab/lab1/scripts/benchmark.py`
- `lab/lab1/results/benchmark_results.json`
- `lab/lab1/results/performance_table.csv`
- `lab/lab1/report/lab1_report.tex`

## 变更记录 (Changelog)

- 2026-04-03T12:40:33: 初始化模块文档，补充 MPI 架构、运行命令、数据分发模型、benchmark 配置与主要缺口。
