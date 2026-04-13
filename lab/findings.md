# Findings

## 2026-04-10

### 仓库现状
- `lab0` 已建立统一运行入口、统一 benchmark、统一 stdout 键值输出契约。
- `lab1` 当前只有 `mpi_matmul_v1`，核心是点对点 `MPI_Send/MPI_Recv` 的按行划分实现。
- `lab2` 目录当前只有题目材料：`2-MPI矩阵乘法进阶.docx`、`2-MPI矩阵乘法进阶.pptx`，尚无代码骨架。

### 从题目提取出的硬性要求
- 在上次实验的 MPI-v1 基础上改进并讨论不同通信方式对性能的影响。
- 使用 MPI 集合通信实现并行矩阵乘法中的进程间通信。
- 使用 `MPI_Type_create_struct` 聚合 MPI 进程内变量后通信，例如 `m/n/k` 等元数据。
- 记录 1/2/4/8/16 进程与 128/256/512/1024/2048 规模下的时间开销，并做性能和扩展性分析。
- 选做：尝试不同数据/任务划分方式。

### 可复用模式
- `lab0` 强调统一入口 + benchmark 输出契约稳定。
- `lab1` 已有矩阵生成、checksum、max_abs、`--dump` 控制、benchmark 脚本等可直接复用的实验骨架。

### 已确认的 lab2 设计
- `lab2` 采用独立实验模块，不依赖 `lab1` 运行。
- 版本策略为“两版本实现、结构预留第三版”：
  - `mpi_v1_p2p`：点对点对照组
  - `mpi_v2_collective`：集合通信主版本
- `mpi_v2_collective` 的主通信方式为：
  - `MPI_Type_create_struct` 广播配置
  - `MPI_Scatterv` 分发 `A`
  - `MPI_Bcast` 广播 `B`
  - `MPI_Gatherv` 汇总 `C`
- `lab2/scripts/benchmark.py` 将统一跑多版本、多进程、多规模。
- `lab2/scripts/plot.py` 将生成折线图、加速比图、效率图、版本对比图、热力图。
- Python 依赖与命令统一以 `uv` 工作流管理。
- 报告目标是“代码 -> 数据 -> 图表 -> 报告”闭环，并突出可视化丰富度。
