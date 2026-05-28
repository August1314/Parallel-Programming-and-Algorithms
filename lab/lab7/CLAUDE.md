# lab7 CLAUDE.md

## 模块职责

`lab/lab7` 是 MPI 并行 FFT 与 parallel_for 应用分析实验模块。

当前模块包含：

- 串行 FFT 参考实现（Cooley-Tukey 算法）；
- MPI 并行 FFT 实现（6-step 转置算法）；
- Lab6 heated_plate 可扩展性分析；
- Valgrind massif 内存分析；
- Docker Linux 构建、benchmark、绘图、导表与测试闭环。

## 核心要求

- 核心程序保持稳定 `key=value` 输出；
- canonical 运行环境统一为 Docker Linux；
- MPI 程序使用 `mpicxx` 编译，`mpirun` 运行；
- Python 工作流统一使用 `uv run python ...`。

## 常用命令

```bash
# 构建
./lab/lab7/scripts/build.sh

# 运行串行 FFT
./lab/lab7/scripts/run_serial.sh 1024 331.0 --dump

# 运行 MPI FFT
./lab/lab7/scripts/run_mpi_fft.sh 1024 4 331.0 --dump

# FFT benchmark
cd ./lab/lab7 && uv run python ./scripts/benchmark_fft.py

# Lab6 heated_plate 扩展分析
cd ./lab/lab7 && uv run python ./scripts/benchmark_lab6_extended.py

# Valgrind massif 内存分析
./lab/lab7/scripts/run_valgrind.sh

# 绘图
cd ./lab/lab7 && uv run python ./scripts/plot.py

# 导出报告素材
cd ./lab/lab7 && uv run python ./scripts/export_report_assets.py

# 编译报告
cd ./lab/lab7/report && xelatex lab7_report.tex && xelatex lab7_report.tex
```
