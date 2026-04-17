# lab3 CLAUDE.md

## 模块职责

`lab/lab3` 是 Pthreads 实验模块，当前包含核心程序、benchmark、可视化、表格导出、报告与基础自动化验证。

核心程序包括：

- `pthreads_matmul_v1`：矩阵乘法，按连续行块静态划分。
- `pthreads_matmul_v2_partitioned`：矩阵乘法，按循环行分配。
- `pthreads_array_sum_v1`：数组求和，每线程局部和，主线程合并。
- `pthreads_array_sum_v2_reduction`：数组求和，线程直接用 mutex 累加共享总和。

## 目录职责

- `src/`：Pthreads 核心实现与公共头。
- `bin/`：构建产物。
- `scripts/benchmark.py`：批量运行 benchmark，生成 `benchmark_results.json`、`summary_by_version.csv` 以及基础表格。
- `scripts/plot.py`：基于 summary 生成 runtime / speedup / efficiency / 对比图。
- `scripts/export_report_assets.py`：导出报告直接可消费的 tables 资产。
- `tests/`：核心程序契约测试、TDD 证据测试、流水线产物测试。
- `report/`：LaTeX 报告与最终 PDF。

## 输出契约

所有核心程序保持稳定 `key=value` 输出。
默认 seed 为 `20250401`。
`--dump` 仅在小规模输入下打印完整矩阵或数组。

## TDD 说明

- `test_tdd_demonstration.py` 用于保留“先失败后通过”的演示痕迹。
- `test_tdd_evidence.py` 用于校验 lab3 当前二进制输出满足既定契约。
- `test_benchmark.py` 用于验证 benchmark / plot / export 的闭环产物是否存在且非空。

## 常用命令

```bash
make -C ./lab/lab3
./lab/lab3/bin/pthreads_matmul_v1 4 4 4 20250401 2 --dump
./lab/lab3/bin/pthreads_matmul_v2_partitioned 4 4 4 20250401 2 --dump
./lab/lab3/bin/pthreads_array_sum_v1 16 20250401 4 --dump
./lab/lab3/bin/pthreads_array_sum_v2_reduction 16 20250401 4 --dump
cd ./lab/lab3 && uv run python ./scripts/benchmark.py
cd ./lab/lab3 && uv run python ./scripts/plot.py
cd ./lab/lab3 && uv run python ./scripts/export_report_assets.py
cd ./lab/lab3 && uv run python -m unittest ./tests/test_core_programs.py
cd ./lab/lab3 && uv run python -m unittest ./tests/test_benchmark.py
cd ./lab/lab3 && xelatex -interaction=nonstopmode report/lab3_report.tex
```
