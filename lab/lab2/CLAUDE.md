# lab2 CLAUDE.md

## 模块职责

`lab/lab2` 是 MPI 矩阵乘法进阶实验模块，包含点对点对照版、集合通信进阶版、benchmark、绘图与报告草稿。

## 核心要求

- 保持 stdout 为稳定 `key=value`
- `mpi_v2_collective` 必须使用集合通信和 `MPI_Type_create_struct`
- Python 工作流统一使用 `uv run python ...`

## 常用命令

```bash
make -C ./lab/lab2
mpirun -np 2 --oversubscribe ./lab/lab2/bin/mpi_matmul_v1 4 4 4 20250401 --dump
mpirun -np 2 --oversubscribe ./lab/lab2/bin/mpi_matmul_v2_collective 4 4 4 20250401 --dump
cd ./lab/lab2 && uv run python ./scripts/benchmark.py
cd ./lab/lab2 && uv run python ./scripts/plot.py
cd ./lab/lab2 && uv run python ./scripts/export_report_assets.py
cd ./lab/lab2 && uv run python -m unittest ./tests/test_benchmark.py
```
