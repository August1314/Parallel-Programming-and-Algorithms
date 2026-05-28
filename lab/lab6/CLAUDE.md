# lab6 CLAUDE.md

## 模块职责

`lab/lab6` 是 Pthreads `parallel_for` 并行应用实验模块。

当前模块包含：

- Heated Plate 原始 OpenMP 对照版本；
- 基于 Pthreads `parallel_for` 的 Heated Plate 版本；
- `block`、`cyclic`、`dynamic` 三种 `parallel_for` 调度方式；
- Docker Linux 构建、benchmark、绘图、导表与测试闭环。

## 核心要求

- 核心程序保持稳定 `key=value` 输出；
- canonical 运行环境统一为 Docker Linux；
- Python 工作流统一使用 `uv run python ...`；
- 不改变课程参考资料中的边界条件：上边界 0，左右和下边界 100。

## 常用命令

```bash
./lab/lab6/scripts/docker_build_image.sh
./lab/lab6/scripts/docker_make.sh
./lab/lab6/scripts/run_openmp_heated_plate.sh 128 128 0.1 4
./lab/lab6/scripts/run_pthreads_heated_plate.sh block 128 128 0.1 4 8
./lab/lab6/scripts/run_pthreads_heated_plate.sh cyclic 128 128 0.1 4 8
./lab/lab6/scripts/run_pthreads_heated_plate.sh dynamic 128 128 0.1 4 8
cd ./lab/lab6 && uv run python ./scripts/benchmark.py
cd ./lab/lab6 && uv run python ./scripts/plot.py
cd ./lab/lab6 && uv run python ./scripts/export_report_assets.py
cd ./lab/lab6 && uv run python -m unittest ./tests/test_core_programs.py
cd ./lab/lab6 && uv run python -m unittest ./tests/test_benchmark.py
```
