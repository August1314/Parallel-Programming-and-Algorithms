# lab5 CLAUDE.md

## 模块职责

`lab/lab5` 是 OpenMP 与 `parallel_for` 动态库实验模块。

当前模块包含：

- OpenMP 矩阵乘法三版本：默认调度、`schedule(static,1)`、`schedule(dynamic,1)`
- `parallel_for` 动态链接库 `libparallel_for.so`
- 基于 `parallel_for` 的矩阵乘法驱动程序
- Docker Linux 构建、benchmark、绘图、导表与测试闭环

## 核心要求

- 核心程序保持稳定 `key=value` 输出
- canonical 运行环境统一为 Docker Linux
- `parallel_for` 仅实现连续块分配，不扩展额外调度参数
- Python 工作流统一使用 `uv run python ...`

## 常用命令

```bash
./lab/lab5/scripts/docker_build_image.sh
./lab/lab5/scripts/docker_make.sh
./lab/lab5/scripts/run_openmp_matmul.sh default 512 512 512 20250401 4
./lab/lab5/scripts/run_parallel_for_matmul.sh 512 512 512 20250401 4
cd ./lab/lab5 && uv run python ./scripts/benchmark.py
cd ./lab/lab5 && uv run python ./scripts/plot.py
cd ./lab/lab5 && uv run python ./scripts/export_report_assets.py
cd ./lab/lab5 && uv run python -m unittest ./tests/test_core_programs.py
cd ./lab/lab5 && uv run python -m unittest ./tests/test_benchmark.py
```
