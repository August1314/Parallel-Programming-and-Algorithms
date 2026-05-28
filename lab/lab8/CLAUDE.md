# lab8 CLAUDE.md

## 模块职责

`lab/lab8` 是 OpenMP 并行多源最短路径搜索（APSP）实验模块。

当前模块包含：

- 串行 Repeated Dijkstra APSP 参考实现；
- OpenMP 并行 Repeated Dijkstra APSP 实现；
- Flower（花网络）和 Mouse（鼠脑网络）两个真实网络数据集；
- build、benchmark、plot 脚本与测试闭环。

## 核心要求

- 核心程序保持稳定 `key=value` 输出；
- 串行使用 Apple Clang，OpenMP 使用 GCC-15（Homebrew）；
- Python 工作流统一使用 `uv run python ...`；
- 报告使用 xelatex + ctexart 编译。

## 常用命令

```bash
# 构建
./lab/lab8/scripts/build.sh

# 生成测试查询
cd ./lab/lab8 && uv run python ./scripts/generate_queries.py

# 运行串行
./lab/lab8/bin/apsp_serial data/updated_flower.csv data/queries_updated_flower.txt
./lab/lab8/bin/apsp_serial data/updated_mouse.csv data/queries_updated_mouse.txt

# 运行 OpenMP 并行
OMP_NUM_THREADS=4 ./lab/lab8/bin/apsp_openmp data/updated_flower.csv data/queries_updated_flower.txt 4
OMP_NUM_THREADS=8 ./lab/lab8/bin/apsp_openmp data/updated_mouse.csv data/queries_updated_mouse.txt 8

# Benchmark
cd ./lab/lab8 && uv run python ./scripts/benchmark.py

# 绘图
cd ./lab/lab8 && uv run python ./scripts/plot.py

# 编译报告
cd ./lab/lab8/report && xelatex lab8_report.tex && xelatex lab8_report.tex
```
