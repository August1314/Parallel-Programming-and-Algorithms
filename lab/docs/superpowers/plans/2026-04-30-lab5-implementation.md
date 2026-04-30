# Lab5 实施计划

日期：2026-04-30

## 1. 总体顺序

1. 创建 `lab5` 模块骨架与文档。
2. 实现公共矩阵协议。
3. 实现 OpenMP 三个版本。
4. 实现 `parallel_for` 动态库与矩阵乘法驱动程序。
5. 补齐 Docker 构建与运行脚本。
6. 实现 benchmark、plot、export。
7. 编写测试并完成实际验证。

## 2. 关键文件

- `src/matmul_common.hpp`：公共输入输出与矩阵工具
- `src/openmp_matmul.cpp`：OpenMP 三版本共用源码
- `src/parallel_for.hpp` / `src/parallel_for.cpp`：动态库接口与实现
- `src/parallel_for_matmul.cpp`：库调用示例与性能测试程序
- `Makefile`：Linux 容器内构建规则
- `scripts/*.sh`：Docker 构建与运行入口
- `scripts/benchmark.py`：统一采集性能数据
- `scripts/plot.py`：统一生成图表
- `scripts/export_report_assets.py`：导出报告表格
- `tests/*.py`：回归测试

## 3. 验证检查点

### 检查点 1：构建

- `./lab/lab5/scripts/docker_build_image.sh`
- `./lab/lab5/scripts/docker_make.sh`

期望：

- `lab/lab5/bin/` 下出现 3 个 OpenMP 二进制、1 个 `.so`、1 个 `parallel_for_matmul`

### 检查点 2：核心程序

- OpenMP 三版本可在相同输入下输出相同 `checksum`
- `parallel_for_matmul` 可运行且输出相同 `checksum`
- `--dump` 在小规模下打印 `A/B/C`

### 检查点 3：动态库加载

- `parallel_for_matmul` 在 Docker Linux 中可直接运行
- `ldd ./bin/parallel_for_matmul` 中出现 `libparallel_for.so`

### 检查点 4：数据闭环

- `benchmark.py` 产出 JSON / CSV
- `plot.py` 产出全部图表
- `export_report_assets.py` 产出报告表格

## 4. 风险与处理

- 宿主机不能直接运行 Linux 二进制：
  - 统一通过 Docker 容器执行
- `.so` 找不到：
  - 链接时设置 `rpath=$ORIGIN`
- benchmark 过慢：
  - 通过环境变量覆盖线程数、规模和重复次数，供测试缩小配置

## 5. 完成标准

- 4 个版本都能在统一 Docker 环境构建和运行
- OpenMP 三种调度具备可比较的性能数据
- Linux `.so` 真实生成并被其他程序调用
- benchmark、图表、表格和报告骨架闭环成立
