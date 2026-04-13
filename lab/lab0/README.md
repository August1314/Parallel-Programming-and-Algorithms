# Lab1 串行矩阵乘法

## 目录结构

- `src`：Python/C++ 源码
- `scripts`：构建、运行、基准测试脚本
- `bin`：构建产物
- `results`：基准测试结果

## 版本说明

- `v1_python`：纯 Python 三重循环
- `v2_cpp_baseline`：C++ 基础三重循环
- `v3_cpp_loop_reorder`：调整循环顺序改善缓存访问
- `v4_cpp_optimized`：基于 `v3`，启用 `-O3`
- `v5_cpp_unrolled`：在 `v4` 基础上做有限循环展开
- `v6_mkl`：调用 `cblas_dgemm`，需要 Intel MKL 环境

## 使用方式

构建：

```bash
./lab/lab1/scripts/build_lab1.sh
```

运行单个版本：

```bash
./lab/lab1/scripts/run_matrix_mul v2_cpp_baseline 512 512 512 20250401
./lab/lab1/scripts/run_matrix_mul v1_python 4 4 4 20250401 --dump
```

运行基准测试：

```bash
./lab/lab1/scripts/benchmark_lab1.py --sizes 512x512x512 1024x1024x1024 --repeat 3 --peak-gflops 100
```

输出指标包括：

- `time_sec`：单次运行耗时
- `checksum`：结果矩阵校验值
- `max_abs`：结果矩阵中元素绝对值最大值
- `relative_speedup`：相对前一个版本的加速比
- `absolute_speedup`：相对 Python 基线的绝对加速比
- `gflops`：浮点性能
- `peak_percent`：相对于机器峰值性能的百分比

## 报告填写建议

实验报告可按以下结构填写：

1. 实验目的：说明矩阵乘法及串行优化分析目标。
2. 实验过程和核心代码：按 6 个版本解释优化思路、关键代码和构建方式。
3. 实验结果：粘贴基准测试表格，分析缓存局部性、编译优化、循环展开对性能的影响。
4. 实验感想：说明实现过程中遇到的问题，例如编译参数、结果校验、MKL 环境缺失及处理方式。

## MKL 说明

当前仓库保留了 `v6_mkl` 的源码和统一接口。若本机无 Intel MKL：

- `build_lab1.sh` 仍会构建可执行文件；
- 运行 `v6_mkl` 时会直接报出环境缺失；
- 在 Ubuntu/Intel + MKL 环境中重新构建后即可补跑该版本数据。
