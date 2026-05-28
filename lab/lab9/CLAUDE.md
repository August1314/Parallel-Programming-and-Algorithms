# Lab9 — CUDA 矩阵转置

## 概述

使用 CUDA 实现并行矩阵转置，对比 3 种 kernel 实现的性能。实验在智算习堂（NVIDIA RTX 3090）平台上运行。

## 目录

```
lab9/
├── src/
│   ├── hello_world.cu    # CUDA Hello World 入门程序
│   ├── transpose.cu      # 矩阵转置（3种 kernel）
│   └── common.hpp        # 公共工具函数
├── scripts/
│   ├── build.sh          # 编译脚本
│   ├── run_hello.sh      # 运行 Hello World
│   ├── run_transpose.sh  # 运行转置单次测试
│   ├── benchmark.py      # 性能基准测试
│   └── plot.py           # 可视化
├── report/
│   └── lab9_report.tex   # LaTeX 实验报告
├── results/              # 基准测试结果 + 图表
├── Makefile
└── CLAUDE.md
```

## 3 种 Kernel

| ID | 名称 | 访存特征 | 性能预期 |
|----|------|----------|----------|
| 1 | naive | 读：非合并，写：合并 | 最差（基线） |
| 2 | coalesced_read | 读：合并，写：非合并 | 中等 |
| 3 | shared_mem | 读/写均合并（共享内存分块） | 最优 |

## 快速开始（智算习堂）

```bash
# 构建
./scripts/build.sh

# Hello World
./scripts/run_hello.sh 2 4 8

# 单次转置测试
./scripts/run_transpose.sh 1024 3 32 32

# 完整基准测试
uv run python scripts/benchmark.py

# 生成图表
uv run python scripts/plot.py
```

## 本地开发

本机为 Apple M4，无法编译/运行 CUDA 代码。所有 `.cu` 文件在本地编写后上传至智算习堂编译测试。

## 关键参数

- **矩阵规模 N**：512, 1024, 2048
- **Block 大小**：8×8, 16×16, 32×32
- **GPU 架构**：sm_86 (RTX 3090 Ampere)
- **每配置重复**：3 次取平均

## 输出格式

统一 key=value 输出，与 lab6/7/8 一致：

```
experiment=transpose
N=1024
kernel=shared_mem
block_dim=32x32
kernel_time_ms=0.123456
bandwidth_gb_s=250.0000
checksum=523456.0000
correct=true
```

## 报告结构

中山大学实验报告模板（ctexart），覆盖：实验目的、平台、算法设计、CUDA 并行策略、3 种 kernel 分析、block size 影响、矩阵规模扩展性、带宽分析。
