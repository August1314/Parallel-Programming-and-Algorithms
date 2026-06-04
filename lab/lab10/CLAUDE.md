# Lab10 — CUDA 并行通用矩阵乘法 (GEMM)

## 概述

使用 CUDA 实现并行通用矩阵乘法 (C = A × B)，对比 3 种 kernel 实现的性能。实验在智算习堂（NVIDIA RTX 3090 / A100）平台上运行。

## 目录

```
lab10/
├── src/
│   ├── gemm.cu          # CUDA GEMM 实现（3种 kernel）
│   └── common.hpp       # 公共工具函数
├── scripts/
│   ├── build.sh         # 编译脚本
│   ├── run_gemm.sh      # 运行单次测试
│   ├── benchmark.py     # 性能基准测试
│   └── plot.py          # 可视化
├── report/
│   └── lab10_report.tex # LaTeX 实验报告
├── results/             # 基准测试结果 + 图表
│   └── figures/
├── Makefile
└── CLAUDE.md
```

## 3 种 Kernel

| ID | 名称 | 访存特征 | 性能预期 |
|----|------|----------|----------|
| 1 | naive | 全局内存，每线程一元素，A 冗余读取 | 基线（小矩阵可能不差） |
| 2 | tiled_16 | 共享内存分块 (TILE=16)，读写均合并 | 中等规模较优 |
| 3 | tiled_32 | 共享内存分块 (TILE=32)，更大复用率 | 大矩阵最优 |

## 快速开始（智算习堂）

```bash
# 构建
./scripts/build.sh

# 单次 GEMM 测试
./scripts/run_gemm.sh 512 512 512 1 16 16   # naive, 16×16 block
./scripts/run_gemm.sh 1024 1024 1024 2       # tiled_16
./scripts/run_gemm.sh 2048 2048 2048 3       # tiled_32

# 完整基准测试
uv run python scripts/benchmark.py

# 生成图表
uv run python scripts/plot.py
```

## 本地开发

本机为 Apple M4，无法编译/运行 CUDA 代码。所有 `.cu` 文件在本地编写后上传至智算习堂编译测试。

## 关键参数

- **矩阵规模**：M=N=K ∈ {128, 256, 512, 1024, 2048}
- **Kernel 类型**：naive (全局内存), tiled_16, tiled_32
- **Block 大小**：8×8, 16×16, 32×32（naive kernel 可变；tiled kernel 固定匹配 TILE_DIM）
- **GPU 架构**：sm_37 (平台 CUDA Toolkit 兼容层)
- **每配置重复**：3 次取平均

## 分析维度

1. **线程块大小影响**：对比 naive kernel 在 8×8, 16×16, 32×32 block 下的性能
2. **访存方式影响**：全局内存 (naive) vs 共享内存分块 (tiled_16/tiled_32)
3. **分块大小影响**：TILE=16 vs TILE=32 的共享内存使用效率
4. **矩阵规模扩展性**：从 128 到 2048 的性能变化趋势

## 输出格式

统一 key=value 输出，与 lab6/7/8/9 一致：

```
experiment=gemm
M=512
N=512
K=512
kernel=naive
tile_dim=0
block_dim=16x16
grid_dim=32x32
kernel_time_ms=0.123456
gflops=2174.0000
bandwidth_gb_s=300.0000
checksum=12345.6789
correct=true
```

## 报告结构

中山大学实验报告模板（ctexart），覆盖：实验目的、平台、问题描述、算法设计、CUDA 并行化策略、3 种 kernel 分析、block size 影响、矩阵规模扩展性、GFLOPS 与带宽分析。
