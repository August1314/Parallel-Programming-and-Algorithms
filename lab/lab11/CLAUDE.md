# Lab11 — CUDA 卷积计算 (CNN Convolution)

## 概述

使用 CUDA 实现 2D CNN 卷积操作（无 filter 翻转，无 bias），对比 4 种实现方式的性能。实验在智算习堂（NVIDIA RTX 3090 / A100）平台上运行。

## 目录

```
lab11/
├── src/
│   ├── convolution.cu    # CUDA 卷积实现（4 种方法）
│   └── common.hpp        # 公共工具函数
├── scripts/
│   ├── build.sh          # 编译脚本（多目标: conv/cudnn/all）
│   ├── run_conv.sh       # 运行单次测试
│   ├── benchmark.py      # 性能基准测试
│   └── plot.py           # 可视化
├── report/
│   └── lab11_report.tex  # LaTeX 实验报告
├── results/              # 基准测试结果 + 图表
│   └── figures/
├── Makefile
└── CLAUDE.md
```

## 4 种实现

| ID | 名称 | 方法 | 定位 |
|----|------|------|------|
| 1 | naive_conv | 直接滑窗，纯全局内存，每线程一个输出元素 | 性能下限基线 |
| 2 | tiled_conv | 直接滑窗，共享内存分块（含 halo 区域） | 手写优化最佳 |
| 3 | im2col_gemm | im2col 展开 + 内嵌 tiled GEMM (TILE=16/32) | 算法变换方案 |
| 4 | cudnn | cuDNN cudnnConvolutionForward | 库函数基线 |

## 快速开始（智算习堂）

```bash
# 构建
./scripts/build.sh           # kernels 1-3
./scripts/build.sh cudnn     # 带 cuDNN 支持
./scripts/build.sh all       # 全部

# 单次测试
./scripts/run_conv.sh 256 1 1    # naive_conv, stride=1
./scripts/run_conv.sh 256 1 2    # tiled_conv
./scripts/run_conv.sh 256 1 3    # im2col_gemm
./scripts/run_conv.sh 256 1 4    # cuDNN

# 完整基准测试
uv run python scripts/benchmark.py

# 生成图表
uv run python scripts/plot.py
```

## 本地开发

本机为 Apple M4，无法编译/运行 CUDA 代码。所有 `.cu` 文件在本地编写后上传至智算习堂编译测试。

## 关键参数

- **输入规模**: H=W ∈ {32, 64, 128, 256, 512, 1024, 2048, 4096}
- **通道数**: C_in = C_out = 3
- **Kernel**: 3×3
- **Stride**: 1, 2, 3
- **Padding**: 1（"same" 模式，stride=1 时输出尺寸=输入尺寸）
- **Block 大小**: 8×8, 16×16, 32×32（naive_conv 可变；tiled_conv 固定 16×16）
- **GPU 架构**: sm_37 (基础), sm_86 (cuDNN)
- **每配置重复**: 3 次取平均

## 分析维度

1. 4 种方法的运行时间 vs 输入规模
2. 不同 stride 对性能的影响
3. im2col 变换开销 vs GEMM 效率的权衡（时间分解）
4. 显存占用对比
5. naive_conv block size 影响 (8×8, 16×16, 32×32)
6. 与 cuDNN 的差距分析

## 输出格式

统一 key=value 输出：

```
experiment=convolution
H=256
W=256
C=3
C_out=3
KH=3
KW=3
stride=1
padding=1
H_out=256
W_out=256
kernel=naive_conv
block_dim=16x16
grid_dim=16x16x3
kernel_time_ms=2.345678
im2col_time_ms=0.000000
gemm_time_ms=0.000000
gflops=1.2345
memory_usage_mb=12.34
checksum=12345.6789
correct=true
```

## 报告结构

中山大学实验报告模板（ctexart），覆盖：实验目的、平台、问题描述、4 种算法设计、CUDA 并行化策略、性能分析（7 维度）、与 cuDNN 对比、改进方向。
