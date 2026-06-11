# Lab11 CUDA 卷积计算 — 设计文档

**日期**: 2026-06-11
**状态**: 已确认

## 1. 概述

使用 CUDA 实现 2D 卷积操作（CNN 风格，无 filter 翻转，无 bias），对比 4 种实现方式的性能。实验在智算习堂（NVIDIA RTX 3090 / A100）运行，本地 Apple M4 仅用于编码。

## 2. 4 种实现

| ID | 名称 | 方法 | 定位 |
|---|---|---|---|
| 1 | naive_conv | 直接滑窗，纯全局内存，每线程一个输出元素 | 性能下限基线 |
| 2 | tiled_conv | 直接滑窗，共享内存分块（含 halo 区域） | 手写优化最佳 |
| 3 | im2col_gemm | im2col 展开 + 内嵌 tiled GEMM (TILE=16/32) | 算法变换方案 |
| 4 | cudnn | cuDNN cudnnConvolutionForward | 库函数基线 |

## 3. 统一参数

- 输入通道 C=3，Kernel=3×3，Kernel 输入通道=3，Kernel 输出通道=3
- Stride ∈ {1, 2, 3}
- Padding 自动计算: `padding = (KH - 1) / 2 = 1`（"same" 模式，stride=1 时输出尺寸=输入尺寸）
- 无 bias (bias=0)
- 输入规模: [32, 64, 128, 256, 512, 1024, 2048, 4096]（等比）
- 输出通道 = 输入通道 = 3

## 4. 目录结构

```
lab11/
├── src/
│   ├── convolution.cu    # 4 种实现 + main（条件编译 cuDNN）
│   └── common.hpp        # CUDA error check, initTensor, CPU conv verify, checksum
├── scripts/
│   ├── build.sh          # 多目标: conv / cudnn / all
│   ├── run_conv.sh       # 单次运行
│   ├── benchmark.py      # 批量基准测试 → summary.csv
│   └── plot.py           # matplotlib 可视化
├── report/
│   └── lab11_report.tex  # 中山大学实验报告 (ctexart)
├── results/
│   └── figures/
├── Makefile
└── CLAUDE.md
```

## 5. 输出格式（key=value 契约）

```
experiment=convolution
H=256
W=256
C=3
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
memory_usage_mb=12.34
checksum=12345.6789
correct=true
```

## 6. 关键设计决策

### 6.1 im2col 时间分解

im2col 变换和 GEMM 计算分别计时，输出 `im2col_time_ms` 和 `gemm_time_ms`，总和 = `kernel_time_ms`。

### 6.2 OOM 处理

im2col 分配前检查显存需求（4096×4096×3 下 im2col 列矩阵 ~1.73 GB）。超限时 `exit(1)`，benchmark.py 标记 SKIP 并跳过后续更大规模。

### 6.3 cuDNN 条件编译

```c
#ifdef USE_CUDNN
#include <cudnn.h>
// cuDNN 相关代码
#endif
```

build.sh 三目标：
- `./scripts/build.sh` → 默认: kernels 1-3
- `./scripts/build.sh cudnn` → 带 cuDNN 版本 (`-arch=sm_86 -DUSE_CUDNN`)
- `./scripts/build.sh all` → 全部

### 6.4 验证策略

- 小规模 (≤256): CPU 精确验证 (5层循环)
- 大规模: checksum 交叉验证（所有 kernel 输出 checksum 必须一致）
- cuDNN: 使用 tolerance=1e-3（cuDNN 内部算法可能产生不同舍入路径）

### 6.5 GEMM kernel 命名

从 lab10 复制 GEMM kernel 时加 `conv` 前缀（如 `convGemmTiled16`），保持各 lab 独立可编译。

## 7. 分析维度

1. 4 种方法运行时间 vs 输入规模（runtime_vs_size.png）
2. 不同 stride 对性能的影响
3. im2col 时间分解（time_breakdown.png）
4. 显存占用对比（memory_usage.png）
5. naive_conv block size 影响 (8×8, 16×16, 32×32)
6. tiled_conv 的共享内存分块效果
7. 与 cuDNN 差距分析

## 8. 实现顺序

1. `common.hpp` — 公共工具
2. `convolution.cu` — 4 种 kernel + main
3. `Makefile` — 编译快捷方式
4. `scripts/build.sh` — 编译脚本
5. `scripts/run_conv.sh` — 单次运行
6. `scripts/benchmark.py` — 基准测试
7. `scripts/plot.py` — 可视化
8. `report/lab11_report.tex` — 实验报告
9. `CLAUDE.md` — 模块文档
