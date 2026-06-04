#!/usr/bin/env python3
"""Generate performance plots for lab10 CUDA GEMM benchmark results."""
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).parent.absolute()
LAB_DIR = SCRIPT_DIR.parent
RESULTS_DIR = LAB_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CSV_PATH = RESULTS_DIR / "summary.csv"

# 中文字体：尝试多个备选
for _font in ["Songti SC", "Noto Sans CJK SC", "PingFang SC",
              "Heiti SC", "SimHei", "Arial Unicode MS", "DejaVu Sans"]:
    try:
        matplotlib.font_manager.findfont(_font, fallback_to_default=False)
        plt.rcParams["font.family"] = _font
        break
    except Exception:
        continue

plt.rcParams["axes.unicode_minus"] = False


def load_data(csv_path: Path) -> list[dict]:
    """读取 benchmark CSV 数据"""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "ok":
                row["M"] = int(row["M"])
                row["N"] = int(row["N"])
                row["K"] = int(row["K"])
                row["avg_kernel_time_ms"] = float(row["avg_kernel_time_ms"])
                row["gflops"] = float(row.get("gflops", 0))
                row["bandwidth_gb_s"] = float(row.get("bandwidth_gb_s", 0))
                row["block_x"] = int(row["block_x"])
                row["block_y"] = int(row["block_y"])
                rows.append(row)
    return rows


def plot_runtime_vs_size(data: list[dict]):
    """图1: Kernel 运行时间 vs 矩阵规模（最佳 block 大小）"""
    # 对每个 kernel × size 选最佳 block
    kernels = ["naive", "tiled_16", "tiled_32"]
    sizes = sorted(set(r["M"] for r in data))

    fig, ax = plt.subplots(figsize=(8, 5))

    for kernel in kernels:
        times = []
        for size in sizes:
            kernel_data = [r for r in data
                           if r["M"] == size and r["kernel"] == kernel]
            if kernel_data:
                best = min(kernel_data,
                          key=lambda r: r["avg_kernel_time_ms"])
                times.append(best["avg_kernel_time_ms"])
            else:
                times.append(np.nan)
        ax.plot(sizes, times, "o-", linewidth=1.5, markersize=6,
                label=kernel)

    ax.set_xlabel("矩阵规模 (M=N=K)")
    ax.set_ylabel("Kernel 运行时间 (ms)")
    ax.set_title("GEMM Kernel 运行时间 vs 矩阵规模")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "runtime_vs_size.png", dpi=150)
    plt.close(fig)
    print("  -> runtime_vs_size.png")


def plot_gflops_vs_size(data: list[dict]):
    """图2: GFLOPS vs 矩阵规模"""
    kernels = ["naive", "tiled_16", "tiled_32"]
    sizes = sorted(set(r["M"] for r in data))

    fig, ax = plt.subplots(figsize=(8, 5))

    for kernel in kernels:
        gflops_vals = []
        for size in sizes:
            kernel_data = [r for r in data
                           if r["M"] == size and r["kernel"] == kernel]
            if kernel_data:
                best = max(kernel_data, key=lambda r: r["gflops"])
                gflops_vals.append(best["gflops"])
            else:
                gflops_vals.append(np.nan)
        ax.plot(sizes, gflops_vals, "s-", linewidth=1.5, markersize=6,
                label=kernel)

    ax.set_xlabel("矩阵规模 (M=N=K)")
    ax.set_ylabel("GFLOPS")
    ax.set_title("GEMM 计算性能 vs 矩阵规模")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "gflops_vs_size.png", dpi=150)
    plt.close(fig)
    print("  -> gflops_vs_size.png")


def plot_block_size_impact(data: list[dict]):
    """图3: Block 大小对性能的影响（naive kernel）"""
    naive_data = [r for r in data if r["kernel"] == "naive"]
    sizes = sorted(set(r["M"] for r in naive_data))

    block_labels = ["8×8", "16×16", "32×32"]
    block_configs = [(8, 8), (16, 16), (32, 32)]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(sizes))
    width = 0.25

    for i, (bx, by) in enumerate(block_configs):
        times = []
        for size in sizes:
            matches = [r for r in naive_data
                       if r["M"] == size
                       and r["block_x"] == bx and r["block_y"] == by]
            if matches:
                times.append(matches[0]["avg_kernel_time_ms"])
            else:
                times.append(np.nan)
        ax.bar(x + i * width, times, width, label=block_labels[i])

    ax.set_xlabel("矩阵规模 (M=N=K)")
    ax.set_ylabel("Kernel 运行时间 (ms)")
    ax.set_title("Block 大小对 Naive Kernel 性能的影响")
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "block_size_impact.png", dpi=150)
    plt.close(fig)
    print("  -> block_size_impact.png")


def plot_speedup(data: list[dict]):
    """图4: 加速比（相对 naive 最佳配置）"""
    sizes = sorted(set(r["M"] for r in data))
    kernels = ["naive", "tiled_16", "tiled_32"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for kernel in kernels:
        speedups = []
        for size in sizes:
            kernel_data = [r for r in data
                           if r["M"] == size and r["kernel"] == kernel]
            if not kernel_data:
                speedups.append(np.nan)
                continue
            best = min(kernel_data, key=lambda r: r["avg_kernel_time_ms"])

            # baseline = naive 最佳
            naive_data = [r for r in data
                          if r["M"] == size and r["kernel"] == "naive"]
            if naive_data:
                baseline = min(naive_data,
                              key=lambda r: r["avg_kernel_time_ms"])
                speedup = baseline["avg_kernel_time_ms"] / best["avg_kernel_time_ms"]
                speedups.append(speedup)
            else:
                speedups.append(np.nan)

        ax.plot(sizes, speedups, "o-", linewidth=1.5, markersize=6,
                label=kernel)

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("矩阵规模 (M=N=K)")
    ax.set_ylabel("加速比 (相对 naive 最佳)")
    ax.set_title("各 Kernel 相对 Naive 最佳配置的加速比")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "speedup.png", dpi=150)
    plt.close(fig)
    print("  -> speedup.png")


def plot_heatmap(data: list[dict]):
    """图5: 运行时间热力图 (Kernel × Size)"""
    kernels = ["naive", "tiled_16", "tiled_32"]
    sizes = sorted(set(r["M"] for r in data))

    heatmap = np.zeros((len(kernels), len(sizes)))

    for i, kernel in enumerate(kernels):
        for j, size in enumerate(sizes):
            kernel_data = [r for r in data
                           if r["M"] == size and r["kernel"] == kernel]
            if kernel_data:
                best = min(kernel_data,
                          key=lambda r: r["avg_kernel_time_ms"])
                heatmap[i, j] = best["avg_kernel_time_ms"]
            else:
                heatmap[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(heatmap, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_yticks(range(len(kernels)))
    ax.set_yticklabels(kernels)
    ax.set_xlabel("矩阵规模 (M=N=K)")
    ax.set_ylabel("Kernel")
    ax.set_title("运行时间热力图 (ms)")

    # 标注数值
    for i in range(len(kernels)):
        for j in range(len(sizes)):
            val = heatmap[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8)

    fig.colorbar(im, ax=ax, label="ms")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "heatmap.png", dpi=150)
    plt.close(fig)
    print("  -> heatmap.png")


def main():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found. Run benchmark.py first.")
        sys.exit(1)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data(CSV_PATH)
    print(f"Loaded {len(data)} records from {CSV_PATH}")
    print("Generating figures...")

    plot_runtime_vs_size(data)
    plot_gflops_vs_size(data)
    plot_block_size_impact(data)
    plot_speedup(data)
    plot_heatmap(data)

    print(f"Done. Figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
