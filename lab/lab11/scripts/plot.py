#!/usr/bin/env python3
"""Generate performance plots for lab11 CUDA Convolution benchmark."""
from __future__ import annotations
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

# Chinese font: try multiple fallbacks
for _font in ["Songti SC", "Noto Sans CJK SC", "PingFang SC",
              "Heiti SC", "SimHei", "Arial Unicode MS", "DejaVu Sans"]:
    try:
        matplotlib.font_manager.findfont(_font, fallback_to_default=False)
        plt.rcParams["font.family"] = _font
        break
    except Exception:
        continue

plt.rcParams["axes.unicode_minus"] = False

KERNEL_NAMES = ["naive_conv", "tiled_conv", "im2col_gemm", "cudnn"]


def load_data(csv_path: Path) -> list[dict]:
    """Load benchmark CSV data."""
    rows: list[dict] = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "ok":
                row["H"] = int(row["H"])
                row["stride"] = int(row["stride"])
                row["avg_kernel_time_ms"] = float(row["avg_kernel_time_ms"])
                row["im2col_time_ms"] = float(row.get("im2col_time_ms", 0))
                row["gemm_time_ms"] = float(row.get("gemm_time_ms", 0))
                row["gflops"] = float(row.get("gflops", 0))
                row["memory_usage_mb"] = float(row.get("memory_usage_mb", 0))
                row["block_x"] = int(row.get("block_x", 0))
                row["block_y"] = int(row.get("block_y", 0))
                rows.append(row)
    return rows


# ============================================================
# Figure 1: Runtime vs input size (stride=1, best config per kernel)
# ============================================================
def plot_runtime_vs_size(data: list[dict]):
    stride1 = [r for r in data if r["stride"] == 1]
    sizes = sorted(set(r["H"] for r in stride1))

    fig, ax = plt.subplots(figsize=(8, 5))
    for kn in KERNEL_NAMES:
        times = []
        for sz in sizes:
            kd = [r for r in stride1 if r["H"] == sz and r["kernel"] == kn]
            times.append(min(r["avg_kernel_time_ms"] for r in kd) if kd else np.nan)
        ax.plot(sizes, times, "o-", linewidth=1.5, markersize=6, label=kn)

    ax.set_xlabel("输入规模 (H=W)")
    ax.set_ylabel("运行时间 (ms)")
    ax.set_title("卷积运行时间 vs 输入规模 (stride=1)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "runtime_vs_size.png", dpi=150)
    plt.close(fig)
    print("  -> runtime_vs_size.png")


# ============================================================
# Figure 2: Stride impact (one subplot per kernel)
# ============================================================
def plot_stride_impact(data: list[dict]):
    kernels_plot = ["naive_conv", "tiled_conv", "im2col_gemm"]
    strides = sorted(set(r["stride"] for r in data))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, kn in enumerate(kernels_plot):
        ax = axes[idx]
        kd = [r for r in data if r["kernel"] == kn]
        sizes = sorted(set(r["H"] for r in kd))
        for s in strides:
            times = []
            for sz in sizes:
                matches = [r for r in kd if r["H"] == sz and r["stride"] == s]
                times.append(
                    min(r["avg_kernel_time_ms"] for r in matches) if matches
                    else np.nan)
            ax.plot(sizes, times, "o-", linewidth=1.5, markersize=4,
                    label=f"stride={s}")
        ax.set_xlabel("输入规模 (H=W)")
        ax.set_ylabel("运行时间 (ms)")
        ax.set_title(kn)
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)

    fig.suptitle("不同 Stride 对卷积性能的影响")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "stride_impact.png", dpi=150)
    plt.close(fig)
    print("  -> stride_impact.png")


# ============================================================
# Figure 3: im2col time breakdown (stacked bar)
# ============================================================
def plot_time_breakdown(data: list[dict]):
    im2col_data = [r for r in data
                   if r["kernel"] == "im2col_gemm" and r["stride"] == 1]
    if not im2col_data:
        print("  -> time_breakdown.png (skipped: no im2col data)")
        return

    sizes = sorted(set(r["H"] for r in im2col_data))

    im2col_times = []
    gemm_times = []
    for sz in sizes:
        matches = [r for r in im2col_data if r["H"] == sz]
        if matches:
            best = min(matches, key=lambda r: r["avg_kernel_time_ms"])
            im2col_times.append(best["im2col_time_ms"])
            gemm_times.append(best["gemm_time_ms"])
        else:
            im2col_times.append(0)
            gemm_times.append(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(sizes))
    width = 0.35
    ax.bar(x, im2col_times, width, label="im2col 变换时间", color="#E8833A")
    ax.bar(x, gemm_times, width, bottom=im2col_times,
           label="GEMM 计算时间", color="#4A90D9")
    ax.set_xlabel("输入规模 (H=W)")
    ax.set_ylabel("时间 (ms)")
    ax.set_title("im2col+GEMM 时间分解 (stride=1)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "time_breakdown.png", dpi=150)
    plt.close(fig)
    print("  -> time_breakdown.png")


# ============================================================
# Figure 4: Memory usage comparison
# ============================================================
def plot_memory_usage(data: list[dict]):
    stride1 = [r for r in data if r["stride"] == 1]
    sizes = sorted(set(r["H"] for r in stride1))

    fig, ax = plt.subplots(figsize=(8, 5))
    for kn in KERNEL_NAMES:
        mems = []
        for sz in sizes:
            kd = [r for r in stride1 if r["H"] == sz and r["kernel"] == kn]
            mems.append(kd[0]["memory_usage_mb"] if kd else np.nan)
        ax.plot(sizes, mems, "s-", linewidth=1.5, markersize=6, label=kn)

    ax.set_xlabel("输入规模 (H=W)")
    ax.set_ylabel("显存占用 (MB)")
    ax.set_title("各方法显存占用对比 (stride=1)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "memory_usage.png", dpi=150)
    plt.close(fig)
    print("  -> memory_usage.png")


# ============================================================
# Figure 5: Speedup relative to naive_conv best
# ============================================================
def plot_speedup(data: list[dict]):
    stride1 = [r for r in data if r["stride"] == 1]
    sizes = sorted(set(r["H"] for r in stride1))

    fig, ax = plt.subplots(figsize=(8, 5))
    for kn in KERNEL_NAMES:
        speedups = []
        for sz in sizes:
            kd = [r for r in stride1 if r["H"] == sz and r["kernel"] == kn]
            if not kd:
                speedups.append(np.nan)
                continue
            best_time = min(r["avg_kernel_time_ms"] for r in kd)

            naive_kd = [r for r in stride1
                        if r["H"] == sz and r["kernel"] == "naive_conv"]
            if naive_kd:
                baseline = min(r["avg_kernel_time_ms"] for r in naive_kd)
                speedups.append(baseline / best_time)
            else:
                speedups.append(np.nan)
        ax.plot(sizes, speedups, "o-", linewidth=1.5, markersize=6, label=kn)

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("输入规模 (H=W)")
    ax.set_ylabel("加速比 (相对 naive_conv)")
    ax.set_title("各方法相对 Naive 的加速比 (stride=1)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "speedup.png", dpi=150)
    plt.close(fig)
    print("  -> speedup.png")


# ============================================================
# Figure 6: Block size impact on naive_conv
# ============================================================
def plot_block_size_impact(data: list[dict]):
    naive_data = [r for r in data
                  if r["kernel"] == "naive_conv" and r["stride"] == 1]
    sizes = sorted(set(r["H"] for r in naive_data))
    block_configs = [(8, 8), (16, 16), (32, 32)]
    block_labels = ["8x8", "16x16", "32x32"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(sizes))
    width = 0.25

    for i, (bx, by) in enumerate(block_configs):
        times = []
        for sz in sizes:
            matches = [r for r in naive_data
                       if r["H"] == sz
                       and r["block_x"] == bx
                       and r["block_y"] == by]
            times.append(matches[0]["avg_kernel_time_ms"] if matches else np.nan)
        ax.bar(x + i * width, times, width, label=block_labels[i])

    ax.set_xlabel("输入规模 (H=W)")
    ax.set_ylabel("运行时间 (ms)")
    ax.set_title("Block 大小对 Naive 卷积性能的影响")
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "block_size_impact.png", dpi=150)
    plt.close(fig)
    print("  -> block_size_impact.png")


# ============================================================
# Figure 7: Heatmap (kernel x size, stride=1)
# ============================================================
def plot_heatmap(data: list[dict]):
    stride1 = [r for r in data if r["stride"] == 1]
    sizes = sorted(set(r["H"] for r in stride1))
    kernels = [k for k in KERNEL_NAMES
               if any(r["kernel"] == k for r in stride1)]

    heatmap = np.zeros((len(kernels), len(sizes)))
    for i, kn in enumerate(kernels):
        for j, sz in enumerate(sizes):
            kd = [r for r in stride1 if r["H"] == sz and r["kernel"] == kn]
            heatmap[i, j] = (min(r["avg_kernel_time_ms"] for r in kd)
                             if kd else np.nan)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(heatmap, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_yticks(range(len(kernels)))
    ax.set_yticklabels(kernels)
    ax.set_xlabel("输入规模 (H=W)")
    ax.set_ylabel("Kernel")
    ax.set_title("运行时间热力图 (ms, stride=1)")

    for i in range(len(kernels)):
        for j in range(len(sizes)):
            val = heatmap[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7)

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
    plot_stride_impact(data)
    plot_time_breakdown(data)
    plot_memory_usage(data)
    plot_speedup(data)
    plot_block_size_impact(data)
    plot_heatmap(data)

    print(f"Done. Figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
