#!/usr/bin/env python3
"""Generate plots for lab9 CUDA transpose benchmark results."""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent.absolute()
LAB_DIR = SCRIPT_DIR.parent


def get_results_dir():
    configured = os.environ.get("LAB9_RESULTS_DIR")
    return Path(configured) if configured else (LAB_DIR / "results")


def plot_runtime_by_size(df, figures_dir):
    """Kernel time vs matrix size, grouped by kernel type."""
    df_ok = df[df["status"] == "ok"].copy()
    # Use best block size per kernel per N
    idx = df_ok.groupby(["N", "kernel"])["avg_kernel_time_ms"].idxmin()
    best = df_ok.loc[idx].sort_values(["kernel", "N"])

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"naive": "#E74C3C", "coalesced_read": "#F39C12", "shared_mem": "#27AE60"}
    markers = {"naive": "o", "coalesced_read": "s", "shared_mem": "D"}

    for kernel in ["naive", "coalesced_read", "shared_mem"]:
        subset = best[best["kernel"] == kernel]
        if subset.empty:
            continue
        ax.plot(subset["N"], subset["avg_kernel_time_ms"],
                f"{markers.get(kernel, 'o')}-", label=kernel,
                color=colors.get(kernel, "gray"), markersize=10, linewidth=2.5)

    ax.set_xlabel("Matrix Size N", fontsize=12)
    ax.set_ylabel("Kernel Time (ms)", fontsize=12)
    ax.set_title("CUDA Matrix Transpose: Kernel Time vs Matrix Size", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(figures_dir / "runtime_vs_size.png", dpi=200)
    plt.close(fig)


def plot_bandwidth(df, figures_dir):
    """Effective bandwidth vs matrix size."""
    df_ok = df[df["status"] == "ok"].copy()

    # Use best block size per kernel per N
    idx = df_ok.groupby(["N", "kernel"])["avg_kernel_time_ms"].idxmin()
    best = df_ok.loc[idx].sort_values(["kernel", "N"])

    # Compute bandwidth if not present
    if "bandwidth_gb_s" not in best.columns:
        best["bandwidth_gb_s"] = best.apply(
            lambda r: (2.0 * r["N"]**2 * 4) / (r["avg_kernel_time_ms"] / 1000.0) / 1e9,
            axis=1,
        )
    else:
        best["bandwidth_gb_s"] = best["bandwidth_gb_s"].astype(float)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"naive": "#E74C3C", "coalesced_read": "#F39C12", "shared_mem": "#27AE60"}

    for kernel in ["naive", "coalesced_read", "shared_mem"]:
        subset = best[best["kernel"] == kernel]
        if subset.empty:
            continue
        ax.bar(subset["N"].astype(str) + f"\n{kernel}",
               subset["bandwidth_gb_s"], color=colors.get(kernel, "gray"),
               alpha=0.85, label=kernel)

    ax.set_ylabel("Effective Bandwidth (GB/s)", fontsize=12)
    ax.set_title("CUDA Matrix Transpose: Effective Bandwidth", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(figures_dir / "bandwidth.png", dpi=200)
    plt.close(fig)


def plot_block_size_impact(df, figures_dir):
    """Block size impact on performance (grouped bar for N=1024)."""
    df_ok = df[(df["status"] == "ok") & (df["N"] == 1024)].copy()
    # Only naive and coalesced_read (shared_mem uses fixed TILE_DIM)
    df_ok = df_ok[df_ok["kernel"] != "shared_mem"]

    if df_ok.empty:
        print("  Skipping block size impact: no data for N=1024")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    block_labels = sorted(df_ok["block_x"].astype(str).unique())
    kernels = ["naive", "coalesced_read"]
    colors = {"naive": "#E74C3C", "coalesced_read": "#F39C12"}

    x = np.arange(len(block_labels))
    width = 0.35

    for i, kernel in enumerate(kernels):
        subset = df_ok[df_ok["kernel"] == kernel].sort_values("block_x")
        values = []
        for bl in block_labels:
            match = subset[subset["block_x"].astype(str) == bl]
            values.append(match["avg_kernel_time_ms"].values[0] if not match.empty else 0)
        ax.bar(x + i * width, values, width, label=kernel,
               color=colors.get(kernel, "gray"), alpha=0.85)

    ax.set_xlabel("Block Size", fontsize=12)
    ax.set_ylabel("Kernel Time (ms)", fontsize=12)
    ax.set_title("Block Size Impact on Performance (N=1024)", fontsize=14)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([f"{b}×{b}" for b in block_labels])
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(figures_dir / "block_size_impact.png", dpi=200)
    plt.close(fig)


def plot_speedup(df, figures_dir):
    """Speedup of coalesced_read and shared_mem relative to naive."""
    df_ok = df[df["status"] == "ok"].copy()

    # Get best time per kernel per N
    idx = df_ok.groupby(["N", "kernel"])["avg_kernel_time_ms"].idxmin()
    best = df_ok.loc[idx].copy()

    # Get naive baseline per N
    naive_times = {}
    for n in best["N"].unique():
        naive_row = best[(best["N"] == n) & (best["kernel"] == "naive")]
        if not naive_row.empty:
            naive_times[n] = naive_row.iloc[0]["avg_kernel_time_ms"]

    # Compute speedup
    speedup_data = []
    for _, row in best.iterrows():
        n = row["N"]
        if n in naive_times and naive_times[n] > 0:
            su = naive_times[n] / row["avg_kernel_time_ms"]
            speedup_data.append({
                "N": n, "kernel": row["kernel"], "speedup": su,
            })

    if not speedup_data:
        print("  Skipping speedup: no baseline data")
        return

    su_df = pd.DataFrame(speedup_data)
    su_df = su_df[su_df["kernel"] != "naive"]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"coalesced_read": "#F39C12", "shared_mem": "#27AE60"}

    for kernel in ["coalesced_read", "shared_mem"]:
        subset = su_df[su_df["kernel"] == kernel].sort_values("N")
        if subset.empty:
            continue
        ax.plot(subset["N"], subset["speedup"], "o-", label=kernel,
                color=colors.get(kernel, "gray"), markersize=10, linewidth=2.5)

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Matrix Size N", fontsize=12)
    ax.set_ylabel("Speedup vs Naive", fontsize=12)
    ax.set_title("CUDA Matrix Transpose: Speedup over Naive Implementation", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(figures_dir / "speedup.png", dpi=200)
    plt.close(fig)


def plot_heatmap(df, figures_dir):
    """Heatmap: kernel_time by (N × kernel) with best block size."""
    df_ok = df[df["status"] == "ok"].copy()
    idx = df_ok.groupby(["N", "kernel"])["avg_kernel_time_ms"].idxmin()
    best = df_ok.loc[idx]

    pivot = best.pivot_table(index="N", columns="kernel",
                             values="avg_kernel_time_ms", aggfunc="first")
    # Order columns
    col_order = ["naive", "coalesced_read", "shared_mem"]
    pivot = pivot[[c for c in col_order if c in pivot.columns]]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd_r")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=11)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_color = "white" if val > pivot.values.mean() else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=10, color=text_color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Kernel Time (ms)")
    ax.set_title("Transpose Kernel Time Heatmap (N × Kernel)", fontsize=14)
    ax.set_xlabel("Kernel", fontsize=12)
    ax.set_ylabel("Matrix Size N", fontsize=12)
    fig.tight_layout()
    fig.savefig(figures_dir / "heatmap.png", dpi=200)
    plt.close(fig)


def main():
    results_dir = get_results_dir()
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "summary.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run benchmark.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")

    plot_runtime_by_size(df, figures_dir)
    print("  runtime_vs_size.png")

    plot_bandwidth(df, figures_dir)
    print("  bandwidth.png")

    plot_block_size_impact(df, figures_dir)
    print("  block_size_impact.png")

    plot_speedup(df, figures_dir)
    print("  speedup.png")

    plot_heatmap(df, figures_dir)
    print("  heatmap.png")

    print(f"\nAll figures saved to {figures_dir}")


if __name__ == "__main__":
    main()
