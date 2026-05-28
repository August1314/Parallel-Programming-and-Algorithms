#!/usr/bin/env python3
"""Generate plots for lab8 APSP benchmark results."""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).parent.absolute()
LAB_DIR = SCRIPT_DIR.parent


def get_results_dir():
    configured = os.environ.get("LAB8_RESULTS_DIR")
    return Path(configured) if configured else (LAB_DIR / "results")


def plot_runtime(df, figures_dir):
    """Runtime vs threads for each dataset."""
    fig, ax = plt.subplots(figsize=(10, 6))
    df_ok = df[df["status"] == "ok"].copy()

    colors = {"flower": "#E74C3C", "mouse": "#3498DB"}
    markers = {"serial": "o", "openmp": "s"}

    for dataset in df_ok["dataset"].unique():
        subset = df_ok[df_ok["dataset"] == dataset].sort_values("num_threads")
        for backend in subset["backend"].unique():
            part = subset[subset["backend"] == backend]
            if not part.empty:
                label = f"{dataset}-{backend}"
                ax.plot(part["num_threads"], part["avg_time_sec"],
                        f"{markers.get(backend, 'o')}-", label=label,
                        color=colors.get(dataset, "gray"),
                        markersize=8, linewidth=2)

    ax.set_xlabel("Number of Threads")
    ax.set_ylabel("Avg Time (s)")
    ax.set_title("APSP Runtime vs Thread Count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "runtime.png", dpi=200)
    plt.close(fig)


def plot_speedup(df, figures_dir):
    """Speedup and efficiency vs threads."""
    df_ok = df[df["status"] == "ok"].copy()
    df_ok["num_threads"] = df_ok["num_threads"].astype(int)

    # Get serial baselines
    baselines = {}
    for dataset in df_ok["dataset"].unique():
        serial = df_ok[(df_ok["dataset"] == dataset) &
                       (df_ok["backend"] == "serial")]
        if not serial.empty:
            baselines[dataset] = serial.iloc[0]["avg_time_sec"]
        # Also check openmp with 1 thread
        omp1 = df_ok[(df_ok["dataset"] == dataset) &
                     (df_ok["backend"] == "openmp") &
                     (df_ok["num_threads"] == 1)]
        if not omp1.empty and dataset not in baselines:
            baselines[dataset] = omp1.iloc[0]["avg_time_sec"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"flower": "#E74C3C", "mouse": "#3498DB"}

    for dataset in sorted(df_ok["dataset"].unique()):
        if dataset not in baselines:
            continue
        baseline = baselines[dataset]
        subset = df_ok[(df_ok["dataset"] == dataset) &
                       (df_ok["backend"] == "openmp")].copy()
        subset["speedup"] = baseline / subset["avg_time_sec"]
        subset["efficiency"] = subset["speedup"] / subset["num_threads"]
        subset = subset.sort_values("num_threads")
        color = colors.get(dataset, "gray")

        ax1.plot(subset["num_threads"], subset["speedup"], "o-",
                 label=dataset, color=color, markersize=8, linewidth=2)
        ax2.plot(subset["num_threads"], subset["efficiency"], "s-",
                 label=dataset, color=color, markersize=8, linewidth=2)

    # Ideal speedup line
    max_t = df_ok["num_threads"].max()
    ax1.plot([1, max_t], [1, max_t], "k--", alpha=0.3, label="Ideal")
    ax1.set_xlabel("Number of Threads")
    ax1.set_ylabel("Speedup")
    ax1.set_title("APSP Speedup")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax2.set_xlabel("Number of Threads")
    ax2.set_ylabel("Efficiency")
    ax2.set_title("APSP Parallel Efficiency")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(figures_dir / "speedup_efficiency.png", dpi=200)
    plt.close(fig)


def plot_heatmap(df, figures_dir):
    """Heatmap of runtime by dataset x threads (OpenMP only)."""
    df_ok = df[(df["status"] == "ok") & (df["backend"] == "openmp")].copy()
    df_ok["num_threads"] = df_ok["num_threads"].astype(int)

    pivot = df_ok.pivot_table(index="num_threads", columns="dataset",
                               values="avg_time_sec", aggfunc="first")
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color="white" if val > pivot.values.mean() else "black")

    plt.colorbar(im, ax=ax, label="Avg time (s)")
    ax.set_title("APSP Runtime Heatmap (Dataset × Threads)")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Threads")
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

    plot_runtime(df, figures_dir)
    print("  runtime.png")

    plot_speedup(df, figures_dir)
    print("  speedup_efficiency.png")

    plot_heatmap(df, figures_dir)
    print("  heatmap.png")

    print(f"\nAll figures saved to {figures_dir}")


if __name__ == "__main__":
    main()
