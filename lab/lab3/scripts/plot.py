#!/usr/bin/env python3
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).parent.absolute()
LAB_DIR = SCRIPT_DIR.parent


def get_results_dir() -> Path:
    configured = os.environ.get("LAB3_RESULTS_DIR") or os.environ.get("LAB2_RESULTS_DIR")
    if not configured:
        return LAB_DIR / "results"
    return Path(configured).expanduser().resolve()


def add_speedup_efficiency(df: pd.DataFrame, size_col: str) -> pd.DataFrame:
    df = df.copy()
    df["threads"] = df["threads"].astype(int)
    df["time_sec"] = df["time_sec"].astype(float)
    df[size_col] = df[size_col].astype(int)
    baselines = df[df["threads"] == 1][["version", size_col, "time_sec"]].rename(columns={"time_sec": "baseline_time"})
    df = df.merge(baselines, on=["version", size_col], how="left")
    df["speedup"] = df["baseline_time"] / df["time_sec"]
    df["efficiency"] = df["speedup"] / df["threads"]
    return df


def save_metric_lines(df: pd.DataFrame, size_col: str, metric: str, title_prefix: str, output_prefix: str, figures_dir: Path) -> None:
    for version in sorted(df["version"].unique()):
        subset = df[df["version"] == version].sort_values(size_col)
        plt.figure(figsize=(8, 5))
        for threads in sorted(subset["threads"].unique()):
            part = subset[subset["threads"] == threads].sort_values(size_col)
            plt.plot(part[size_col], part[metric], marker="o", label=f"threads={threads}")
        plt.title(f"{title_prefix} - {version}")
        plt.xlabel("Matrix size" if size_col == "m" else "Array length")
        plt.ylabel(metric.replace("_", " ").title())
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / f"{output_prefix}_{version}.png", dpi=200)
        plt.close()


def save_heatmaps(df: pd.DataFrame, figures_dir: Path) -> None:
    for version in sorted(df["version"].unique()):
        subset = df[df["version"] == version].copy()
        pivot = subset.pivot_table(index="threads", columns="m", values="time_sec", aggfunc="first")
        plt.figure(figsize=(8, 5))
        plt.imshow(pivot.values, aspect="auto")
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.colorbar(label="Avg time (s)")
        plt.title(f"Matmul heatmap - {version}")
        plt.xlabel("Matrix size")
        plt.ylabel("Threads")
        plt.tight_layout()
        plt.savefig(figures_dir / f"heatmap_matmul_{version}.png", dpi=200)
        plt.close()


def save_version_comparison(df: pd.DataFrame, size_col: str, output_name: str, title: str, figures_dir: Path) -> None:
    available = sorted(df[size_col].unique())
    if not available:
        return
    target_size = available[min(len(available) - 1, 1)] if len(available) > 1 else available[0]
    subset = df[df[size_col] == target_size].pivot(index="threads", columns="version", values="time_sec")
    subset.plot(kind="bar", figsize=(8, 5))
    plt.title(title.format(size=target_size))
    plt.xlabel("Threads")
    plt.ylabel("Avg time (s)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / output_name, dpi=200)
    plt.close()


def main() -> None:
    results_dir = get_results_dir()
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_dir / "summary_by_version.csv")
    df = df[df["status"] == "ok"].copy()

    matmul = df[df["experiment"] == "pthreads_matmul"].copy()
    if not matmul.empty:
        matmul = matmul[matmul["m"].notna()]
        matmul = add_speedup_efficiency(matmul, "m")
        save_metric_lines(matmul, "m", "time_sec", "Matmul runtime", "runtime_matmul", figures_dir)
        save_metric_lines(matmul, "m", "speedup", "Matmul speedup", "speedup_matmul", figures_dir)
        save_metric_lines(matmul, "m", "efficiency", "Matmul efficiency", "efficiency_matmul", figures_dir)
        save_heatmaps(matmul, figures_dir)
        save_version_comparison(matmul, "m", "version_comparison_matmul.png", "Matmul version comparison at size {size}", figures_dir)

    array_sum = df[df["experiment"] == "pthreads_array_sum"].copy()
    if not array_sum.empty:
        array_sum = array_sum[array_sum["length"].notna()]
        array_sum = add_speedup_efficiency(array_sum, "length")
        save_metric_lines(array_sum, "length", "time_sec", "Array sum runtime", "runtime_array_sum", figures_dir)
        save_metric_lines(array_sum, "length", "speedup", "Array sum speedup", "speedup_array_sum", figures_dir)
        save_metric_lines(array_sum, "length", "efficiency", "Array sum efficiency", "efficiency_array_sum", figures_dir)
        save_version_comparison(array_sum, "length", "version_comparison_array_sum.png", "Array sum version comparison at length {size}", figures_dir)


if __name__ == "__main__":
    main()
