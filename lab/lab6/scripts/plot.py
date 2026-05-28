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
    configured = os.environ.get("LAB6_RESULTS_DIR")
    if not configured:
        return LAB_DIR / "results"
    return Path(configured).expanduser().resolve()


def add_speedup_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["threads"] = df["threads"].astype(int)
    df["avg_time_sec"] = df["avg_time_sec"].astype(float)
    df["m"] = df["m"].astype(int)
    df["n"] = df["n"].astype(int)
    baselines = df[df["threads"] == 1][["version", "m", "n", "avg_time_sec"]].rename(columns={"avg_time_sec": "baseline_time"})
    df = df.merge(baselines, on=["version", "m", "n"], how="left")
    df["speedup"] = df["baseline_time"] / df["avg_time_sec"]
    df["efficiency"] = df["speedup"] / df["threads"]
    df["grid"] = df["m"].astype(str) + "x" + df["n"].astype(str)
    return df


def save_metric_lines(df: pd.DataFrame, metric: str, title_prefix: str, output_prefix: str, figures_dir: Path) -> None:
    for version in sorted(df["version"].unique()):
        subset = df[df["version"] == version].sort_values(["m", "n"])
        plt.figure(figsize=(8, 5))
        for threads in sorted(subset["threads"].unique()):
            part = subset[subset["threads"] == threads].sort_values(["m", "n"])
            plt.plot(part["grid"], part[metric], marker="o", label=f"threads={threads}")
        plt.title(f"{title_prefix} - {version}")
        plt.xlabel("Grid size")
        plt.ylabel(metric.replace("_", " ").title())
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / f"{output_prefix}_{version}.png", dpi=200)
        plt.close()


def save_heatmaps(df: pd.DataFrame, figures_dir: Path) -> None:
    for version in sorted(df["version"].unique()):
        subset = df[df["version"] == version]
        pivot = subset.pivot_table(index="threads", columns="grid", values="avg_time_sec", aggfunc="first")
        plt.figure(figsize=(8, 5))
        plt.imshow(pivot.values, aspect="auto")
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.colorbar(label="Avg time (s)")
        plt.title(f"Runtime heatmap - {version}")
        plt.xlabel("Grid size")
        plt.ylabel("Threads")
        plt.tight_layout()
        plt.savefig(figures_dir / f"heatmap_{version}.png", dpi=200)
        plt.close()


def save_schedule_comparison(df: pd.DataFrame, figures_dir: Path) -> None:
    subset = df[df["backend"] == "pthreads_parallel_for"].copy()
    if subset.empty:
        return
    target_threads = 4 if 4 in set(subset["threads"]) else int(subset["threads"].max())
    subset = subset[subset["threads"] == target_threads]
    plt.figure(figsize=(8, 5))
    for version in sorted(subset["version"].unique()):
        part = subset[subset["version"] == version].sort_values(["m", "n"])
        plt.plot(part["grid"], part["avg_time_sec"], marker="o", label=version)
    plt.title(f"Pthreads parallel_for schedule comparison at threads={target_threads}")
    plt.xlabel("Grid size")
    plt.ylabel("Avg time (s)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "pthreads_schedule_comparison.png", dpi=200)
    plt.close()


def save_openmp_comparison(df: pd.DataFrame, figures_dir: Path) -> None:
    if df.empty:
        return
    target_threads = 4 if 4 in set(df["threads"]) else int(df["threads"].max())
    subset = df[df["threads"] == target_threads]
    pivot = subset.pivot(index="grid", columns="version", values="avg_time_sec")
    pivot.plot(kind="bar", figsize=(9, 5))
    plt.title(f"OpenMP original vs Pthreads parallel_for at threads={target_threads}")
    plt.xlabel("Grid size")
    plt.ylabel("Avg time (s)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "openmp_vs_pthreads.png", dpi=200)
    plt.close()


def main() -> None:
    results_dir = get_results_dir()
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_dir / "summary_by_version.csv")
    df = df[df["status"] == "ok"].copy()
    df = add_speedup_efficiency(df)

    save_metric_lines(df, "avg_time_sec", "Runtime", "runtime", figures_dir)
    save_metric_lines(df, "speedup", "Speedup", "speedup", figures_dir)
    save_metric_lines(df, "efficiency", "Efficiency", "efficiency", figures_dir)
    save_heatmaps(df, figures_dir)
    save_schedule_comparison(df, figures_dir)
    save_openmp_comparison(df, figures_dir)


if __name__ == "__main__":
    main()
