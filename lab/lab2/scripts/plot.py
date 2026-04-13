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
    configured = os.environ.get("LAB2_RESULTS_DIR")
    if not configured:
        return LAB_DIR / "results"
    return Path(configured).expanduser().resolve()


def load_summary(summary_csv: Path) -> pd.DataFrame:
    return pd.read_csv(summary_csv)


def save_runtime_lines(df: pd.DataFrame, figures_dir: Path) -> None:
    for version in sorted(df["version"].unique()):
        subset = df[(df["version"] == version) & (df["status"] == "ok")]
        plt.figure(figsize=(8, 5))
        for procs in sorted(subset["processes"].unique()):
            part = subset[subset["processes"] == procs].sort_values("matrix_size")
            plt.plot(part["matrix_size"], part["avg_time_sec"], marker="o", label=f"p={procs}")
        plt.title(f"Runtime - {version}")
        plt.xlabel("Matrix Size")
        plt.ylabel("Average Time (s)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / f"runtime_{version}.png", dpi=200)
        plt.close()


def save_speedup_lines(df: pd.DataFrame, figures_dir: Path) -> None:
    for version in sorted(df["version"].unique()):
        subset = df[(df["version"] == version) & (df["status"] == "ok")].copy()
        baselines = subset[subset["processes"] == 1][["matrix_size", "avg_time_sec"]].rename(columns={"avg_time_sec": "baseline_time"})
        merged = subset.merge(baselines, on="matrix_size", how="left")
        merged["speedup"] = merged["baseline_time"] / merged["avg_time_sec"]
        plt.figure(figsize=(8, 5))
        for procs in sorted(merged["processes"].unique()):
            part = merged[merged["processes"] == procs].sort_values("matrix_size")
            plt.plot(part["matrix_size"], part["speedup"], marker="o", label=f"p={procs}")
        plt.title(f"Speedup - {version}")
        plt.xlabel("Matrix Size")
        plt.ylabel("Speedup")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / f"speedup_{version}.png", dpi=200)
        plt.close()


def save_efficiency_lines(df: pd.DataFrame, figures_dir: Path) -> None:
    for version in sorted(df["version"].unique()):
        subset = df[(df["version"] == version) & (df["status"] == "ok")].copy()
        baselines = subset[subset["processes"] == 1][["matrix_size", "avg_time_sec"]].rename(columns={"avg_time_sec": "baseline_time"})
        merged = subset.merge(baselines, on="matrix_size", how="left")
        merged["efficiency"] = (merged["baseline_time"] / merged["avg_time_sec"]) / merged["processes"]
        plt.figure(figsize=(8, 5))
        for procs in sorted(merged["processes"].unique()):
            part = merged[merged["processes"] == procs].sort_values("matrix_size")
            plt.plot(part["matrix_size"], part["efficiency"], marker="o", label=f"p={procs}")
        plt.title(f"Efficiency - {version}")
        plt.xlabel("Matrix Size")
        plt.ylabel("Efficiency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / f"efficiency_{version}.png", dpi=200)
        plt.close()


def save_version_comparison(df: pd.DataFrame, figures_dir: Path) -> None:
    ok_rows = df[df["status"] == "ok"]
    if ok_rows.empty:
        return

    available_sizes = [size for size in [512, 1024, 2048] if size in set(ok_rows["matrix_size"])]
    if not available_sizes:
        available_sizes = sorted(ok_rows["matrix_size"].unique())[:3]

    available_processes = sorted(ok_rows["processes"].unique())
    if not available_processes:
        return

    target_procs = 4 if 4 in available_processes else available_processes[-1]
    subset = ok_rows[(ok_rows["matrix_size"].isin(available_sizes)) & (ok_rows["processes"] == target_procs)]
    if subset.empty:
        return
    pivot = subset.pivot(index="matrix_size", columns="version", values="avg_time_sec")
    pivot.plot(kind="bar", figsize=(8, 5))
    plt.title(f"Version Comparison at {target_procs} Processes")
    plt.xlabel("Matrix Size")
    plt.ylabel("Average Time (s)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "version_comparison.png", dpi=200)
    plt.close()


def save_heatmaps(df: pd.DataFrame, figures_dir: Path) -> None:
    for version in sorted(df["version"].unique()):
        subset = df[(df["version"] == version) & (df["status"] == "ok")]
        if subset.empty:
            continue
        pivot = subset.pivot(index="processes", columns="matrix_size", values="avg_time_sec")
        plt.figure(figsize=(8, 5))
        plt.imshow(pivot.values, aspect="auto")
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.title(f"Runtime Heatmap - {version}")
        plt.xlabel("Matrix Size")
        plt.ylabel("Processes")
        plt.colorbar(label="Average Time (s)")
        plt.tight_layout()
        plt.savefig(figures_dir / f"heatmap_{version}.png", dpi=200)
        plt.close()


def main() -> None:
    results_dir = get_results_dir()
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    df = load_summary(results_dir / "summary_by_version.csv")
    save_runtime_lines(df, figures_dir)
    save_speedup_lines(df, figures_dir)
    save_efficiency_lines(df, figures_dir)
    save_version_comparison(df, figures_dir)
    save_heatmaps(df, figures_dir)


if __name__ == "__main__":
    main()
