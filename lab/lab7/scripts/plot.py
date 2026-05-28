#!/usr/bin/env python3
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
    configured = os.environ.get("LAB7_RESULTS_DIR")
    return Path(configured).expanduser().resolve() if configured else (LAB_DIR / "results")


def plot_fft_runtime(df, figures_dir):
    """FFT runtime vs N for serial and MPI with different process counts."""
    fig, ax = plt.subplots(figsize=(10, 6))
    df_ok = df[df["status"] == "ok"].copy()

    # Serial line
    serial = df_ok[df_ok["backend"] == "serial"]
    if not serial.empty:
        ax.plot(serial["n"], serial["avg_time_sec"], "o-", linewidth=2,
                markersize=8, label="Serial", color="black")

    # MPI with different process counts
    mpi = df_ok[df_ok["backend"] == "mpi"]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(mpi["num_procs"].unique())))
    for np_val, color in zip(sorted(mpi["num_procs"].unique()), colors):
        subset = mpi[mpi["num_procs"] == np_val]
        ax.plot(subset["n"], subset["avg_time_sec"], "s--", linewidth=1.5,
                markersize=7, label=f"MPI P={int(np_val)}", color=color)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N (问题规模)")
    ax.set_ylabel("Avg Time (s)")
    ax.set_title("FFT Runtime vs Problem Size")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "fft_runtime.png", dpi=200)
    plt.close(fig)


def plot_fft_speedup(df, figures_dir):
    """FFT speedup vs N."""
    df_ok = df[df["status"] == "ok"].copy()
    serial = df_ok[df_ok["backend"] == "serial"][["n", "avg_time_sec"]].rename(
        columns={"avg_time_sec": "serial_time"})
    mpi = df_ok[df_ok["backend"] == "mpi"].merge(serial, on="n")
    mpi["speedup"] = mpi["serial_time"] / mpi["avg_time_sec"]
    mpi["efficiency"] = mpi["speedup"] / mpi["num_procs"]

    # Speedup plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for np_val in sorted(mpi["num_procs"].unique()):
        subset = mpi[mpi["num_procs"] == np_val]
        axes[0].plot(subset["n"], subset["speedup"], "o-", label=f"P={int(np_val)}")
    axes[0].axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("N")
    axes[0].set_ylabel("Speedup")
    axes[0].set_title("FFT Speedup (vs Serial)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for np_val in sorted(mpi["num_procs"].unique()):
        subset = mpi[mpi["num_procs"] == np_val]
        axes[1].plot(subset["n"], subset["efficiency"], "s-", label=f"P={int(np_val)}")
    axes[1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("Efficiency")
    axes[1].set_title("FFT Efficiency")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(figures_dir / "fft_speedup_efficiency.png", dpi=200)
    plt.close(fig)


def plot_fft_mflops(df, figures_dir):
    """FFT MFLOPS comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    df_ok = df[df["status"] == "ok"].copy()

    serial = df_ok[df_ok["backend"] == "serial"]
    if not serial.empty:
        ax.plot(serial["n"], serial["avg_mflops"], "o-", linewidth=2,
                markersize=8, label="Serial", color="black")

    mpi = df_ok[df_ok["backend"] == "mpi"]
    for np_val in sorted(mpi["num_procs"].unique()):
        subset = mpi[mpi["num_procs"] == np_val]
        ax.plot(subset["n"], subset["avg_mflops"], "s--", linewidth=1.5,
                markersize=7, label=f"MPI P={int(np_val)}")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("N (问题规模)")
    ax.set_ylabel("MFLOPS")
    ax.set_title("FFT Performance (MFLOPS)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "fft_mflops.png", dpi=200)
    plt.close(fig)


def plot_fft_error(df, figures_dir):
    """FFT accuracy error vs N."""
    fig, ax = plt.subplots(figsize=(10, 5))
    df_ok = df[df["status"] == "ok"].copy()

    serial = df_ok[df_ok["backend"] == "serial"]
    if not serial.empty:
        ax.plot(serial["n"], serial["avg_error"], "o-", label="Serial", color="black")

    mpi = df_ok[df_ok["backend"] == "mpi"]
    for np_val in sorted(mpi["num_procs"].unique()):
        subset = mpi[mpi["num_procs"] == np_val]
        ax.plot(subset["n"], subset["avg_error"], "s--", label=f"MPI P={int(np_val)}")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N (问题规模)")
    ax.set_ylabel("Error")
    ax.set_title("FFT Accuracy: FFT(FFT(X)) - N*X")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "fft_accuracy.png", dpi=200)
    plt.close(fig)


def plot_heated_plate_extended(results_dir, figures_dir):
    """Plot extended heated_plate analysis from Part 2a."""
    csv_path = results_dir / "heated_plate_extended.csv"
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found, skipping heated_plate plots")
        return

    df = pd.read_csv(csv_path)
    df = df[df["status"] == "ok"].copy()
    if df.empty:
        return

    df["threads"] = df["threads"].astype(int)
    df["avg_time_sec"] = df["avg_time_sec"].astype(float)
    df["grid"] = df["m"].astype(str) + "x" + df["n"].astype(str)

    # Runtime vs threads per grid size
    for version in df["version"].unique():
        subset = df[df["version"] == version]
        fig, ax = plt.subplots(figsize=(9, 5))
        for grid in sorted(subset["grid"].unique(), key=lambda g: int(g.split("x")[0])):
            part = subset[subset["grid"] == grid].sort_values("threads")
            ax.plot(part["threads"], part["avg_time_sec"], "o-", label=grid)
        ax.set_xlabel("Threads")
        ax.set_ylabel("Avg Time (s)")
        ax.set_title(f"Heated Plate Runtime — {version}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / f"heated_plate_runtime_{version}.png", dpi=200)
        plt.close(fig)

    # Speedup
    for version in df["version"].unique():
        subset = df[df["version"] == version].copy()
        baselines = subset[subset["threads"] == 1][["grid", "avg_time_sec"]].rename(
            columns={"avg_time_sec": "t1"})
        subset = subset.merge(baselines, on="grid")
        subset["speedup"] = subset["t1"] / subset["avg_time_sec"]
        subset["efficiency"] = subset["speedup"] / subset["threads"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        for grid in sorted(subset["grid"].unique(), key=lambda g: int(g.split("x")[0])):
            part = subset[subset["grid"] == grid].sort_values("threads")
            ax1.plot(part["threads"], part["speedup"], "o-", label=grid)
            ax2.plot(part["threads"], part["efficiency"], "s-", label=grid)
        ax1.set_xlabel("Threads")
        ax1.set_ylabel("Speedup")
        ax1.set_title(f"Speedup — {version}")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax2.set_xlabel("Threads")
        ax2.set_ylabel("Efficiency")
        ax2.set_title(f"Efficiency — {version}")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / f"heated_plate_speedup_{version}.png", dpi=200)
        plt.close(fig)

    # Heatmap
    for version in df["version"].unique():
        subset = df[df["version"] == version]
        pivot = subset.pivot_table(index="threads", columns="grid",
                                   values="avg_time_sec", aggfunc="first")
        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        plt.colorbar(im, ax=ax, label="Avg time (s)")
        ax.set_title(f"Runtime Heatmap — {version}")
        ax.set_xlabel("Grid Size")
        ax.set_ylabel("Threads")
        fig.tight_layout()
        fig.savefig(figures_dir / f"heated_plate_heatmap_{version}.png", dpi=200)
        plt.close(fig)


def main():
    results_dir = get_results_dir()
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # FFT plots
    csv_path = results_dir / "summary.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        plot_fft_runtime(df, figures_dir)
        plot_fft_speedup(df, figures_dir)
        plot_fft_mflops(df, figures_dir)
        plot_fft_error(df, figures_dir)
        print(f"FFT plots saved to {figures_dir}")
    else:
        print(f"Warning: {csv_path} not found, skipping FFT plots")

    # Heated plate extended plots
    plot_heated_plate_extended(results_dir, figures_dir)
    print(f"Heated plate plots saved to {figures_dir}")


if __name__ == "__main__":
    main()
