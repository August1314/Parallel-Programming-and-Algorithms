#!/usr/bin/env python3
import os
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent.absolute()
LAB_DIR = SCRIPT_DIR.parent


def get_results_dir() -> Path:
    configured = os.environ.get("LAB5_RESULTS_DIR")
    if not configured:
        return LAB_DIR / "results"
    return Path(configured).expanduser().resolve()


def add_speedup_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["threads"] = df["threads"].astype(int)
    df["avg_time_sec"] = df["avg_time_sec"].astype(float)
    df["m"] = df["m"].astype(int)
    baselines = df[df["threads"] == 1][["version", "m", "avg_time_sec"]].rename(columns={"avg_time_sec": "baseline_time"})
    df = df.merge(baselines, on=["version", "m"], how="left")
    df["speedup"] = df["baseline_time"] / df["avg_time_sec"]
    df["efficiency"] = df["speedup"] / df["threads"]
    return df


def main() -> None:
    results_dir = get_results_dir()
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_dir / "summary_by_version.csv")
    df = df[df["status"] == "ok"].copy()
    df = add_speedup_efficiency(df)

    df.pivot_table(index=["backend", "version", "threads"], columns="m", values="avg_time_sec").to_csv(tables_dir / "time_table_matmul.csv")
    df.pivot_table(index=["backend", "version", "threads"], columns="m", values="speedup").to_csv(tables_dir / "speedup_table_matmul.csv")
    df.pivot_table(index=["backend", "version", "threads"], columns="m", values="efficiency").to_csv(tables_dir / "efficiency_table_matmul.csv")

    openmp = df[df["backend"] == "openmp"].pivot_table(index=["version", "threads"], columns="m", values="avg_time_sec")
    openmp.to_csv(tables_dir / "openmp_schedule_table.csv")


if __name__ == "__main__":
    main()
