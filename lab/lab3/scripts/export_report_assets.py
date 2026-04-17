#!/usr/bin/env python3
import os
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent.absolute()
LAB_DIR = SCRIPT_DIR.parent


def get_results_dir() -> Path:
    configured = os.environ.get("LAB3_RESULTS_DIR") or os.environ.get("LAB2_RESULTS_DIR")
    if not configured:
        return LAB_DIR / "results"
    return Path(configured).expanduser().resolve()


def export_tables(df: pd.DataFrame, experiment: str, size_col: str, prefix: str, tables_dir: Path) -> None:
    subset = df[(df["experiment"] == experiment) & (df["status"] == "ok")].copy()
    subset[size_col] = subset[size_col].astype(int)
    subset["threads"] = subset["threads"].astype(int)
    subset["time_sec"] = subset["time_sec"].astype(float)

    baselines = subset[subset["threads"] == 1][["version", size_col, "time_sec"]].rename(columns={"time_sec": "baseline_time"})
    subset = subset.merge(baselines, on=["version", size_col], how="left")
    subset["speedup"] = subset["baseline_time"] / subset["time_sec"]
    subset["efficiency"] = subset["speedup"] / subset["threads"]

    subset.pivot_table(index=["version", "threads"], columns=size_col, values="time_sec").to_csv(tables_dir / f"time_table_{prefix}.csv")
    subset.pivot_table(index=["version", "threads"], columns=size_col, values="speedup").to_csv(tables_dir / f"speedup_table_{prefix}.csv")
    subset.pivot_table(index=["version", "threads"], columns=size_col, values="efficiency").to_csv(tables_dir / f"efficiency_table_{prefix}.csv")


def main() -> None:
    results_dir = get_results_dir()
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_dir / "summary_by_version.csv")
    export_tables(df, "pthreads_matmul", "m", "matmul", tables_dir)
    export_tables(df, "pthreads_array_sum", "length", "array_sum", tables_dir)


if __name__ == "__main__":
    main()
