#!/usr/bin/env python3
import os
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).parent.absolute()
LAB_DIR = SCRIPT_DIR.parent


def get_results_dir() -> Path:
    configured = os.environ.get("LAB2_RESULTS_DIR")
    if not configured:
        return LAB_DIR / "results"
    return Path(configured).expanduser().resolve()


def main() -> None:
    results_dir = get_results_dir()
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(results_dir / "summary_by_version.csv")
    ok = df[df["status"] == "ok"].copy()
    baselines = ok[ok["processes"] == 1][["version", "matrix_size", "avg_time_sec"]].rename(columns={"avg_time_sec": "baseline_time"})
    ok = ok.merge(baselines, on=["version", "matrix_size"], how="left")
    ok["speedup"] = ok["baseline_time"] / ok["avg_time_sec"]
    ok["efficiency"] = ok["speedup"] / ok["processes"]
    ok.pivot_table(index=["version", "processes"], columns="matrix_size", values="avg_time_sec").to_csv(tables_dir / "time_table.csv")
    ok.pivot_table(index=["version", "processes"], columns="matrix_size", values="speedup").to_csv(tables_dir / "speedup_table.csv")
    ok.pivot_table(index=["version", "processes"], columns="matrix_size", values="efficiency").to_csv(tables_dir / "efficiency_table.csv")


if __name__ == "__main__":
    main()
