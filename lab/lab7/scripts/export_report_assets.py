#!/usr/bin/env python3
import os
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent.absolute()
LAB_DIR = SCRIPT_DIR.parent


def get_results_dir():
    configured = os.environ.get("LAB7_RESULTS_DIR")
    return Path(configured).expanduser().resolve() if configured else (LAB_DIR / "results")


def add_speedup_fft(df):
    df = df.copy()
    serial = df[df["backend"] == "serial"][["n", "avg_time_sec"]].rename(
        columns={"avg_time_sec": "serial_time"})
    df = df.merge(serial, on="n", how="left")
    df["speedup"] = df["serial_time"] / df["avg_time_sec"]
    df["efficiency"] = df["speedup"] / df["num_procs"]
    return df


def main():
    results_dir = get_results_dir()
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # FFT tables
    csv_path = results_dir / "summary.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = df[df["status"] == "ok"].copy()
        df = add_speedup_fft(df)

        time_pivot = df.pivot_table(index=["backend", "num_procs"],
                                    columns="n", values="avg_time_sec", aggfunc="first")
        time_pivot.to_csv(tables_dir / "fft_time_table.csv")

        speedup_pivot = df.pivot_table(index=["backend", "num_procs"],
                                       columns="n", values="speedup", aggfunc="first")
        speedup_pivot.to_csv(tables_dir / "fft_speedup_table.csv")

        mflops_pivot = df.pivot_table(index=["backend", "num_procs"],
                                      columns="n", values="avg_mflops", aggfunc="first")
        mflops_pivot.to_csv(tables_dir / "fft_mflops_table.csv")

    # Heated plate tables
    hp_csv = results_dir / "heated_plate_extended.csv"
    if hp_csv.exists():
        hp_df = pd.read_csv(hp_csv)
        hp_df = hp_df[hp_df["status"] == "ok"].copy()
        hp_df["grid"] = hp_df["m"].astype(str) + "x" + hp_df["n"].astype(str)

        for version in hp_df["version"].unique():
            subset = hp_df[hp_df["version"] == version]
            pivot = subset.pivot_table(index="threads", columns="grid",
                                       values="avg_time_sec", aggfunc="first")
            pivot.to_csv(tables_dir / f"heated_plate_time_{version}.csv")

        hp_df.to_csv(tables_dir / "heated_plate_all.csv", index=False)

    print(f"Report assets exported to {tables_dir}")


if __name__ == "__main__":
    main()
