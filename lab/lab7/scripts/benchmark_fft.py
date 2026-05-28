#!/usr/bin/env python3
import csv
import json
import math
import os
import re
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
LAB_DIR = SCRIPT_DIR.parent

FFT_N_VALUES = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
FFT_P_VALUES = [1, 2, 4, 8]
NUM_RUNS = 3
SEED = 331.0
TIMEOUT_SEC = 300

REQUIRED_KEYS = ("experiment", "backend", "n", "nits", "error",
                 "time_sec", "mflops", "checksum", "num_procs")


def get_results_dir():
    configured = os.environ.get("LAB7_RESULTS_DIR")
    return Path(configured).expanduser().resolve() if configured else (LAB_DIR / "results")


def parse_key_value_output(output):
    data = {}
    for line in output.strip().splitlines():
        for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", line):
            data[key] = value
    return data


def run_serial_case(n):
    cmd = ["./bin/fft_serial", str(n), str(SEED)]
    try:
        result = subprocess.run(cmd, cwd=LAB_DIR, text=True,
                                capture_output=True, timeout=TIMEOUT_SEC, check=False)
    except subprocess.TimeoutExpired as exc:
        return {"success": False, "command": cmd, "error": "timeout",
                "stdout": exc.stdout or "", "stderr": exc.stderr or "", "parsed": {}}

    parsed = parse_key_value_output(result.stdout)
    missing = [k for k in REQUIRED_KEYS if k not in parsed]
    success = result.returncode == 0 and not missing
    return {
        "success": success,
        "command": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "parsed": parsed,
        "error": "" if success else f"missing: {missing}"
    }


def run_mpi_case(n, np_val):
    cmd = ["mpirun", "-np", str(np_val), "--oversubscribe",
           "./bin/fft_mpi", str(n), str(SEED)]
    try:
        result = subprocess.run(cmd, cwd=LAB_DIR, text=True,
                                capture_output=True, timeout=TIMEOUT_SEC, check=False)
    except subprocess.TimeoutExpired as exc:
        return {"success": False, "command": cmd, "error": "timeout",
                "stdout": exc.stdout or "", "stderr": exc.stderr or "", "parsed": {}}

    parsed = parse_key_value_output(result.stdout)
    missing = [k for k in REQUIRED_KEYS if k not in parsed]
    success = result.returncode == 0 and not missing
    return {
        "success": success,
        "command": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "parsed": parsed,
        "error": "" if success else f"missing: {missing}"
    }


def can_run_mpi(n, np_val):
    """Check if N/P >= P (needed for 6-step FFT)"""
    local_n = n // np_val
    return n % np_val == 0 and local_n >= np_val and (local_n & (local_n - 1)) == 0


def summarize_runs(n, np_val, backend, runs):
    successful = [r for r in runs if r["success"]]
    failed = [r for r in runs if not r["success"]]

    if not successful:
        return {
            "experiment": "fft", "backend": backend, "n": n,
            "num_procs": np_val, "status": "failed",
            "avg_time_sec": None, "avg_error": None,
            "avg_mflops": None, "checksum": None,
            "requested_runs": len(runs),
            "successful_runs": 0, "failed_runs": len(failed),
        }

    times = [float(r["parsed"]["time_sec"]) for r in successful]
    errors = [float(r["parsed"]["error"]) for r in successful]
    mflops_vals = [float(r["parsed"]["mflops"]) for r in successful]
    checksums = [float(r["parsed"]["checksum"]) for r in successful]

    return {
        "experiment": "fft",
        "backend": backend,
        "n": n,
        "num_procs": np_val,
        "status": "ok",
        "avg_time_sec": sum(times) / len(times),
        "avg_error": sum(errors) / len(errors),
        "avg_mflops": sum(mflops_vals) / len(mflops_vals),
        "checksum": checksums[0],
        "requested_runs": len(runs),
        "successful_runs": len(successful),
        "failed_runs": len(failed),
        "runs": runs,
    }


def main():
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Serial FFT
    for n in FFT_N_VALUES:
        runs = [run_serial_case(n) for _ in range(NUM_RUNS)]
        results.append(summarize_runs(n, 1, "serial", runs))
        status = results[-1]["status"]
        t = results[-1]["avg_time_sec"]
        print(f"serial N={n}: status={status}, time={t}")

    # MPI FFT
    for n in FFT_N_VALUES:
        for np_val in FFT_P_VALUES:
            if np_val == 1:
                continue  # already covered by serial
            if not can_run_mpi(n, np_val):
                print(f"mpi N={n} P={np_val}: skipped (N/P < P)")
                continue
            runs = [run_mpi_case(n, np_val) for _ in range(NUM_RUNS)]
            results.append(summarize_runs(n, np_val, "mpi", runs))
            status = results[-1]["status"]
            t = results[-1]["avg_time_sec"]
            print(f"mpi N={n} P={np_val}: status={status}, time={t}")

    # Write results
    (results_dir / "benchmark_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # CSV summary
    csv_path = results_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "backend", "n", "num_procs", "status",
                         "avg_time_sec", "avg_error", "avg_mflops", "checksum"])
        for r in results:
            writer.writerow([r["experiment"], r["backend"], r["n"], r["num_procs"],
                             r["status"],
                             f"{r['avg_time_sec']:.9f}" if r["avg_time_sec"] else "",
                             f"{r['avg_error']:.9e}" if r["avg_error"] else "",
                             f"{r['avg_mflops']:.3f}" if r["avg_mflops"] else "",
                             f"{r['checksum']:.6f}" if r["checksum"] else ""])

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
