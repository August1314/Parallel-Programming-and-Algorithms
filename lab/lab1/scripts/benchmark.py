#!/usr/bin/env python3
import json
import math
import re
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
LAB1_DIR = SCRIPT_DIR.parent
BIN_DIR = LAB1_DIR / "bin"
RESULTS_DIR = LAB1_DIR / "results"
EXECUTABLE = BIN_DIR / "mpi_matmul_v1"

PROCESS_COUNTS = [1, 2, 4, 8, 16]
MATRIX_SIZES = [128, 256, 512, 1024, 2048]
SEED = 20250401
NUM_RUNS = 3
TIMEOUT_SEC = 300
REQUIRED_KEYS = ("time_sec", "checksum", "max_abs", "processes", "m", "n", "k", "seed")


def parse_key_value_output(output: str) -> dict:
    data = {}
    for line in output.strip().splitlines():
        for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", line):
            data[key] = value
    return data


def benchmark_command(m: int, n: int, k: int, num_procs: int, seed: int) -> list[str]:
    return [
        "mpirun",
        "-np",
        str(num_procs),
        "--oversubscribe",
        str(EXECUTABLE),
        str(m),
        str(n),
        str(k),
        str(seed),
    ]


def run_benchmark(m: int, n: int, k: int, num_procs: int, seed: int = SEED) -> dict:
    cmd = benchmark_command(m, n, k, num_procs, seed)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SEC,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "success": False,
            "command": cmd,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "parsed": {},
            "error": f"timeout after {TIMEOUT_SEC} seconds",
        }
    except OSError as exc:
        return {
            "success": False,
            "command": cmd,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "parsed": {},
            "error": f"failed to execute command: {exc}",
        }

    parsed = parse_key_value_output(result.stdout)
    missing_keys = [key for key in REQUIRED_KEYS if key not in parsed]
    success = result.returncode == 0 and not missing_keys

    if not success:
        error_parts = []
        if result.returncode != 0:
            error_parts.append(f"return code {result.returncode}")
        if missing_keys:
            error_parts.append(f"missing keys: {', '.join(missing_keys)}")
        if result.stderr.strip():
            error_parts.append("stderr captured")
        error_message = "; ".join(error_parts) if error_parts else "unknown benchmark failure"
    else:
        error_message = ""

    return {
        "success": success,
        "command": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "parsed": parsed,
        "error": error_message,
    }


def aggregate_case_results(m: int, n: int, k: int, num_procs: int, run_records: list[dict]) -> dict:
    successful_runs = [record for record in run_records if record["success"]]
    failed_runs = [record for record in run_records if not record["success"]]

    times = [float(record["parsed"]["time_sec"]) for record in successful_runs]
    checksums = [float(record["parsed"]["checksum"]) for record in successful_runs]
    max_abs_values = [float(record["parsed"]["max_abs"]) for record in successful_runs]

    checksum_consistent = True
    if len(checksums) > 1:
        reference = checksums[0]
        checksum_consistent = all(math.isclose(value, reference, rel_tol=0.0, abs_tol=1e-9) for value in checksums[1:])

    if failed_runs:
        status = "partial_failure" if successful_runs else "failed"
    elif not checksum_consistent:
        status = "checksum_mismatch"
    else:
        status = "ok"

    return {
        "processes": num_procs,
        "matrix_size": m,
        "m": m,
        "n": n,
        "k": k,
        "requested_runs": len(run_records),
        "successful_runs": len(successful_runs),
        "failed_runs": len(failed_runs),
        "status": status,
        "avg_time_sec": (sum(times) / len(times)) if times else None,
        "min_time_sec": min(times) if times else None,
        "max_time_sec": max(times) if times else None,
        "all_times": times,
        "checksums": checksums,
        "max_abs_values": max_abs_values,
        "checksum_consistent": checksum_consistent,
        "baseline_checksum": None,
        "baseline_checksum_match": None,
        "runs": [
            {
                "run_index": index + 1,
                "success": record["success"],
                "returncode": record["returncode"],
                "error": record["error"],
                "stdout": record["stdout"],
                "stderr": record["stderr"],
                "parsed": record["parsed"],
                "command": record["command"],
            }
            for index, record in enumerate(run_records)
        ],
    }


def attach_baseline_consistency(results: list[dict]) -> None:
    baseline_checksums = {}
    for entry in results:
        if (
            entry["processes"] == 1
            and entry["successful_runs"] > 0
            and entry["checksum_consistent"]
        ):
            baseline_checksums[entry["matrix_size"]] = entry["checksums"][0]

    for entry in results:
        baseline_checksum = baseline_checksums.get(entry["matrix_size"])
        entry["baseline_checksum"] = baseline_checksum

        if baseline_checksum is None or not entry["checksums"]:
            entry["baseline_checksum_match"] = None
            continue

        entry["baseline_checksum_match"] = all(
            math.isclose(checksum, baseline_checksum, rel_tol=0.0, abs_tol=1e-9)
            for checksum in entry["checksums"]
        )

        if entry["status"] == "ok" and not entry["baseline_checksum_match"]:
            entry["status"] = "baseline_checksum_mismatch"


def format_metric(value: float | None, precision: int = 6) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}"


def main():
    if not EXECUTABLE.exists():
        print(f"Error: Executable not found at {EXECUTABLE}")
        print("Please run 'make' first.")
        raise SystemExit(1)

    RESULTS_DIR.mkdir(exist_ok=True)

    results = []

    print("Starting MPI Matrix Multiplication Benchmark")
    print("=" * 60)

    for num_procs in PROCESS_COUNTS:
        for size in MATRIX_SIZES:
            print(f"\nProcesses: {num_procs}, Matrix Size: {size}x{size}")

            run_records = []
            for run in range(NUM_RUNS):
                print(f"  Run {run + 1}/{NUM_RUNS}...", end=" ", flush=True)
                record = run_benchmark(size, size, size, num_procs)
                run_records.append(record)

                if record["success"]:
                    time_sec = float(record["parsed"]["time_sec"])
                    checksum = float(record["parsed"]["checksum"])
                    print(f"time: {time_sec:.6f}s checksum: {checksum:.9f}")
                else:
                    print("FAILED")
                    print(f"    reason: {record['error']}")
                    if record["stderr"].strip():
                        print(f"    stderr: {record['stderr'].strip()}")

            case_result = aggregate_case_results(size, size, size, num_procs, run_records)
            results.append(case_result)

            print(
                "  Summary: "
                f"status={case_result['status']} "
                f"success={case_result['successful_runs']}/{case_result['requested_runs']} "
                f"avg_time={format_metric(case_result['avg_time_sec'])}"
            )

    attach_baseline_consistency(results)

    results_file = RESULTS_DIR / "benchmark_results.json"
    with results_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_file}")

    print("\n" + "=" * 60)
    print("Performance Summary (Average Time in Seconds)")
    print("=" * 60)

    header = f"{'Procs':<8}"
    for size in MATRIX_SIZES:
        header += f"{size:<12}"
    print(header)
    print("-" * len(header))

    for num_procs in PROCESS_COUNTS:
        row = f"{num_procs:<8}"
        for size in MATRIX_SIZES:
            matching = [r for r in results if r["processes"] == num_procs and r["matrix_size"] == size]
            row += f"{format_metric(matching[0]['avg_time_sec']):<12}" if matching else f"{'N/A':<12}"
        print(row)

    print("\n" + "=" * 60)
    print("Speedup (relative to 1 process)")
    print("=" * 60)

    print(header)
    print("-" * len(header))

    baseline_times = {}
    for size in MATRIX_SIZES:
        matching = [r for r in results if r["processes"] == 1 and r["matrix_size"] == size]
        if matching and matching[0]["avg_time_sec"] is not None:
            baseline_times[size] = matching[0]["avg_time_sec"]

    for num_procs in PROCESS_COUNTS:
        row = f"{num_procs:<8}"
        for size in MATRIX_SIZES:
            matching = [r for r in results if r["processes"] == num_procs and r["matrix_size"] == size]
            if matching and size in baseline_times and matching[0]["avg_time_sec"] is not None:
                speedup = baseline_times[size] / matching[0]["avg_time_sec"]
                row += f"{speedup:<12.2f}"
            else:
                row += f"{'N/A':<12}"
        print(row)

    csv_file = RESULTS_DIR / "performance_table.csv"
    with csv_file.open("w", encoding="utf-8") as f:
        f.write("进程数/矩阵规模")
        for size in MATRIX_SIZES:
            f.write(f",{size}")
        f.write("\n")

        for num_procs in PROCESS_COUNTS:
            f.write(f"{num_procs}")
            for size in MATRIX_SIZES:
                matching = [r for r in results if r["processes"] == num_procs and r["matrix_size"] == size]
                if matching and matching[0]["avg_time_sec"] is not None:
                    f.write(f",{matching[0]['avg_time_sec']:.6f}")
                else:
                    f.write(",N/A")
            f.write("\n")

    print(f"\nCSV table saved to {csv_file}")

    problematic_cases = [
        entry
        for entry in results
        if entry["status"] != "ok" or entry["baseline_checksum_match"] is False
    ]

    if problematic_cases:
        print("\n" + "=" * 60)
        print("Problematic Cases")
        print("=" * 60)
        for entry in problematic_cases:
            print(
                f"processes={entry['processes']} size={entry['matrix_size']} "
                f"status={entry['status']} "
                f"success={entry['successful_runs']}/{entry['requested_runs']} "
                f"baseline_match={entry['baseline_checksum_match']}"
            )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
