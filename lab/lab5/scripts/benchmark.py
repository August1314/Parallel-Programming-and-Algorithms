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

VERSIONS = {
    "openmp_default": {
        "backend": "openmp",
        "command": ["./scripts/run_openmp_matmul.sh", "default"],
    },
    "openmp_static1": {
        "backend": "openmp",
        "command": ["./scripts/run_openmp_matmul.sh", "static1"],
    },
    "openmp_dynamic1": {
        "backend": "openmp",
        "command": ["./scripts/run_openmp_matmul.sh", "dynamic1"],
    },
    "parallel_for_row_block": {
        "backend": "pthreads_parallel_for",
        "command": ["./scripts/run_parallel_for_matmul.sh"],
    },
}

THREAD_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8]
MATRIX_SIZES = [512, 1024, 2048]
SEED = 20250401
NUM_RUNS = 3
TIMEOUT_SEC = 300
REQUIRED_KEYS = ("experiment", "backend", "version", "time_sec", "checksum", "max_abs", "threads", "m", "n", "k", "seed")


def get_results_dir() -> Path:
    configured = os.environ.get("LAB5_RESULTS_DIR")
    if not configured:
        return LAB_DIR / "results"
    return Path(configured).expanduser().resolve()


def parse_int_list(name: str, default: list[int]) -> list[int]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def parse_key_value_output(output: str) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in output.strip().splitlines():
        for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", line):
            data[key] = value
    return data


def run_case(version: str, size: int, threads: int, seed: int) -> dict:
    info = VERSIONS[version]
    cmd = [*info["command"], str(size), str(size), str(size), str(seed), str(threads)]
    try:
        result = subprocess.run(
            cmd,
            cwd=LAB_DIR,
            text=True,
            capture_output=True,
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

    parsed = parse_key_value_output(result.stdout)
    missing = [key for key in REQUIRED_KEYS if key not in parsed]
    success = result.returncode == 0 and not missing

    error = ""
    if not success:
        parts: list[str] = []
        if result.returncode != 0:
            parts.append(f"return code {result.returncode}")
        if missing:
            parts.append(f"missing keys: {', '.join(missing)}")
        if result.stderr.strip():
            parts.append("stderr captured")
        error = "; ".join(parts) if parts else "unknown benchmark failure"

    return {
        "success": success,
        "command": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "parsed": parsed,
        "error": error,
    }


def summarize_runs(version: str, backend: str, size: int, threads: int, runs: list[dict]) -> dict:
    successful = [run for run in runs if run["success"]]
    failed = [run for run in runs if not run["success"]]
    times = [float(run["parsed"]["time_sec"]) for run in successful]
    checksums = [float(run["parsed"]["checksum"]) for run in successful]
    max_abs_values = [float(run["parsed"]["max_abs"]) for run in successful]
    checksum_consistent = len(checksums) <= 1 or all(math.isclose(value, checksums[0], rel_tol=0.0, abs_tol=1e-9) for value in checksums[1:])
    max_abs_consistent = len(max_abs_values) <= 1 or all(math.isclose(value, max_abs_values[0], rel_tol=0.0, abs_tol=1e-9) for value in max_abs_values[1:])

    if failed:
        status = "partial_failure" if successful else "failed"
    elif not checksum_consistent or not max_abs_consistent:
        status = "checksum_mismatch"
    else:
        status = "ok"

    return {
        "experiment": "matmul",
        "backend": backend,
        "version": version,
        "threads": threads,
        "m": size,
        "n": size,
        "k": size,
        "seed": SEED,
        "requested_runs": len(runs),
        "successful_runs": len(successful),
        "failed_runs": len(failed),
        "status": status,
        "avg_time_sec": (sum(times) / len(times)) if times else None,
        "checksums": checksums,
        "max_abs_values": max_abs_values,
        "checksum": checksums[0] if checksums else None,
        "max_abs": max_abs_values[0] if max_abs_values else None,
        "baseline_checksum_match": None,
        "runs": runs,
    }


def attach_baseline_consistency(results: list[dict]) -> None:
    baselines: dict[tuple[str, int], float] = {}
    for row in results:
        if row["status"] == "ok" and row["threads"] == 1 and row["checksum"] is not None:
            baselines[(row["version"], row["m"])] = float(row["checksum"])

    for row in results:
        baseline = baselines.get((row["version"], row["m"]))
        if baseline is None or row["checksum"] is None:
            row["baseline_checksum_match"] = None
            continue
        row["baseline_checksum_match"] = math.isclose(float(row["checksum"]), baseline, rel_tol=0.0, abs_tol=1e-9)
        if row["status"] == "ok" and row["baseline_checksum_match"] is False:
            row["status"] = "baseline_checksum_mismatch"


def write_summary_csv(results: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["experiment", "backend", "version", "status", "threads", "avg_time_sec", "m", "n", "k", "baseline_checksum_match"])
        for row in results:
            writer.writerow([
                row["experiment"],
                row["backend"],
                row["version"],
                row["status"],
                row["threads"],
                "" if row["avg_time_sec"] is None else f"{row['avg_time_sec']:.9f}",
                row["m"],
                row["n"],
                row["k"],
                row["baseline_checksum_match"],
            ])


def write_performance_table(results: list[dict], path: Path) -> None:
    versions = [name for name in VERSIONS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["backend", "version", "threads", *MATRIX_SIZES])
        for version in versions:
            for threads in THREAD_COUNTS:
                row = [VERSIONS[version]["backend"], version, threads]
                for size in MATRIX_SIZES:
                    match = next((item for item in results if item["version"] == version and item["threads"] == threads and item["m"] == size), None)
                    row.append(f"{match['avg_time_sec']:.6f}" if match and match["avg_time_sec"] is not None else "N/A")
                writer.writerow(row)


def main() -> None:
    global THREAD_COUNTS, MATRIX_SIZES, NUM_RUNS

    THREAD_COUNTS = parse_int_list("LAB5_BENCH_THREADS", THREAD_COUNTS)
    MATRIX_SIZES = parse_int_list("LAB5_BENCH_SIZES", MATRIX_SIZES)
    NUM_RUNS = parse_positive_int("LAB5_BENCH_RUNS", NUM_RUNS)

    subprocess.run(["./scripts/docker_make.sh"], cwd=LAB_DIR, check=True)

    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for version, info in VERSIONS.items():
        for threads in THREAD_COUNTS:
            for size in MATRIX_SIZES:
                runs = [run_case(version, size, threads, SEED) for _ in range(NUM_RUNS)]
                results.append(summarize_runs(version, info["backend"], size, threads, runs))

    attach_baseline_consistency(results)

    with (results_dir / "benchmark_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    write_summary_csv(results, results_dir / "summary_by_version.csv")
    write_performance_table(results, results_dir / "performance_table.csv")

    if any(row["status"] != "ok" for row in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
