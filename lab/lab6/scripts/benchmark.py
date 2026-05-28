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
    "openmp_original": {
        "backend": "openmp",
        "command": ["./scripts/run_openmp_heated_plate.sh"],
    },
    "pthreads_parallel_for_block": {
        "backend": "pthreads_parallel_for",
        "command": ["./scripts/run_pthreads_heated_plate.sh", "block"],
    },
    "pthreads_parallel_for_cyclic": {
        "backend": "pthreads_parallel_for",
        "command": ["./scripts/run_pthreads_heated_plate.sh", "cyclic"],
    },
    "pthreads_parallel_for_dynamic": {
        "backend": "pthreads_parallel_for",
        "command": ["./scripts/run_pthreads_heated_plate.sh", "dynamic"],
    },
}

THREAD_COUNTS = [1, 2, 4, 8]
GRID_SIZES = [(128, 128), (256, 256), (500, 500)]
EPSILON = 0.1
NUM_RUNS = 3
CHUNK_SIZE = 8
TIMEOUT_SEC = 300
REQUIRED_KEYS = (
    "experiment",
    "backend",
    "version",
    "time_sec",
    "checksum",
    "max_abs",
    "threads",
    "m",
    "n",
    "epsilon",
    "schedule",
    "iterations",
    "diff",
)


def get_results_dir() -> Path:
    configured = os.environ.get("LAB6_RESULTS_DIR")
    if not configured:
        return LAB_DIR / "results"
    return Path(configured).expanduser().resolve()


def parse_int_list(name: str, default: list[int]) -> list[int]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_grid_sizes(name: str, default: list[tuple[int, int]]) -> list[tuple[int, int]]:
    raw = os.environ.get(name)
    if not raw:
        return default
    sizes: list[tuple[int, int]] = []
    for part in raw.split(","):
        item = part.strip().lower()
        if not item:
            continue
        if "x" in item:
            left, right = item.split("x", 1)
            sizes.append((int(left), int(right)))
        else:
            value = int(item)
            sizes.append((value, value))
    return sizes


def parse_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def parse_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


def parse_key_value_output(output: str) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in output.strip().splitlines():
        for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", line):
            data[key] = value
    return data


def run_case(version: str, m: int, n: int, threads: int, epsilon: float) -> dict:
    info = VERSIONS[version]
    cmd = [*info["command"], str(m), str(n), str(epsilon), str(threads)]
    if info["backend"] == "pthreads_parallel_for":
        cmd.append(str(CHUNK_SIZE))
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


def summarize_runs(version: str, backend: str, m: int, n: int, threads: int, runs: list[dict]) -> dict:
    successful = [run for run in runs if run["success"]]
    failed = [run for run in runs if not run["success"]]
    times = [float(run["parsed"]["time_sec"]) for run in successful]
    checksums = [float(run["parsed"]["checksum"]) for run in successful]
    max_abs_values = [float(run["parsed"]["max_abs"]) for run in successful]
    iterations = [int(run["parsed"]["iterations"]) for run in successful]
    diffs = [float(run["parsed"]["diff"]) for run in successful]
    checksum_consistent = len(checksums) <= 1 or all(math.isclose(value, checksums[0], rel_tol=0.0, abs_tol=1e-8) for value in checksums[1:])
    iteration_consistent = len(iterations) <= 1 or all(value == iterations[0] for value in iterations[1:])

    if failed:
        status = "partial_failure" if successful else "failed"
    elif not checksum_consistent or not iteration_consistent:
        status = "checksum_mismatch"
    else:
        status = "ok"

    parsed0 = successful[0]["parsed"] if successful else {}
    return {
        "experiment": "heated_plate",
        "backend": backend,
        "version": version,
        "threads": threads,
        "m": m,
        "n": n,
        "epsilon": EPSILON,
        "schedule": parsed0.get("schedule", ""),
        "chunk_size": parsed0.get("chunk_size", ""),
        "requested_runs": len(runs),
        "successful_runs": len(successful),
        "failed_runs": len(failed),
        "status": status,
        "avg_time_sec": (sum(times) / len(times)) if times else None,
        "iterations": iterations[0] if iterations else None,
        "diff": diffs[0] if diffs else None,
        "checksums": checksums,
        "max_abs_values": max_abs_values,
        "checksum": checksums[0] if checksums else None,
        "max_abs": max_abs_values[0] if max_abs_values else None,
        "thread1_checksum_match": None,
        "openmp_reference_match": None,
        "runs": runs,
    }


def attach_consistency(results: list[dict]) -> None:
    thread1_baselines: dict[tuple[str, int, int], float] = {}
    openmp_baselines: dict[tuple[int, int, int], float] = {}
    for row in results:
        if row["status"] == "ok" and row["checksum"] is not None:
            if row["threads"] == 1:
                thread1_baselines[(row["version"], row["m"], row["n"])] = float(row["checksum"])
            if row["version"] == "openmp_original":
                openmp_baselines[(row["m"], row["n"], row["threads"])] = float(row["checksum"])

    for row in results:
        checksum = row["checksum"]
        baseline = thread1_baselines.get((row["version"], row["m"], row["n"]))
        if baseline is not None and checksum is not None:
            row["thread1_checksum_match"] = math.isclose(float(checksum), baseline, rel_tol=0.0, abs_tol=1e-8)
            if row["status"] == "ok" and row["thread1_checksum_match"] is False:
                row["status"] = "thread1_checksum_mismatch"

        reference = openmp_baselines.get((row["m"], row["n"], row["threads"]))
        if reference is not None and checksum is not None:
            row["openmp_reference_match"] = math.isclose(float(checksum), reference, rel_tol=0.0, abs_tol=1e-8)
            if row["status"] == "ok" and row["openmp_reference_match"] is False:
                row["status"] = "openmp_reference_mismatch"


def write_summary_csv(results: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "experiment",
            "backend",
            "version",
            "status",
            "threads",
            "avg_time_sec",
            "m",
            "n",
            "epsilon",
            "schedule",
            "iterations",
            "thread1_checksum_match",
            "openmp_reference_match",
        ])
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
                row["epsilon"],
                row["schedule"],
                row["iterations"],
                row["thread1_checksum_match"],
                row["openmp_reference_match"],
            ])


def write_performance_table(results: list[dict], path: Path) -> None:
    size_labels = [f"{m}x{n}" for m, n in GRID_SIZES]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["backend", "version", "threads", *size_labels])
        for version in VERSIONS:
            for threads in THREAD_COUNTS:
                row = [VERSIONS[version]["backend"], version, threads]
                for m, n in GRID_SIZES:
                    match = next((item for item in results if item["version"] == version and item["threads"] == threads and item["m"] == m and item["n"] == n), None)
                    row.append(f"{match['avg_time_sec']:.6f}" if match and match["avg_time_sec"] is not None else "N/A")
                writer.writerow(row)


def main() -> None:
    global THREAD_COUNTS, GRID_SIZES, NUM_RUNS, EPSILON, CHUNK_SIZE

    THREAD_COUNTS = parse_int_list("LAB6_BENCH_THREADS", THREAD_COUNTS)
    GRID_SIZES = parse_grid_sizes("LAB6_BENCH_SIZES", GRID_SIZES)
    NUM_RUNS = parse_positive_int("LAB6_BENCH_RUNS", NUM_RUNS)
    EPSILON = parse_float("LAB6_BENCH_EPSILON", EPSILON)
    CHUNK_SIZE = parse_positive_int("LAB6_BENCH_CHUNK_SIZE", CHUNK_SIZE)

    subprocess.run(["./scripts/docker_make.sh"], cwd=LAB_DIR, check=True)

    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for version, info in VERSIONS.items():
        for threads in THREAD_COUNTS:
            for m, n in GRID_SIZES:
                runs = [run_case(version, m, n, threads, EPSILON) for _ in range(NUM_RUNS)]
                results.append(summarize_runs(version, info["backend"], m, n, threads, runs))

    attach_consistency(results)

    (results_dir / "benchmark_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_csv(results, results_dir / "summary_by_version.csv")
    write_performance_table(results, results_dir / "performance_table.csv")


if __name__ == "__main__":
    main()
