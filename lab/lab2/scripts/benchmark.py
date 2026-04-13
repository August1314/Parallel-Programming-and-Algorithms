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
BIN_DIR = LAB_DIR / "bin"
EXECUTABLES = {
    "mpi_v1_p2p": BIN_DIR / "mpi_matmul_v1",
    "mpi_v2_collective": BIN_DIR / "mpi_matmul_v2_collective",
}
PROCESS_COUNTS = [1, 2, 4, 8, 16]
MATRIX_SIZES = [128, 256, 512, 1024, 2048]
SEED = 20250401
NUM_RUNS = 3
TIMEOUT_SEC = 300
REQUIRED_KEYS = ("version", "time_sec", "checksum", "max_abs", "processes", "m", "n", "k", "seed")


def get_results_dir() -> Path:
    configured = os.environ.get("LAB2_RESULTS_DIR")
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


def parse_key_value_output(output: str) -> dict:
    data = {}
    for line in output.strip().splitlines():
        for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", line):
            data[key] = value
    return data


def benchmark_command(executable: Path, m: int, n: int, k: int, num_procs: int, seed: int) -> list[str]:
    return ["mpirun", "-np", str(num_procs), "--oversubscribe", str(executable), str(m), str(n), str(k), str(seed)]


def run_case(version: str, m: int, n: int, k: int, num_procs: int, seed: int = SEED) -> dict:
    cmd = benchmark_command(EXECUTABLES[version], m, n, k, num_procs, seed)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SEC, check=False)
    except subprocess.TimeoutExpired as exc:
        return {"success": False, "version": version, "command": cmd, "returncode": None, "stdout": exc.stdout or "", "stderr": exc.stderr or "", "parsed": {}, "error": f"timeout after {TIMEOUT_SEC} seconds"}
    except OSError as exc:
        return {"success": False, "version": version, "command": cmd, "returncode": None, "stdout": "", "stderr": "", "parsed": {}, "error": f"failed to execute command: {exc}"}

    parsed = parse_key_value_output(result.stdout)
    missing_keys = [key for key in REQUIRED_KEYS if key not in parsed]
    success = result.returncode == 0 and not missing_keys
    if success:
        error = ""
    else:
        parts = []
        if result.returncode != 0:
            parts.append(f"return code {result.returncode}")
        if missing_keys:
            parts.append(f"missing keys: {', '.join(missing_keys)}")
        if result.stderr.strip():
            parts.append("stderr captured")
        error = "; ".join(parts) if parts else "unknown benchmark failure"
    return {"success": success, "version": version, "command": cmd, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "parsed": parsed, "error": error}


def summarize_runs(version: str, size: int, num_procs: int, runs: list[dict]) -> dict:
    successful = [r for r in runs if r["success"]]
    failed = [r for r in runs if not r["success"]]
    times = [float(r["parsed"]["time_sec"]) for r in successful]
    checksums = [float(r["parsed"]["checksum"]) for r in successful]
    checksum_consistent = len(checksums) <= 1 or all(math.isclose(v, checksums[0], rel_tol=0.0, abs_tol=1e-9) for v in checksums[1:])
    if failed:
        status = "partial_failure" if successful else "failed"
    elif not checksum_consistent:
        status = "checksum_mismatch"
    else:
        status = "ok"
    return {
        "version": version,
        "processes": num_procs,
        "matrix_size": size,
        "m": size,
        "n": size,
        "k": size,
        "requested_runs": len(runs),
        "successful_runs": len(successful),
        "failed_runs": len(failed),
        "status": status,
        "avg_time_sec": (sum(times) / len(times)) if times else None,
        "all_times": times,
        "checksums": checksums,
        "baseline_checksum": None,
        "baseline_checksum_match": None,
        "runs": runs,
    }


def attach_baseline_consistency(results: list[dict]) -> None:
    baselines = {}
    for row in results:
        if row["processes"] == 1 and row["successful_runs"] > 0 and row["status"] == "ok":
            baselines[(row["version"], row["matrix_size"])] = row["checksums"][0]
    for row in results:
        baseline = baselines.get((row["version"], row["matrix_size"]))
        row["baseline_checksum"] = baseline
        if baseline is None or not row["checksums"]:
            row["baseline_checksum_match"] = None
        else:
            row["baseline_checksum_match"] = all(math.isclose(v, baseline, rel_tol=0.0, abs_tol=1e-9) for v in row["checksums"])
            if row["status"] == "ok" and row["baseline_checksum_match"] is False:
                row["status"] = "baseline_checksum_mismatch"


def write_csv_summary(results: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["version", "processes", "matrix_size", "avg_time_sec", "status", "successful_runs", "failed_runs", "baseline_checksum_match"])
        for row in results:
            writer.writerow([row["version"], row["processes"], row["matrix_size"], row["avg_time_sec"], row["status"], row["successful_runs"], row["failed_runs"], row["baseline_checksum_match"]])


def write_course_table(results: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["version", "进程数/矩阵规模", *MATRIX_SIZES])
        for version in EXECUTABLES:
            for procs in PROCESS_COUNTS:
                row = [version, procs]
                for size in MATRIX_SIZES:
                    match = next((item for item in results if item["version"] == version and item["processes"] == procs and item["matrix_size"] == size), None)
                    row.append(f"{match['avg_time_sec']:.6f}" if match and match["avg_time_sec"] is not None else "N/A")
                writer.writerow(row)


def main() -> None:
    global PROCESS_COUNTS, MATRIX_SIZES, NUM_RUNS
    PROCESS_COUNTS = parse_int_list("LAB2_BENCH_PROCS", PROCESS_COUNTS)
    MATRIX_SIZES = parse_int_list("LAB2_BENCH_SIZES", MATRIX_SIZES)
    NUM_RUNS = parse_positive_int("LAB2_BENCH_RUNS", NUM_RUNS)

    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for version, executable in EXECUTABLES.items():
        if not executable.exists():
            raise SystemExit(f"missing executable: {executable}")
        for procs in PROCESS_COUNTS:
            for size in MATRIX_SIZES:
                runs = [run_case(version, size, size, size, procs) for _ in range(NUM_RUNS)]
                results.append(summarize_runs(version, size, procs, runs))
    attach_baseline_consistency(results)
    with (results_dir / "benchmark_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    write_course_table(results, results_dir / "performance_table.csv")
    write_csv_summary(results, results_dir / "summary_by_version.csv")
    bad = [row for row in results if row["status"] != "ok"]
    if bad:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
