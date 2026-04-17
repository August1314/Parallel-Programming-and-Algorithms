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
    "matmul_v1_row_block": BIN_DIR / "pthreads_matmul_v1",
    "matmul_v2_cyclic_rows": BIN_DIR / "pthreads_matmul_v2_partitioned",
    "array_sum_v1_local_sum": BIN_DIR / "pthreads_array_sum_v1",
    "array_sum_v2_mutex_shared_sum": BIN_DIR / "pthreads_array_sum_v2_reduction",
}

THREAD_COUNTS = [1, 2, 4, 8, 16]
MATRIX_SIZES = [128, 256, 512, 1024, 2048]
ARRAY_SIZES = [1_000_000, 2_000_000, 4_000_000, 8_000_000, 16_000_000, 32_000_000, 64_000_000, 128_000_000]
SEED = 20250401
NUM_RUNS = 3
TIMEOUT_SEC = 300

REQUIRED_KEYS_MATMUL = ("experiment", "version", "time_sec", "checksum", "max_abs", "threads", "m", "n", "k", "seed")
REQUIRED_KEYS_ARRAY = ("experiment", "version", "time_sec", "sum", "threads", "length", "seed")


def get_results_dir() -> Path:
    configured = os.environ.get("LAB3_RESULTS_DIR") or os.environ.get("LAB2_RESULTS_DIR")
    if not configured:
        return LAB_DIR / "results"
    return Path(configured).expanduser().resolve()


def parse_int_list(primary: str, fallback: str, default: list[int]) -> list[int]:
    raw = os.environ.get(primary) or os.environ.get(fallback)
    if not raw:
        return default
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_positive_int(primary: str, fallback: str, default: int) -> int:
    raw = os.environ.get(primary) or os.environ.get(fallback)
    return int(raw) if raw else default


def parse_key_value_output(output: str) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in output.strip().splitlines():
        for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", line):
            data[key] = value
    return data


def run_cmd(cmd: list[str]) -> dict:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SEC, check=False)
    except subprocess.TimeoutExpired as exc:
        return {
            "success": False,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "parsed": {},
            "error": f"timeout after {TIMEOUT_SEC} seconds",
        }
    parsed = parse_key_value_output(result.stdout)
    return {
        "success": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "parsed": parsed,
        "error": "",
    }


def run_matmul(version_key: str, size: int, threads: int, seed: int) -> dict:
    exe = EXECUTABLES[version_key]
    cmd = [str(exe), str(size), str(size), str(size), str(seed), str(threads)]
    result = run_cmd(cmd)
    parsed = result["parsed"]
    missing = [key for key in REQUIRED_KEYS_MATMUL if key not in parsed]
    success = result["success"] and not missing
    if not success:
        parts = []
        if result["returncode"] not in (None, 0):
            parts.append(f"return code {result['returncode']}")
        if missing:
            parts.append(f"missing keys: {', '.join(missing)}")
        if result["stderr"].strip():
            parts.append("stderr captured")
        result["error"] = "; ".join(parts) if parts else "unknown benchmark failure"
    return {**result, "version_key": version_key, "size": size, "threads": threads, "seed": seed}


def run_array_sum(version_key: str, length: int, threads: int, seed: int) -> dict:
    exe = EXECUTABLES[version_key]
    cmd = [str(exe), str(length), str(seed), str(threads)]
    result = run_cmd(cmd)
    parsed = result["parsed"]
    missing = [key for key in REQUIRED_KEYS_ARRAY if key not in parsed]
    success = result["success"] and not missing
    if not success:
        parts = []
        if result["returncode"] not in (None, 0):
            parts.append(f"return code {result['returncode']}")
        if missing:
            parts.append(f"missing keys: {', '.join(missing)}")
        if result["stderr"].strip():
            parts.append("stderr captured")
        result["error"] = "; ".join(parts) if parts else "unknown benchmark failure"
    return {**result, "version_key": version_key, "length": length, "threads": threads, "seed": seed}


def summarize_runs(rows: list[dict], kind: str) -> dict:
    successful = [r for r in rows if r["success"]]
    failed = [r for r in rows if not r["success"]]
    times = [float(r["parsed"]["time_sec"]) for r in successful]
    summary = {
        "status": "ok",
        "requested_runs": len(rows),
        "successful_runs": len(successful),
        "failed_runs": len(failed),
        "avg_time_sec": (sum(times) / len(times)) if times else None,
        "runs": rows,
    }
    if kind == "matmul":
        checksums = [float(r["parsed"]["checksum"]) for r in successful]
        max_abs_values = [float(r["parsed"]["max_abs"]) for r in successful]
        checksum_consistent = len(checksums) <= 1 or all(math.isclose(v, checksums[0], rel_tol=0.0, abs_tol=1e-9) for v in checksums[1:])
        max_abs_consistent = len(max_abs_values) <= 1 or all(math.isclose(v, max_abs_values[0], rel_tol=0.0, abs_tol=1e-9) for v in max_abs_values[1:])
        summary.update({
            "checksums": checksums,
            "max_abs_values": max_abs_values,
            "checksum": checksums[0] if checksums else None,
            "max_abs": max_abs_values[0] if max_abs_values else None,
        })
        if failed or not successful:
            summary["status"] = "partial_failure" if successful else "failed"
        elif not checksum_consistent or not max_abs_consistent:
            summary["status"] = "checksum_mismatch"
    else:
        sums = [float(r["parsed"]["sum"]) for r in successful]
        sum_consistent = len(sums) <= 1 or all(math.isclose(v, sums[0], rel_tol=0.0, abs_tol=1e-9) for v in sums[1:])
        summary.update({
            "sums": sums,
            "sum": sums[0] if sums else None,
        })
        if failed or not successful:
            summary["status"] = "partial_failure" if successful else "failed"
        elif not sum_consistent:
            summary["status"] = "sum_mismatch"
    return summary


def write_summary_csv(results: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "version", "status", "threads", "time_sec", "m", "n", "k", "length"])
        for row in results:
            writer.writerow([
                row["experiment"],
                row["version"],
                row["status"],
                row["threads"],
                "" if row["avg_time_sec"] is None else f"{row['avg_time_sec']:.9f}",
                "" if row.get("m") is None else row["m"],
                "" if row.get("n") is None else row["n"],
                "" if row.get("k") is None else row["k"],
                "" if row.get("length") is None else row["length"],
            ])


def write_table(rows: list[dict], size_key: str, value_key: str, output_path: Path) -> None:
    versions = sorted({row["version"] for row in rows})
    threads = sorted({row["threads"] for row in rows})
    sizes = sorted({int(row[size_key]) for row in rows})
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["version", "threads", *sizes])
        for version in versions:
            for thread in threads:
                line = [version, thread]
                for size in sizes:
                    match = next((r for r in rows if r["version"] == version and r["threads"] == thread and int(r[size_key]) == size), None)
                    if not match or match.get(value_key) is None:
                        line.append("N/A")
                    else:
                        line.append(f"{float(match[value_key]):.6f}")
                writer.writerow(line)


def attach_speedup_efficiency(rows: list[dict], size_key: str) -> None:
    baseline = {}
    for row in rows:
        if row["threads"] == 1 and row["status"] == "ok" and row["avg_time_sec"] is not None:
            baseline[(row["version"], int(row[size_key]))] = float(row["avg_time_sec"])
    for row in rows:
        base = baseline.get((row["version"], int(row[size_key])))
        if base is None or row["avg_time_sec"] is None:
            row["speedup"] = None
            row["efficiency"] = None
        else:
            speedup = base / float(row["avg_time_sec"])
            row["speedup"] = speedup
            row["efficiency"] = speedup / int(row["threads"])


def main() -> None:
    global THREAD_COUNTS, MATRIX_SIZES, ARRAY_SIZES, NUM_RUNS
    THREAD_COUNTS = parse_int_list("LAB3_BENCH_THREADS", "LAB2_BENCH_PROCS", THREAD_COUNTS)
    MATRIX_SIZES = parse_int_list("LAB3_BENCH_MATMUL_SIZES", "LAB2_BENCH_SIZES", MATRIX_SIZES)
    ARRAY_SIZES = parse_int_list("LAB3_BENCH_ARRAY_SIZES", "LAB2_BENCH_ARRAY_SIZES", ARRAY_SIZES)
    NUM_RUNS = parse_positive_int("LAB3_BENCH_RUNS", "LAB2_BENCH_RUNS", NUM_RUNS)

    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for version_key in ("matmul_v1_row_block", "matmul_v2_cyclic_rows"):
        exe = EXECUTABLES[version_key]
        if not exe.exists():
            raise SystemExit(f"missing executable: {exe}")
        for threads in THREAD_COUNTS:
            for size in MATRIX_SIZES:
                runs = [run_matmul(version_key, size, threads, SEED) for _ in range(NUM_RUNS)]
                summary = summarize_runs(runs, "matmul")
                version = runs[0]["parsed"].get("version", version_key) if runs else version_key
                results.append({
                    "experiment": "pthreads_matmul",
                    "version": version,
                    "threads": threads,
                    "m": size,
                    "n": size,
                    "k": size,
                    "seed": SEED,
                    **summary,
                })

    for version_key in ("array_sum_v1_local_sum", "array_sum_v2_mutex_shared_sum"):
        exe = EXECUTABLES[version_key]
        if not exe.exists():
            raise SystemExit(f"missing executable: {exe}")
        for threads in THREAD_COUNTS:
            for length in ARRAY_SIZES:
                runs = [run_array_sum(version_key, length, threads, SEED) for _ in range(NUM_RUNS)]
                summary = summarize_runs(runs, "array_sum")
                version = runs[0]["parsed"].get("version", version_key) if runs else version_key
                results.append({
                    "experiment": "pthreads_array_sum",
                    "version": version,
                    "threads": threads,
                    "length": length,
                    "seed": SEED,
                    **summary,
                })

    matmul_rows = [row for row in results if row["experiment"] == "pthreads_matmul"]
    array_rows = [row for row in results if row["experiment"] == "pthreads_array_sum"]
    attach_speedup_efficiency(matmul_rows, "m")
    attach_speedup_efficiency(array_rows, "length")

    (results_dir / "benchmark_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_csv(results, results_dir / "summary_by_version.csv")

    write_table(matmul_rows, "m", "avg_time_sec", tables_dir / "time_table_matmul.csv")
    write_table(matmul_rows, "m", "speedup", tables_dir / "speedup_table_matmul.csv")
    write_table(matmul_rows, "m", "efficiency", tables_dir / "efficiency_table_matmul.csv")
    write_table(array_rows, "length", "avg_time_sec", tables_dir / "time_table_array_sum.csv")
    write_table(array_rows, "length", "speedup", tables_dir / "speedup_table_array_sum.csv")
    write_table(array_rows, "length", "efficiency", tables_dir / "efficiency_table_array_sum.csv")

    bad = [row for row in results if row["status"] != "ok"]
    if bad:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
