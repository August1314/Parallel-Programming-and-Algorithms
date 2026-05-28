#!/usr/bin/env python3
"""Extended benchmark for lab6 heated_plate —  experiment part 2a."""
import csv
import json
import math
import os
import re
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
LAB7_DIR = SCRIPT_DIR.parent
LAB6_DIR = LAB7_DIR.parent / "lab6"

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
GRID_SIZES = [(64, 64), (128, 128), (256, 256), (512, 512)]
EPSILON = 0.1
NUM_RUNS = 3
CHUNK_SIZE = 8
TIMEOUT_SEC = 300
REQUIRED_KEYS = (
    "experiment", "backend", "version", "time_sec", "checksum", "max_abs",
    "threads", "m", "n", "epsilon", "schedule", "iterations", "diff",
)


def parse_key_value_output(output):
    data = {}
    for line in output.strip().splitlines():
        for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", line):
            data[key] = value
    return data


def run_case(version, m, n, threads, epsilon):
    info = VERSIONS[version]
    cmd = [*info["command"], str(m), str(n), str(epsilon), str(threads)]
    if info["backend"] == "pthreads_parallel_for":
        cmd.append(str(CHUNK_SIZE))
    try:
        result = subprocess.run(cmd, cwd=LAB6_DIR, text=True,
                                capture_output=True, timeout=TIMEOUT_SEC, check=False)
    except subprocess.TimeoutExpired as exc:
        return {"success": False, "parsed": {}, "error": "timeout",
                "stdout": exc.stdout or "", "stderr": exc.stderr or ""}

    parsed = parse_key_value_output(result.stdout)
    missing = [k for k in REQUIRED_KEYS if k not in parsed]
    success = result.returncode == 0 and not missing
    return {
        "success": success, "parsed": parsed,
        "returncode": result.returncode,
        "stdout": result.stdout, "stderr": result.stderr,
        "error": "" if success else f"missing: {missing}"
    }


def summarize(version, backend, m, n, threads, runs):
    successful = [r for r in runs if r["success"]]
    if not successful:
        return {"version": version, "backend": backend, "threads": threads,
                "m": m, "n": n, "status": "failed",
                "avg_time_sec": None, "checksum": None}

    times = [float(r["parsed"]["time_sec"]) for r in successful]
    checksums = [float(r["parsed"]["checksum"]) for r in successful]
    return {
        "version": version, "backend": backend, "threads": threads,
        "m": m, "n": n, "status": "ok",
        "avg_time_sec": sum(times) / len(times),
        "checksum": checksums[0],
        "iterations": int(successful[0]["parsed"]["iterations"]),
    }


def main():
    output_dir = LAB7_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure lab6 is built
    subprocess.run(["./scripts/docker_make.sh"], cwd=LAB6_DIR, check=True)

    results = []
    for version, info in VERSIONS.items():
        for m, n in GRID_SIZES:
            for threads in THREAD_COUNTS:
                runs = [run_case(version, m, n, threads, EPSILON) for _ in range(NUM_RUNS)]
                row = summarize(version, info["backend"], m, n, threads, runs)
                results.append(row)
                print(f"{version} {m}x{n} t={threads}: {row['status']} time={row['avg_time_sec']}")

    (output_dir / "heated_plate_extended.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = output_dir / "heated_plate_extended.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["version", "backend", "m", "n", "threads", "status",
                         "avg_time_sec", "checksum", "iterations"])
        for r in results:
            writer.writerow([r["version"], r["backend"], r["m"], r["n"], r["threads"],
                             r["status"],
                             f"{r['avg_time_sec']:.9f}" if r["avg_time_sec"] else "",
                             r.get("checksum", ""), r.get("iterations", "")])

    print(f"\nHeated plate extended results saved to {output_dir}")


if __name__ == "__main__":
    main()
