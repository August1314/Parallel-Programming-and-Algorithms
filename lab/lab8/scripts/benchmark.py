#!/usr/bin/env python3
"""Benchmark APSP serial and OpenMP implementations."""
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
LAB_DIR = SCRIPT_DIR.parent
DATA_DIR = LAB_DIR / "data"

GRAPH_FILES = {
    "flower": DATA_DIR / "updated_flower.csv",
    "mouse": DATA_DIR / "updated_mouse.csv",
}
QUERY_FILES = {
    "flower": DATA_DIR / "queries_updated_flower.txt",
    "mouse": DATA_DIR / "queries_updated_mouse.txt",
}
THREAD_COUNTS = [1, 2, 4, 8, 16]
NUM_RUNS = 3
TIMEOUT_SEC = 600

REQUIRED_KEYS = ("experiment", "backend", "num_nodes", "num_edges",
                 "num_queries", "num_threads", "time_sec", "checksum")


def get_results_dir():
    configured = os.environ.get("LAB8_RESULTS_DIR")
    return Path(configured) if configured else (LAB_DIR / "results")


def parse_kv_output(output):
    data = {}
    for line in output.strip().splitlines():
        for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)="
                                     r"([^\s]+)", line):
            data[key.strip()] = value.strip()
    return data


def run_serial(graph_path, queries_path):
    cmd = [str(LAB_DIR / "bin" / "apsp_serial"),
           str(graph_path), str(queries_path)]
    try:
        result = subprocess.run(cmd, cwd=LAB_DIR, text=True,
                                capture_output=True, timeout=TIMEOUT_SEC,
                                check=False)
    except subprocess.TimeoutExpired as exc:
        return {"success": False, "command": cmd, "error": "timeout",
                "stdout": exc.stdout or "", "stderr": exc.stderr or "",
                "parsed": {}}

    parsed = parse_kv_output(result.stdout)
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


def run_openmp(graph_path, queries_path, num_threads):
    cmd = [str(LAB_DIR / "bin" / "apsp_openmp"),
           str(graph_path), str(queries_path), str(num_threads)]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(num_threads)
    try:
        result = subprocess.run(cmd, cwd=LAB_DIR, text=True,
                                capture_output=True, timeout=TIMEOUT_SEC,
                                check=False, env=env)
    except subprocess.TimeoutExpired as exc:
        return {"success": False, "command": cmd, "error": "timeout",
                "stdout": exc.stdout or "", "stderr": exc.stderr or "",
                "parsed": {}}

    parsed = parse_kv_output(result.stdout)
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


def summarize_runs(dataset_name, backend, num_threads, runs):
    successful = [r for r in runs if r["success"]]
    failed = [r for r in runs if not r["success"]]

    if not successful:
        return {
            "dataset": dataset_name,
            "backend": backend,
            "num_threads": num_threads,
            "status": "failed",
            "avg_time_sec": None,
            "checksum": None,
            "requested_runs": len(runs),
            "successful_runs": 0,
            "failed_runs": len(failed),
        }

    times = [float(r["parsed"]["time_sec"]) for r in successful]
    checksums = [float(r["parsed"]["checksum"]) for r in successful]

    # Pick up metadata from first successful run
    p = successful[0]["parsed"]
    return {
        "dataset": dataset_name,
        "backend": backend,
        "num_nodes": int(p.get("num_nodes", 0)),
        "num_edges": int(p.get("num_edges", 0)),
        "num_queries": int(p.get("num_queries", 0)),
        "num_threads": num_threads,
        "status": "ok",
        "avg_time_sec": sum(times) / len(times),
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

    for dataset_name, graph_path in GRAPH_FILES.items():
        queries_path = QUERY_FILES[dataset_name]
        if not graph_path.exists():
            print(f"Skip {dataset_name}: graph file not found")
            continue
        if not queries_path.exists():
            print(f"Skip {dataset_name}: queries file not found, run generate_queries.py first")
            continue

        # Serial (1 thread baseline)
        print(f"\n=== {dataset_name} serial ===")
        runs = [run_serial(graph_path, queries_path) for _ in range(NUM_RUNS)]
        summary = summarize_runs(dataset_name, "serial", 1, runs)
        results.append(summary)
        print(f"  status={summary['status']}, time={summary['avg_time_sec']:.4f}s"
              if summary['avg_time_sec'] else f"  status={summary['status']}")

        # OpenMP with different thread counts
        for nt in THREAD_COUNTS:
            print(f"\n=== {dataset_name} openmp threads={nt} ===")
            runs = [run_openmp(graph_path, queries_path, nt)
                    for _ in range(NUM_RUNS)]
            summary = summarize_runs(dataset_name, "openmp", nt, runs)
            results.append(summary)
            t_str = f"{summary['avg_time_sec']:.4f}s" if summary['avg_time_sec'] else "N/A"
            print(f"  status={summary['status']}, time={t_str}")

    # Write JSON
    json_path = results_dir / "benchmark_results.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"\nJSON saved to {json_path}")

    # Write CSV
    csv_path = results_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "backend", "num_nodes", "num_edges",
                         "num_threads", "status",
                         "avg_time_sec", "checksum"])
        for r in results:
            writer.writerow([
                r["dataset"], r["backend"],
                r.get("num_nodes", ""), r.get("num_edges", ""),
                r["num_threads"], r["status"],
                f"{r['avg_time_sec']:.6f}" if r["avg_time_sec"] else "",
                f"{r['checksum']:.6f}" if r["checksum"] else "",
            ])
    print(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
