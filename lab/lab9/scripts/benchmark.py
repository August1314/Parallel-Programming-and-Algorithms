#!/usr/bin/env python3
"""Benchmark lab9 CUDA matrix transpose across sizes, kernels, and block dims."""
import csv
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
LAB_DIR = SCRIPT_DIR.parent
BIN_DIR = LAB_DIR / "bin"

# Test matrix
MATRIX_SIZES = [512, 1024, 2048]
KERNELS = [
    (1, "naive"),
    (2, "coalesced_read"),
    (3, "shared_mem"),
]
BLOCK_DIMS = [(8, 8), (16, 16), (32, 32)]
REPEAT = 3


def parse_output(output: str) -> dict[str, str]:
    """Parse key=value lines from program output."""
    result = {}
    for line in output.strip().split("\n"):
        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)=" + r"([^\s]+)", line)
        if m:
            result[m.group(1)] = m.group(2)
    return result


def run_config(n: int, kernel_id: int, block_x: int, block_y: int) -> dict:
    """Run transpose once and return parsed metrics."""
    binary = BIN_DIR / "transpose"
    if not binary.exists():
        raise FileNotFoundError(f"Binary not found: {binary}. Run build.sh first.")

    cmd = [str(binary), str(n), str(kernel_id), str(block_x), str(block_y)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if proc.returncode != 0:
        return {"status": "error", "stderr": proc.stderr.strip()}

    metrics = parse_output(proc.stdout)
    metrics["status"] = "ok"
    metrics["stderr"] = ""
    return metrics


def main():
    results_dir = os.environ.get("LAB9_RESULTS_DIR",
                                 str(LAB_DIR / "results"))
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    csv_path = results_path / "summary.csv"
    rows = 0

    with open(csv_path, "w", newline="") as f:
        writer = None

        for n in MATRIX_SIZES:
            for kernel_id, kernel_name in KERNELS:
                # shared_mem kernel uses fixed TILE_DIM, skip non-32 block configs
                if kernel_id == 3:
                    block_list = [(32, 32)]
                else:
                    block_list = BLOCK_DIMS

                for bx, by in block_list:
                    times = []
                    best_metrics = None

                    for run_idx in range(REPEAT):
                        try:
                            metrics = run_config(n, kernel_id, bx, by)
                        except Exception as e:
                            print(f"  FAIL N={n} kernel={kernel_name} "
                                  f"block={bx}x{by} run={run_idx+1}: {e}")
                            continue

                        if metrics.get("status") != "ok":
                            print(f"  FAIL N={n} kernel={kernel_name} "
                                  f"block={bx}x{by} run={run_idx+1}: "
                                  f"{metrics.get('stderr', 'unknown error')}")
                            continue

                        try:
                            t = float(metrics.get("kernel_time_ms", "0"))
                        except ValueError:
                            continue

                        times.append(t)
                        best_metrics = metrics
                        print(f"  N={n:>4} kernel={kernel_name:<16} "
                              f"block={bx}x{by} run={run_idx+1}: "
                              f"{t:.6f}ms")

                    if not times:
                        print(f"  SKIP N={n} kernel={kernel_name} block={bx}x{by}: "
                              f"all runs failed")
                        continue

                    avg_time = statistics.mean(times)

                    row = {
                        "N": n,
                        "kernel": kernel_name,
                        "kernel_id": kernel_id,
                        "block_x": bx,
                        "block_y": by,
                        "avg_kernel_time_ms": round(avg_time, 6),
                        "num_runs": len(times),
                        "status": "ok",
                    }

                    # Merge extra fields from last successful run
                    for key in ["bandwidth_gb_s", "checksum", "correct",
                                "grid_dim", "block_dim"]:
                        if key in best_metrics:
                            row[key] = best_metrics[key]

                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                        writer.writeheader()

                    writer.writerow(row)
                    rows += 1

    print(f"\nSaved {rows} records to {csv_path}")
    return 0 if rows > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
