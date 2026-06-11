#!/usr/bin/env python3
"""Benchmark lab11 CUDA Convolution across sizes, strides, and kernels."""
from __future__ import annotations
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

# Test configuration
INPUT_SIZES = [32, 64, 128, 256, 512, 1024, 2048, 4096]
STRIDES = [1, 2, 3]
KERNELS = [
    (1, "naive_conv"),
    (2, "tiled_conv"),
    (3, "im2col_gemm"),
    (4, "cudnn"),
]
# Block size variants for naive_conv only
NAIVE_BLOCK_DIMS = [(8, 8), (16, 16), (32, 32)]
REPEAT = 3

# Track OOM per (kernel_id, stride): skip larger sizes after OOM
oom_skip: set[tuple[int, int]] = set()


def parse_output(output: str) -> dict[str, str]:
    """Parse key=value format output."""
    result: dict[str, str] = {}
    for line in output.strip().split("\n"):
        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", line)
        if m:
            result[m.group(1)] = m.group(2)
    return result


def run_config(hw: int, stride: int, kernel_id: int,
               block_x: int = 16, block_y: int = 16) -> dict:
    """Run a single convolution configuration."""
    if kernel_id == 4:
        binary = BIN_DIR / "conv_cudnn"
    else:
        binary = BIN_DIR / "conv"

    if not binary.exists():
        return {"status": "skip", "stderr": f"Binary not found: {binary}"}

    cmd = [str(binary), str(hw), str(stride), str(kernel_id)]
    if kernel_id == 1:
        cmd += [str(block_x), str(block_y)]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return {"status": "error", "stderr": "timeout (>600s)"}

    if proc.returncode != 0:
        stderr_lower = proc.stderr.lower()
        if any(kw in stderr_lower for kw in
               ["out of memory", "insufficient", "memory"]):
            return {"status": "oom", "stderr": proc.stderr.strip()}
        return {"status": "error", "stderr": proc.stderr.strip()}

    metrics = parse_output(proc.stdout)
    metrics["status"] = "ok"
    metrics["stderr"] = ""
    return metrics


def main() -> int:
    results_dir = Path(os.environ.get(
        "LAB11_RESULTS_DIR", str(LAB_DIR / "results")))
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "summary.csv"
    cudnn_available = (BIN_DIR / "conv_cudnn").exists()

    rows = 0
    with open(csv_path, "w", newline="") as f:
        writer = None

        for size in INPUT_SIZES:
            for stride in STRIDES:
                for kernel_id, kernel_name in KERNELS:
                    # Skip cuDNN if not built
                    if kernel_id == 4 and not cudnn_available:
                        print(f"  SKIP size={size} stride={stride} "
                              f"{kernel_name}: cuDNN binary not built")
                        continue

                    # Skip if OOM'd earlier for this (kernel_id, stride)
                    oom_key = (kernel_id, stride)
                    if oom_key in oom_skip:
                        continue

                    # Block size list per kernel
                    if kernel_id == 1:
                        block_list = NAIVE_BLOCK_DIMS
                    else:
                        block_list = [(0, 0)]

                    for bx, by in block_list:
                        times: list[float] = []
                        best_metrics: dict = {}

                        for run_idx in range(REPEAT):
                            try:
                                metrics = run_config(size, stride, kernel_id,
                                                     bx, by)
                            except Exception as e:
                                print(f"  FAIL H=W={size} stride={stride} "
                                      f"kernel={kernel_name} "
                                      f"run={run_idx+1}: {e}")
                                continue

                            status = metrics.get("status", "unknown")

                            if status == "oom":
                                print(f"  OOM  H=W={size} stride={stride} "
                                      f"kernel={kernel_name}: "
                                      f"skipping larger sizes for this config")
                                oom_skip.add(oom_key)
                                break

                            if status == "skip":
                                print(f"  SKIP H=W={size} stride={stride} "
                                      f"kernel={kernel_name}: "
                                      f"{metrics.get('stderr', '')}")
                                break

                            if status != "ok":
                                print(f"  FAIL H=W={size} stride={stride} "
                                      f"kernel={kernel_name} "
                                      f"run={run_idx+1}: "
                                      f"{metrics.get('stderr', 'unknown')}")
                                continue

                            try:
                                t = float(metrics.get("kernel_time_ms", "0"))
                            except ValueError:
                                continue

                            times.append(t)
                            best_metrics = metrics
                            gflops = metrics.get("gflops", "N/A")
                            print(f"  H=W={size:>4} stride={stride} "
                                  f"kernel={kernel_name:<12} "
                                  f"run={run_idx+1}: {t:.6f}ms  "
                                  f"{gflops} GFLOPS")

                        if not times:
                            print(f"  SKIP H=W={size} stride={stride} "
                                  f"kernel={kernel_name}: all runs failed")
                            continue

                        avg_time = round(statistics.mean(times), 6)

                        row: dict = {
                            "H": size,
                            "W": size,
                            "stride": stride,
                            "kernel": kernel_name,
                            "kernel_id": kernel_id,
                            "block_x": bx,
                            "block_y": by,
                            "avg_kernel_time_ms": avg_time,
                            "num_runs": len(times),
                            "status": "ok",
                        }

                        # Merge extra fields from last successful run
                        for key in ["gflops", "memory_usage_mb", "checksum",
                                    "correct", "im2col_time_ms",
                                    "gemm_time_ms", "H_out", "W_out",
                                    "block_dim", "grid_dim", "padding"]:
                            if key in best_metrics:
                                row[key] = best_metrics[key]

                        if writer is None:
                            writer = csv.DictWriter(f,
                                                    fieldnames=list(row.keys()))
                            writer.writeheader()

                        writer.writerow(row)
                        rows += 1

    print(f"\nSaved {rows} records to {csv_path}")
    return 0 if rows > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
