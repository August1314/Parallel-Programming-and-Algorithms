#!/usr/bin/env python3
"""Benchmark lab10 CUDA GEMM across sizes, kernels, and block dimensions."""
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

# 测试配置
MATRIX_SIZES = [128, 256, 512, 1024, 2048]
KERNELS = [
    (1, "naive"),
    (2, "tiled_16"),
    (3, "tiled_32"),
]
# naive kernel 的 block 大小变体
NAIVE_BLOCK_DIMS = [(8, 8), (16, 16), (32, 32)]
REPEAT = 3


def parse_output(output: str) -> dict[str, str]:
    """解析 key=value 格式输出"""
    result = {}
    for line in output.strip().split("\n"):
        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", line)
        if m:
            result[m.group(1)] = m.group(2)
    return result


def run_config(m: int, n: int, k: int, kernel_id: int,
               block_x: int, block_y: int) -> dict:
    """运行一次 GEMM 并返回解析后的指标"""
    binary = BIN_DIR / "gemm"
    if not binary.exists():
        raise FileNotFoundError(
            f"Binary not found: {binary}. Run build.sh first."
        )

    cmd = [str(binary), str(m), str(n), str(k),
           str(kernel_id), str(block_x), str(block_y)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if proc.returncode != 0:
        return {"status": "error", "stderr": proc.stderr.strip()}

    metrics = parse_output(proc.stdout)
    metrics["status"] = "ok"
    metrics["stderr"] = ""
    return metrics


def main():
    results_dir = os.environ.get("LAB10_RESULTS_DIR",
                                 str(LAB_DIR / "results"))
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    csv_path = results_path / "summary.csv"
    rows = 0

    with open(csv_path, "w", newline="") as f:
        writer = None

        for size in MATRIX_SIZES:
            M = N = K = size

            for kernel_id, kernel_name in KERNELS:
                # tiled kernel 使用固定 block 大小
                if kernel_id == 2:
                    block_list = [(16, 16)]
                elif kernel_id == 3:
                    block_list = [(32, 32)]
                else:
                    block_list = NAIVE_BLOCK_DIMS

                for bx, by in block_list:
                    times = []
                    best_metrics = None

                    for run_idx in range(REPEAT):
                        try:
                            metrics = run_config(M, N, K, kernel_id,
                                                 bx, by)
                        except Exception as e:
                            print(f"  FAIL M=N=K={size} kernel={kernel_name} "
                                  f"block={bx}x{by} run={run_idx+1}: {e}")
                            continue

                        if metrics.get("status") != "ok":
                            print(f"  FAIL M=N=K={size} kernel={kernel_name} "
                                  f"block={bx}x{by} run={run_idx+1}: "
                                  f"{metrics.get('stderr', 'unknown error')}")
                            continue

                        try:
                            t = float(metrics.get("kernel_time_ms", "0"))
                        except ValueError:
                            continue

                        times.append(t)
                        best_metrics = metrics
                        gflops = metrics.get("gflops", "N/A")
                        print(f"  M=N=K={size:>4} kernel={kernel_name:<10} "
                              f"block={bx}x{by} run={run_idx+1}: "
                              f"{t:.6f}ms  {gflops} GFLOPS")

                    if not times:
                        print(f"  SKIP M=N=K={size} kernel={kernel_name} "
                              f"block={bx}x{by}: all runs failed")
                        continue

                    avg_time = statistics.mean(times)

                    row = {
                        "M": M,
                        "N": N,
                        "K": K,
                        "kernel": kernel_name,
                        "kernel_id": kernel_id,
                        "block_x": bx,
                        "block_y": by,
                        "avg_kernel_time_ms": round(avg_time, 6),
                        "num_runs": len(times),
                        "status": "ok",
                    }

                    # 合并最后一次成功运行的额外字段
                    for key in ["gflops", "bandwidth_gb_s", "checksum",
                                "correct", "grid_dim", "block_dim",
                                "tile_dim"]:
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
