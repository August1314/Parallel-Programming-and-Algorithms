#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path


VERSIONS = [
    "v1_python",
    "v2_cpp_baseline",
    "v3_cpp_loop_reorder",
    "v4_cpp_optimized",
    "v5_cpp_unrolled",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark lab1 serial matrix multiplication versions with unified v6 routing")
    parser.add_argument("--sizes", nargs="+", default=["512x512x512"])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20250401)
    parser.add_argument("--peak-gflops", type=float, default=0.0)
    parser.add_argument("--include-mkl", action="store_true")
    parser.add_argument("--v6-mode", choices=["docker", "local"], default="docker")
    parser.add_argument("--output", default="results/benchmark_unified_latest.json")
    return parser.parse_args()


def parse_triplet(size: str) -> tuple[int, int, int]:
    parts = size.lower().split("x")
    if len(parts) != 3:
        raise ValueError(f"invalid size: {size}")
    return tuple(int(part) for part in parts)


def parse_program_output(output: str) -> dict[str, float | str]:
    result: dict[str, float | str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.strip().split("=", 1)
        if key in {"version"}:
            result[key] = value
        elif key == "m":
            dims = line.strip().split()
            for item in dims:
                dim_key, dim_value = item.split("=")
                result[dim_key] = int(dim_value)
        else:
            try:
                result[key] = float(value)
            except ValueError:
                result[key] = value
    return result


def run_once(
    script_path: Path,
    version: str,
    m: int,
    n: int,
    k: int,
    seed: int,
    v6_mode: str,
) -> dict[str, float | str]:
    completed = subprocess.run(
        [str(script_path), version, str(m), str(n), str(k), str(seed)],
        check=True,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "LAB1_V6_MODE": v6_mode,
        },
    )
    return parse_program_output(completed.stdout)


def main() -> int:
    args = parse_args()
    root_dir = Path(__file__).resolve().parent.parent
    runner = root_dir / "scripts" / "run_matrix_mul_unified"
    output_path = root_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    versions = list(VERSIONS)
    if args.include_mkl:
        versions.append("v6_mkl")

    summary: list[dict[str, object]] = []
    for size in args.sizes:
        m, n, k = parse_triplet(size)
        baseline_time = None
        previous_time = None
        baseline_checksum = None
        for version in versions:
            measurements = [
                run_once(runner, version, m, n, k, args.seed, args.v6_mode)
                for _ in range(args.repeat)
            ]
            times = [float(item["time_sec"]) for item in measurements]
            avg_time = max(statistics.mean(times), 1e-12)
            median_time = max(statistics.median(times), 1e-12)
            checksum = float(measurements[0]["checksum"])
            for item in measurements[1:]:
                if abs(float(item["checksum"]) - checksum) > 1e-6:
                    raise RuntimeError(f"checksum mismatch detected in {version} for size {size}")
            gflops = 2.0 * m * n * k / avg_time / 1e9

            if baseline_time is None:
                baseline_time = avg_time
            if baseline_checksum is None:
                baseline_checksum = checksum
            elif abs(checksum - baseline_checksum) > 1e-6:
                raise RuntimeError(f"checksum mismatch between versions for size {size}: {version}")
            absolute_speedup = baseline_time / avg_time
            relative_speedup = 1.0 if previous_time is None else previous_time / avg_time
            previous_time = avg_time
            peak_percent = (gflops / args.peak_gflops * 100.0) if args.peak_gflops > 0 else 0.0

            summary.append(
                {
                    "size": size,
                    "version": version,
                    "avg_time_sec": avg_time,
                    "median_time_sec": median_time,
                    "checksum": checksum,
                    "relative_speedup": relative_speedup,
                    "absolute_speedup": absolute_speedup,
                    "gflops": gflops,
                    "peak_percent": peak_percent,
                }
            )

    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("size,version,avg_time_sec,median_time_sec,relative_speedup,absolute_speedup,gflops,peak_percent,checksum")
    for item in summary:
        print(
            f"{item['size']},{item['version']},{item['avg_time_sec']:.6f},{item['median_time_sec']:.6f},"
            f"{item['relative_speedup']:.6f},{item['absolute_speedup']:.6f},{item['gflops']:.6f},"
            f"{item['peak_percent']:.6f},{item['checksum']:.6f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
