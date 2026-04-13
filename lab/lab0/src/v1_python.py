#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serial matrix multiplication baseline")
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("seed", nargs="?", default=20250401, type=int)
    parser.add_argument("--dump", action="store_true")
    return parser.parse_args()

MASK_64 = (1 << 64) - 1


def splitmix64_next(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & MASK_64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK_64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK_64
    return state, (z ^ (z >> 31)) & MASK_64


def next_unit_value(state: int) -> tuple[int, float]:
    state, raw = splitmix64_next(state)
    unit = ((raw >> 11) & ((1 << 53) - 1)) / float(1 << 53)
    return state, unit * 2.0 - 1.0


def generate_matrix(rows: int, cols: int, seed: int, salt: int) -> list[list[float]]:
    state = (seed ^ (salt * 0x9E3779B97F4A7C15)) & MASK_64
    matrix: list[list[float]] = []
    for _ in range(rows):
        row: list[float] = []
        for _ in range(cols):
            state, value = next_unit_value(state)
            row.append(value)
        matrix.append(row)
    return matrix


def checksum_matrix(matrix: list[list[float]]) -> float:
    checksum = 0.0
    index = 0
    for row in matrix:
        for value in row:
            checksum += value * ((index % 17) + 1)
            index += 1
    return checksum


def max_abs_matrix(matrix: list[list[float]]) -> float:
    max_abs = 0.0
    for row in matrix:
        for value in row:
            abs_value = abs(value)
            if abs_value > max_abs:
                max_abs = abs_value
    return max_abs


def matmul(a: list[list[float]], b: list[list[float]], m: int, n: int, k: int) -> list[list[float]]:
    c = [[0.0 for _ in range(k)] for _ in range(m)]
    for i in range(m):
        for j in range(k):
            sum_value = 0.0
            for p in range(n):
                sum_value += a[i][p] * b[p][j]
            c[i][j] = sum_value
    return c


def dump_matrix(name: str, matrix: list[list[float]]) -> None:
    print(f"{name}:")
    for row in matrix:
        print(" ".join(f"{value:.6f}" for value in row))


def main() -> int:
    args = parse_args()
    if args.m <= 0 or args.n <= 0 or args.k <= 0:
        raise SystemExit("matrix dimensions must be positive")

    a = generate_matrix(args.m, args.n, args.seed, 1)
    b = generate_matrix(args.n, args.k, args.seed, 2)

    start = time.perf_counter()
    c = matmul(a, b, args.m, args.n, args.k)
    elapsed = time.perf_counter() - start

    print("version=v1_python")
    print(f"m={args.m} n={args.n} k={args.k}")
    print(f"seed={args.seed}")
    print(f"time_sec={elapsed:.9f}")
    print(f"checksum={checksum_matrix(c):.9f}")
    print(f"max_abs={max_abs_matrix(c):.9f}")

    allow_dump = args.dump and args.m <= 8 and args.n <= 8 and args.k <= 8
    if allow_dump:
        dump_matrix("A", a)
        dump_matrix("B", b)
        dump_matrix("C", c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
