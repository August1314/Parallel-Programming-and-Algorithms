#!/usr/bin/env python3
"""Generate test query files for each dataset."""
import csv
import os
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
LAB_DIR = SCRIPT_DIR.parent
DATA_DIR = LAB_DIR / "data"

SEED = 20250401
NUM_QUERIES = 100


def get_num_nodes(csv_path):
    min_id = float("inf")
    max_id = -1
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = int(row["source"])
            v = int(row["target"])
            min_id = min(min_id, u, v)
            max_id = max(max_id, u, v)
    min_id = int(min_id)
    offset = 1 if min_id == 1 else 0
    return max_id - offset + 1, offset


def main():
    rng = random.Random(SEED)

    for graph_name in ["updated_flower", "updated_mouse"]:
        csv_path = DATA_DIR / f"{graph_name}.csv"
        if not csv_path.exists():
            print(f"Skip: {csv_path} not found")
            continue

        num_nodes, offset = get_num_nodes(csv_path)
        print(f"{graph_name}: num_nodes={num_nodes}, offset={offset}")

        queries_path = DATA_DIR / f"queries_{graph_name}.txt"
        queries = []
        for _ in range(NUM_QUERIES):
            u = rng.randrange(num_nodes) + offset
            v = rng.randrange(num_nodes) + offset
            queries.append((u, v))

        with open(queries_path, "w") as f:
            for u, v in queries:
                f.write(f"{u} {v}\n")

        print(f"  Generated {len(queries)} queries → {queries_path}")


if __name__ == "__main__":
    main()
