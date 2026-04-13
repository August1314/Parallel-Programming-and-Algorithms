#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import subprocess
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
SCRIPTS_DIR = ROOT_DIR / "scripts"


class SubmissionFinalizeTest(unittest.TestCase):
    def test_report_benchmark_json_contains_v6(self) -> None:
        payload = json.loads((RESULTS_DIR / "report_benchmark.json").read_text(encoding="utf-8"))
        versions = {item["version"] for item in payload["summary_512"]}

        self.assertIn("v6_mkl", versions)
        self.assertTrue(payload["mkl"]["available"])

    def test_report_benchmark_csv_contains_v6(self) -> None:
        with (RESULTS_DIR / "report_benchmark.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        versions = {row["version"] for row in rows}
        self.assertIn("v6_mkl", versions)

    def test_legacy_run_entrypoint_works(self) -> None:
        legacy_runner = SCRIPTS_DIR / "run_matrix_mul"
        completed = subprocess.run(
            [str(legacy_runner), "v1_python", "2", "2", "2", "20250401"],
            check=False,
            text=True,
            capture_output=True,
        )

        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertIn("version=v1_python", completed.stdout)

    def test_legacy_benchmark_entrypoint_is_available(self) -> None:
        legacy_benchmark = SCRIPTS_DIR / "benchmark_lab1.py"
        completed = subprocess.run(
            ["python3", str(legacy_benchmark), "--help"],
            check=False,
            text=True,
            capture_output=True,
        )

        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertIn("--include-mkl", completed.stdout)


if __name__ == "__main__":
    unittest.main()
