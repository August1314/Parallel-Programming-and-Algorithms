import subprocess
import unittest
from unittest import mock

from lab.lab1.scripts import benchmark


class BenchmarkScriptTests(unittest.TestCase):
    def test_parse_key_value_output_ignores_non_metrics_lines(self):
        output = "\n".join(
            [
                "version=mpi_v1_p2p",
                "m=128 n=128 k=128",
                "checksum=-1713.139278331",
                "noise without equals",
                "time_sec=0.001212000",
            ]
        )

        parsed = benchmark.parse_key_value_output(output)

        self.assertEqual(parsed["version"], "mpi_v1_p2p")
        self.assertEqual(parsed["m"], "128")
        self.assertEqual(parsed["n"], "128")
        self.assertEqual(parsed["k"], "128")
        self.assertEqual(parsed["checksum"], "-1713.139278331")
        self.assertEqual(parsed["time_sec"], "0.001212000")
        self.assertNotIn("noise without equals", parsed)

    def test_run_benchmark_returns_structured_failure_when_command_fails(self):
        completed = subprocess.CompletedProcess(
            args=["mpirun"],
            returncode=213,
            stdout="",
            stderr="No network interfaces were found",
        )

        with mock.patch("lab.lab1.scripts.benchmark.subprocess.run", return_value=completed):
            result = benchmark.run_benchmark(128, 128, 128, 4)

        self.assertFalse(result["success"])
        self.assertEqual(result["returncode"], 213)
        self.assertIn("return code 213", result["error"])
        self.assertIn("stderr captured", result["error"])
        self.assertEqual(result["stderr"], "No network interfaces were found")

    def test_aggregate_case_results_marks_checksum_mismatch(self):
        run_records = [
            {
                "success": True,
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "error": "",
                "command": ["mpirun"],
                "parsed": {
                    "time_sec": "0.100000000",
                    "checksum": "1.0",
                    "max_abs": "2.0",
                    "processes": "2",
                    "m": "128",
                    "n": "128",
                    "k": "128",
                    "seed": "20250401",
                },
            },
            {
                "success": True,
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "error": "",
                "command": ["mpirun"],
                "parsed": {
                    "time_sec": "0.200000000",
                    "checksum": "1.5",
                    "max_abs": "2.0",
                    "processes": "2",
                    "m": "128",
                    "n": "128",
                    "k": "128",
                    "seed": "20250401",
                },
            },
        ]

        aggregated = benchmark.aggregate_case_results(128, 128, 128, 2, run_records)

        self.assertEqual(aggregated["status"], "checksum_mismatch")
        self.assertFalse(aggregated["checksum_consistent"])
        self.assertEqual(aggregated["successful_runs"], 2)
        self.assertEqual(aggregated["failed_runs"], 0)


if __name__ == "__main__":
    unittest.main()
