import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent.parent


def run_script(relative_path: str, extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    subprocess.run(
        [sys.executable, relative_path],
        cwd=LAB_DIR,
        env=env,
        check=True,
    )


class BenchmarkArtifactsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        subprocess.run(["make"], cwd=LAB_DIR, check=True)
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="lab2-test-results-")
        cls.results_dir = Path(cls.temp_dir.name)
        bench_env = {
            "LAB2_BENCH_PROCS": "1,2",
            "LAB2_BENCH_SIZES": "128,512",
            "LAB2_BENCH_RUNS": "1",
            "LAB2_RESULTS_DIR": str(cls.results_dir),
        }
        run_script("./scripts/benchmark.py", extra_env=bench_env)
        artifact_env = {"LAB2_RESULTS_DIR": str(cls.results_dir)}
        run_script("./scripts/plot.py", extra_env=artifact_env)
        run_script("./scripts/export_report_assets.py", extra_env=artifact_env)

        cls.benchmark_json = cls.results_dir / "benchmark_results.json"
        cls.summary_csv = cls.results_dir / "summary_by_version.csv"
        cls.figures_dir = cls.results_dir / "figures"
        cls.tables_dir = cls.results_dir / "tables"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_required_files_exist(self):
        self.assertTrue(self.benchmark_json.exists())
        self.assertTrue(self.summary_csv.exists())
        self.assertTrue((self.tables_dir / "time_table.csv").exists())
        self.assertTrue((self.tables_dir / "speedup_table.csv").exists())
        self.assertTrue((self.tables_dir / "efficiency_table.csv").exists())

    def test_benchmark_json_has_required_keys(self):
        data = json.loads(self.benchmark_json.read_text(encoding="utf-8"))
        self.assertTrue(data)
        sample = data[0]
        for key in ["version", "processes", "matrix_size", "avg_time_sec", "status", "runs", "checksums", "baseline_checksum_match"]:
            self.assertIn(key, sample)

    def test_plot_outputs_exist(self):
        expected = [
            "runtime_mpi_v1_p2p.png",
            "runtime_mpi_v2_collective.png",
            "speedup_mpi_v1_p2p.png",
            "speedup_mpi_v2_collective.png",
            "efficiency_mpi_v1_p2p.png",
            "efficiency_mpi_v2_collective.png",
            "version_comparison.png",
            "heatmap_mpi_v1_p2p.png",
            "heatmap_mpi_v2_collective.png",
        ]
        for name in expected:
            self.assertTrue((self.figures_dir / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
