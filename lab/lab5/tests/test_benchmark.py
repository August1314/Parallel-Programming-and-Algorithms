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
        subprocess.run(["chmod", "+x", "./scripts/docker_build_image.sh", "./scripts/docker_make.sh", "./scripts/run_openmp_matmul.sh", "./scripts/run_parallel_for_matmul.sh"], cwd=LAB_DIR, check=True)
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="lab5-test-results-")
        cls.results_dir = Path(cls.temp_dir.name)
        bench_env = {
            "LAB5_BENCH_THREADS": "1,2",
            "LAB5_BENCH_SIZES": "512",
            "LAB5_BENCH_RUNS": "1",
            "LAB5_RESULTS_DIR": str(cls.results_dir),
        }
        run_script("./scripts/benchmark.py", extra_env=bench_env)
        artifact_env = {"LAB5_RESULTS_DIR": str(cls.results_dir)}
        run_script("./scripts/plot.py", extra_env=artifact_env)
        run_script("./scripts/export_report_assets.py", extra_env=artifact_env)

        cls.benchmark_json = cls.results_dir / "benchmark_results.json"
        cls.summary_csv = cls.results_dir / "summary_by_version.csv"
        cls.figures_dir = cls.results_dir / "figures"
        cls.tables_dir = cls.results_dir / "tables"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_required_files_exist_and_nonempty(self):
        self.assertTrue(self.benchmark_json.exists())
        self.assertTrue(self.summary_csv.exists())
        for name in [
            "time_table_matmul.csv",
            "speedup_table_matmul.csv",
            "efficiency_table_matmul.csv",
            "openmp_schedule_table.csv",
        ]:
            path = self.tables_dir / name
            self.assertTrue(path.exists(), name)
            self.assertGreater(path.stat().st_size, 0, name)

    def test_benchmark_json_has_required_keys(self):
        data = json.loads(self.benchmark_json.read_text(encoding="utf-8"))
        self.assertTrue(data)
        sample = data[0]
        for key in ["experiment", "backend", "version", "threads", "avg_time_sec", "status", "runs", "baseline_checksum_match"]:
            self.assertIn(key, sample)

    def test_plot_outputs_exist(self):
        expected = [
            "runtime_openmp_default.png",
            "runtime_openmp_static1.png",
            "runtime_openmp_dynamic1.png",
            "runtime_parallel_for_row_block.png",
            "speedup_openmp_default.png",
            "speedup_openmp_static1.png",
            "speedup_openmp_dynamic1.png",
            "speedup_parallel_for_row_block.png",
            "efficiency_openmp_default.png",
            "efficiency_openmp_static1.png",
            "efficiency_openmp_dynamic1.png",
            "efficiency_parallel_for_row_block.png",
            "heatmap_openmp_default.png",
            "heatmap_openmp_static1.png",
            "heatmap_openmp_dynamic1.png",
            "heatmap_parallel_for_row_block.png",
            "openmp_schedule_comparison.png",
            "version_comparison_all.png",
        ]
        for name in expected:
            self.assertTrue((self.figures_dir / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
