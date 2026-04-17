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
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="lab3-test-results-")
        cls.results_dir = Path(cls.temp_dir.name)
        bench_env = {
            "LAB3_BENCH_THREADS": "1,2",
            "LAB3_BENCH_MATMUL_SIZES": "128,512",
            "LAB3_BENCH_ARRAY_SIZES": "1000000,4000000",
            "LAB3_BENCH_RUNS": "1",
            "LAB3_RESULTS_DIR": str(cls.results_dir),
        }
        run_script("./scripts/benchmark.py", extra_env=bench_env)
        artifact_env = {"LAB3_RESULTS_DIR": str(cls.results_dir)}
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
        expected_tables = [
            "time_table_matmul.csv",
            "speedup_table_matmul.csv",
            "efficiency_table_matmul.csv",
            "time_table_array_sum.csv",
            "speedup_table_array_sum.csv",
            "efficiency_table_array_sum.csv",
        ]
        for name in expected_tables:
            path = self.tables_dir / name
            self.assertTrue(path.exists(), name)
            self.assertGreater(path.stat().st_size, 0, name)

    def test_benchmark_json_has_required_keys(self):
        data = json.loads(self.benchmark_json.read_text(encoding="utf-8"))
        self.assertTrue(data)
        sample = data[0]
        for key in ["experiment", "version", "threads", "avg_time_sec", "status", "runs"]:
            self.assertIn(key, sample)

    def test_matmul_and_array_rows_have_required_fields(self):
        data = json.loads(self.benchmark_json.read_text(encoding="utf-8"))
        matmul = next(item for item in data if item["experiment"] == "pthreads_matmul")
        array_sum = next(item for item in data if item["experiment"] == "pthreads_array_sum")
        for key in ["m", "n", "k", "checksum", "max_abs"]:
            self.assertIn(key, matmul)
        for key in ["length", "sum"]:
            self.assertIn(key, array_sum)

    def test_plot_outputs_exist(self):
        expected = [
            "runtime_matmul_v1_row_block.png",
            "runtime_matmul_v2_cyclic_rows.png",
            "speedup_matmul_v1_row_block.png",
            "speedup_matmul_v2_cyclic_rows.png",
            "efficiency_matmul_v1_row_block.png",
            "efficiency_matmul_v2_cyclic_rows.png",
            "heatmap_matmul_v1_row_block.png",
            "heatmap_matmul_v2_cyclic_rows.png",
            "version_comparison_matmul.png",
            "runtime_array_sum_v1_local_sum.png",
            "runtime_array_sum_v2_mutex_shared_sum.png",
            "speedup_array_sum_v1_local_sum.png",
            "speedup_array_sum_v2_mutex_shared_sum.png",
            "efficiency_array_sum_v1_local_sum.png",
            "efficiency_array_sum_v2_mutex_shared_sum.png",
            "version_comparison_array_sum.png",
        ]
        for name in expected:
            self.assertTrue((self.figures_dir / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
