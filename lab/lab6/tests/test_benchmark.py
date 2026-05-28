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
        subprocess.run(
            [
                "chmod",
                "+x",
                "./scripts/docker_build_image.sh",
                "./scripts/docker_make.sh",
                "./scripts/run_openmp_heated_plate.sh",
                "./scripts/run_pthreads_heated_plate.sh",
            ],
            cwd=LAB_DIR,
            check=True,
        )
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="lab6-test-results-")
        cls.results_dir = Path(cls.temp_dir.name)
        bench_env = {
            "LAB6_BENCH_THREADS": "1,2",
            "LAB6_BENCH_SIZES": "32x32",
            "LAB6_BENCH_RUNS": "1",
            "LAB6_BENCH_EPSILON": "1.0",
            "LAB6_BENCH_CHUNK_SIZE": "4",
            "LAB6_RESULTS_DIR": str(cls.results_dir),
        }
        run_script("./scripts/benchmark.py", extra_env=bench_env)
        artifact_env = {"LAB6_RESULTS_DIR": str(cls.results_dir)}
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
            "time_table_heated_plate.csv",
            "speedup_table_heated_plate.csv",
            "efficiency_table_heated_plate.csv",
            "pthreads_schedule_table.csv",
        ]:
            path = self.tables_dir / name
            self.assertTrue(path.exists(), name)
            self.assertGreater(path.stat().st_size, 0, name)

    def test_benchmark_json_has_required_keys(self):
        data = json.loads(self.benchmark_json.read_text(encoding="utf-8"))
        self.assertTrue(data)
        self.assertEqual(8, len(data))
        sample = data[0]
        required_keys = [
            "experiment",
            "backend",
            "version",
            "threads",
            "avg_time_sec",
            "status",
            "runs",
            "openmp_reference_match",
            "thread1_checksum_match",
        ]
        for key in required_keys:
            self.assertIn(key, sample)
        for row in data:
            with self.subTest(version=row.get("version"), threads=row.get("threads")):
                for key in required_keys:
                    self.assertIn(key, row)
                self.assertEqual("ok", row["status"])
                self.assertIn(row["threads"], [1, 2])

                if row["backend"] == "openmp":
                    self.assertIs(row["openmp_reference_match"], True)
                if row["backend"] == "pthreads_parallel_for":
                    self.assertIs(row["openmp_reference_match"], True)
                self.assertIs(row["thread1_checksum_match"], True)

    def test_plot_outputs_exist(self):
        expected = [
            "runtime_openmp_original.png",
            "runtime_pthreads_parallel_for_block.png",
            "runtime_pthreads_parallel_for_cyclic.png",
            "runtime_pthreads_parallel_for_dynamic.png",
            "speedup_openmp_original.png",
            "speedup_pthreads_parallel_for_block.png",
            "speedup_pthreads_parallel_for_cyclic.png",
            "speedup_pthreads_parallel_for_dynamic.png",
            "efficiency_openmp_original.png",
            "efficiency_pthreads_parallel_for_block.png",
            "efficiency_pthreads_parallel_for_cyclic.png",
            "efficiency_pthreads_parallel_for_dynamic.png",
            "heatmap_openmp_original.png",
            "heatmap_pthreads_parallel_for_block.png",
            "heatmap_pthreads_parallel_for_cyclic.png",
            "heatmap_pthreads_parallel_for_dynamic.png",
            "pthreads_schedule_comparison.png",
            "openmp_vs_pthreads.png",
        ]
        for name in expected:
            self.assertTrue((self.figures_dir / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
