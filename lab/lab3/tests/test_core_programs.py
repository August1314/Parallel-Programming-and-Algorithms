import subprocess
import unittest
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent.parent
BIN_DIR = LAB_DIR / "bin"


def parse_key_values(stdout: str) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in stdout.strip().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


class CoreProgramsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        subprocess.run(["make"], cwd=LAB_DIR, check=True)

    def run_program(self, name: str, *args: str) -> dict[str, str]:
        result = subprocess.run(
            [str(BIN_DIR / name), *args],
            cwd=LAB_DIR,
            text=True,
            capture_output=True,
            check=True,
        )
        return parse_key_values(result.stdout)

    def test_matmul_v1_threads_1_and_3_match(self):
        single = self.run_program("pthreads_matmul_v1", "4", "4", "4", "20250401", "1")
        multi = self.run_program("pthreads_matmul_v1", "4", "4", "4", "20250401", "3")
        self.assertEqual(single["experiment"], "pthreads_matmul")
        self.assertEqual(single["version"], "v1_row_block")
        self.assertEqual(single["checksum"], multi["checksum"])
        self.assertEqual(single["max_abs"], multi["max_abs"])

    def test_array_sum_v1_threads_1_and_4_match(self):
        single = self.run_program("pthreads_array_sum_v1", "32", "20250401", "1")
        multi = self.run_program("pthreads_array_sum_v1", "32", "20250401", "4")
        self.assertEqual(single["experiment"], "pthreads_array_sum")
        self.assertEqual(single["version"], "v1_local_sum")
        self.assertEqual(single["sum"], multi["sum"])

    def test_dump_small_matmul_contains_matrix_labels(self):
        result = subprocess.run(
            [str(BIN_DIR / "pthreads_matmul_v2_partitioned"), "2", "2", "2", "20250401", "2", "--dump"],
            cwd=LAB_DIR,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("A:", result.stdout)
        self.assertIn("B:", result.stdout)
        self.assertIn("C:", result.stdout)


if __name__ == "__main__":
    unittest.main()
