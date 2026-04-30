import subprocess
import unittest
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent.parent


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
        subprocess.run(["chmod", "+x", "./scripts/docker_build_image.sh", "./scripts/docker_make.sh", "./scripts/run_openmp_matmul.sh", "./scripts/run_parallel_for_matmul.sh"], cwd=LAB_DIR, check=True)
        subprocess.run(["./scripts/docker_make.sh"], cwd=LAB_DIR, check=True)

    def run_program(self, command: list[str]) -> dict[str, str]:
        result = subprocess.run(
            command,
            cwd=LAB_DIR,
            text=True,
            capture_output=True,
            check=True,
        )
        return parse_key_values(result.stdout)

    def test_all_versions_match_same_checksum_at_threads_1(self):
        outputs = [
            self.run_program(["./scripts/run_openmp_matmul.sh", "default", "4", "4", "4", "20250401", "1"]),
            self.run_program(["./scripts/run_openmp_matmul.sh", "static1", "4", "4", "4", "20250401", "1"]),
            self.run_program(["./scripts/run_openmp_matmul.sh", "dynamic1", "4", "4", "4", "20250401", "1"]),
            self.run_program(["./scripts/run_parallel_for_matmul.sh", "4", "4", "4", "20250401", "1"]),
        ]
        checksums = {item["checksum"] for item in outputs}
        max_abs_values = {item["max_abs"] for item in outputs}
        self.assertEqual(len(checksums), 1)
        self.assertEqual(len(max_abs_values), 1)

    def test_threads_1_and_4_match_for_parallel_for(self):
        single = self.run_program(["./scripts/run_parallel_for_matmul.sh", "4", "4", "4", "20250401", "1"])
        multi = self.run_program(["./scripts/run_parallel_for_matmul.sh", "4", "4", "4", "20250401", "4"])
        self.assertEqual(single["checksum"], multi["checksum"])
        self.assertEqual(single["max_abs"], multi["max_abs"])

    def test_dump_output_contains_matrix_labels(self):
        result = subprocess.run(
            ["./scripts/run_openmp_matmul.sh", "default", "2", "2", "2", "20250401", "2", "--dump"],
            cwd=LAB_DIR,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("A:", result.stdout)
        self.assertIn("B:", result.stdout)
        self.assertIn("C:", result.stdout)

    def test_parallel_for_binary_links_shared_library(self):
        result = subprocess.run(
            [
                "docker",
                "run",
                "--platform",
                "linux/arm64",
                "--rm",
                "-v",
                f"{LAB_DIR.parent.parent}:/workspace",
                "-w",
                "/workspace/lab/lab5",
                "parallel-programming-lab5:latest",
                "ldd",
                "./bin/parallel_for_matmul",
            ],
            cwd=LAB_DIR,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("libparallel_for.so", result.stdout)


if __name__ == "__main__":
    unittest.main()
