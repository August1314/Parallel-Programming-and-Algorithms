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

    def test_pthreads_schedules_match_openmp(self):
        outputs = [
            self.run_program(["./scripts/run_openmp_heated_plate.sh", "32", "32", "1.0", "2"]),
            self.run_program(["./scripts/run_pthreads_heated_plate.sh", "block", "32", "32", "1.0", "2", "4"]),
            self.run_program(["./scripts/run_pthreads_heated_plate.sh", "cyclic", "32", "32", "1.0", "2", "4"]),
            self.run_program(["./scripts/run_pthreads_heated_plate.sh", "dynamic", "32", "32", "1.0", "2", "4"]),
        ]
        checksums = {item["checksum"] for item in outputs}
        iterations = {item["iterations"] for item in outputs}
        self.assertEqual(len(checksums), 1)
        self.assertEqual(len(iterations), 1)

    def test_threads_1_and_4_match_for_dynamic_schedule(self):
        single = self.run_program(["./scripts/run_pthreads_heated_plate.sh", "dynamic", "32", "32", "1.0", "1", "4"])
        multi = self.run_program(["./scripts/run_pthreads_heated_plate.sh", "dynamic", "32", "32", "1.0", "4", "4"])
        self.assertEqual(single["checksum"], multi["checksum"])
        self.assertEqual(single["iterations"], multi["iterations"])

    def test_dump_output_contains_w_matrix(self):
        result = subprocess.run(
            ["./scripts/run_pthreads_heated_plate.sh", "block", "5", "5", "10.0", "2", "1", "--dump"],
            cwd=LAB_DIR,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("W:", result.stdout)

    def test_pthreads_binary_links_shared_library(self):
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
                "/workspace/lab/lab6",
                "parallel-programming-lab6:latest",
                "ldd",
                "./bin/heated_plate_pthreads",
            ],
            cwd=LAB_DIR,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("libparallel_for.so", result.stdout)


if __name__ == "__main__":
    unittest.main()
