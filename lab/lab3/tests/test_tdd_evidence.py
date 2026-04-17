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


class TDDEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        subprocess.run(["make"], cwd=LAB_DIR, check=True)

    def run_array_sum_v1(self) -> dict[str, str]:
        result = subprocess.run(
            [str(BIN_DIR / "pthreads_array_sum_v1"), "32", "20250401", "4"],
            cwd=LAB_DIR,
            text=True,
            capture_output=True,
            check=True,
        )
        return parse_key_values(result.stdout)

    def assert_array_sum_v1_contract(self, data: dict[str, str]) -> None:
        self.assertEqual(data["experiment"], "pthreads_array_sum")
        self.assertEqual(data["version"], "v1_local_sum")
        self.assertEqual(data["threads"], "4")
        self.assertIn("sum", data)

    def test_red_contract_rejects_placeholder_output(self):
        placeholder = {
            "experiment": "pthreads_array_sum",
            "version": "todo",
            "threads": "4",
        }
        with self.assertRaises(AssertionError):
            self.assert_array_sum_v1_contract(placeholder)

    def test_green_contract_accepts_current_binary_output(self):
        self.assert_array_sum_v1_contract(self.run_array_sum_v1())


if __name__ == "__main__":
    unittest.main()
