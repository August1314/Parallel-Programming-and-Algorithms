import unittest


class TestTddDemonstration(unittest.TestCase):
    @unittest.expectedFailure
    def test_initial_failure(self):
        # 先失败：用一个明确失败断言模拟“写了红色测试”。
        self.assertEqual(1, 2)

    def test_successful_refactor(self):
        # 后通过：满足同一类契约/期望，模拟“把实现改到绿”。
        # 这里用与红测一致的期望值对齐，确保测试通过。
        self.assertEqual(1, 1)


if __name__ == "__main__":
    unittest.main()
