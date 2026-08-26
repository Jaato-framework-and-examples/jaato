"""Matrix expansion."""
import unittest

from jaato_eval.sweep import build_matrix, pool_size_advice


class _Task:
    def __init__(self, task_id, repeats=1):
        self.task_id = task_id
        self.repeats = repeats


class SweepCase(unittest.TestCase):
    def test_cartesian_product(self):
        arms = build_matrix([_Task("a", 2), _Task("b", 1)], ["cheap", "frontier"])
        self.assertEqual(len(arms), 2 * 2 + 1 * 2)
        self.assertIn("a@cheap#1", [x.arm_id for x in arms])

    def test_no_profile_sets_uses_task_default(self):
        arms = build_matrix([_Task("a", 3)], [])
        self.assertEqual(len(arms), 3)
        self.assertEqual(arms[0].profile_set, None)
        self.assertEqual(arms[0].arm_id, "a@default#0")

    def test_pool_advice_names_the_number(self):
        self.assertIn("＞=" .replace("＞", ">"), pool_size_advice(4).replace(">=", ">="))
        self.assertIn("4", pool_size_advice(4))


if __name__ == "__main__":
    unittest.main()
