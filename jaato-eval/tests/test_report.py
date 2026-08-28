"""Pivot arithmetic: BLOCKED never in a denominator, always visible."""
import unittest

from jaato_eval.report import build_cells, render_markdown
from jaato_eval.verdict import BLOCKED, FAIL, PASS


def _rec(task, pset, state, cost=None, blocked_reason=None, payload_hash=None):
    return {"task_id": task, "profile_set": pset, "state": state,
            "usage": {"cost_usd": cost, "spend_total_tokens": 100},
            "duration_seconds": 1.0, "turns": 2,
            "blocked_reason": blocked_reason, "payload_hash": payload_hash,
            "verdicts": []}


class ReportCase(unittest.TestCase):
    def test_pass_rate_excludes_blocked(self):
        cells = build_cells([
            _rec("t", "cheap", PASS), _rec("t", "cheap", FAIL),
            _rec("t", "cheap", BLOCKED, blocked_reason="429"),
        ])
        cell = cells[("t", "cheap")]
        self.assertEqual(cell.exercised, 2)
        self.assertAlmostEqual(cell.pass_rate, 0.5)
        self.assertEqual(cell.blocked, 1)

    def test_all_blocked_pass_rate_is_none_not_zero(self):
        """Zero would say 'it always failed'; the truth is 'we never
        found out', and the two must not print the same."""
        cells = build_cells([_rec("t", "s", BLOCKED, blocked_reason="r")])
        self.assertIsNone(cells[("t", "s")].pass_rate)
        self.assertIn("—", render_markdown([_rec("t", "s", BLOCKED, blocked_reason="r")]))

    def test_cost_none_when_unreported(self):
        cells = build_cells([_rec("t", "s", PASS, cost=None)])
        self.assertIsNone(cells[("t", "s")].cost_usd)

    def test_cost_summed_when_reported(self):
        cells = build_cells([_rec("t", "s", PASS, cost=0.01),
                             _rec("t", "s", PASS, cost=0.02)])
        self.assertAlmostEqual(cells[("t", "s")].cost_usd, 0.03)

    def test_determinism_from_payload_hashes(self):
        same = build_cells([_rec("t", "s", PASS, payload_hash="a"),
                            _rec("t", "s", PASS, payload_hash="a")])
        self.assertAlmostEqual(same[("t", "s")].determinism, 1.0)
        split = build_cells([_rec("t", "s", PASS, payload_hash="a"),
                             _rec("t", "s", PASS, payload_hash="b")])
        self.assertAlmostEqual(split[("t", "s")].determinism, 0.5)

    def test_blocked_reasons_surface_in_markdown(self):
        md = render_markdown([_rec("t", "s", BLOCKED, blocked_reason="toolchain absent")])
        self.assertIn("Blocked — nothing was exercised", md)
        self.assertIn("toolchain absent", md)

    def test_cost_dash_is_explained(self):
        md = render_markdown([_rec("t", "s", PASS)])
        self.assertIn("does not mean free", md)


if __name__ == "__main__":
    unittest.main()
