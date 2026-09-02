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


class DeterminismCase(unittest.TestCase):
    """What `det` is a share OF (jaato #798).

    The column promises "the share of arms sharing the modal payload hash".
    It used to compute `1 / distinct_hashes`, which is a different number
    whenever the hashes are not equally frequent, and which treated an arm
    that produced no payload as though it had agreed.

    The two definitions coincide on every two-arm cell, so
    `test_determinism_from_payload_hashes` above passes under both and is
    left exactly as it was. These are the cases that separate them.
    """

    def _cell(self, *hashes, states=None):
        states = states or [PASS] * len(hashes)
        recs = [_rec("t", "s", st, payload_hash=h)
                for h, st in zip(hashes, states)]
        return build_cells(recs)[("t", "s")]

    # -- the arithmetic ------------------------------------------------

    def test_three_arms_two_agreeing_is_the_modal_share(self):
        """2 of 3 agreed. Counting distinct hashes gave 1/2 = 50%."""
        cell = self._cell("a", "a", "b")
        self.assertEqual(cell.answered, 3)
        self.assertAlmostEqual(cell.determinism, 2 / 3)

    def test_three_arms_all_agreeing(self):
        self.assertAlmostEqual(self._cell("a", "a", "a").determinism, 1.0)

    def test_three_arms_none_agreeing(self):
        """Three distinct hashes: the modal share is 1 of 3, not 1/3 by
        coincidence — this case reads the same under both definitions and
        is here to pin it rather than to discriminate."""
        self.assertAlmostEqual(self._cell("a", "b", "c").determinism, 1 / 3)

    def test_four_arms_three_agreeing(self):
        """Distinct-count would say 50%; the modal share is 75%."""
        self.assertAlmostEqual(self._cell("a", "a", "a", "b").determinism, 0.75)

    def test_the_modal_hash_wins_regardless_of_order(self):
        self.assertAlmostEqual(self._cell("b", "a", "a").determinism, 2 / 3)

    # -- what it is a share OF -----------------------------------------

    def test_an_arm_with_no_payload_is_not_an_agreeing_arm(self):
        """The shipped case: one arm answered, one died before it could.

        This printed 100%, which the footer described as "byte-identical
        across repeats" — from a single observation.
        """
        cell = self._cell(None, "a", states=[FAIL, FAIL])
        self.assertEqual(cell.exercised, 2)
        self.assertEqual(cell.answered, 1)
        self.assertIsNone(cell.determinism)

    def test_a_single_arm_cannot_agree_with_anything(self):
        cell = self._cell("a")
        self.assertEqual(cell.answered, 1)
        self.assertIsNone(cell.determinism)

    def test_no_arm_answered_is_none(self):
        cell = self._cell(None, None, states=[FAIL, FAIL])
        self.assertEqual(cell.answered, 0)
        self.assertIsNone(cell.determinism)

    def test_silent_arms_do_not_dilute_the_share(self):
        """A missing payload must not count AGAINST agreement either.

        Two arms agreed and a third never answered: the honest reading is
        "both arms that answered agreed", not 2/3. Counting it against
        would repeat the error `pass_rate` returns None to avoid.
        """
        cell = self._cell("a", "a", None, states=[PASS, PASS, FAIL])
        self.assertEqual(cell.answered, 2)
        self.assertEqual(cell.exercised, 3)
        self.assertAlmostEqual(cell.determinism, 1.0)

    # -- what the reader is shown --------------------------------------

    def test_markdown_prints_the_denominator(self):
        md = render_markdown([
            _rec("t", "s", PASS, payload_hash="a"),
            _rec("t", "s", PASS, payload_hash="a"),
            _rec("t", "s", FAIL, payload_hash=None),
        ])
        self.assertIn("100% (2 of 3)", md)

    def test_markdown_shows_an_em_dash_when_one_arm_answered(self):
        md = render_markdown([
            _rec("t", "s", FAIL, payload_hash=None),
            _rec("t", "s", FAIL, payload_hash="a"),
        ])
        self.assertIn("— (1 of 2)", md)
        # Assert on the DATA ROW, not the document: the footer legitimately
        # contains "100% = byte-identical" as part of its explanation.
        row = next(l for l in md.split("\n")
                   if l.startswith("| t |"))
        self.assertNotIn("100%", row)

    def test_footer_describes_what_is_actually_computed(self):
        md = render_markdown([_rec("t", "s", PASS, payload_hash="a")])
        self.assertIn("ANSWERING arms", md)
