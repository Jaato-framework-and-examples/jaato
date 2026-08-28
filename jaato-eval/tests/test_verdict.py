"""The vendored verdict module must keep the semantics it was lifted for.

These are the executed comparison that makes the vendoring survivable —
without them the copy in ``jaato_eval/verdict.py`` is a second copy of a
fact that cannot fail, which is exactly what its provenance note warns
about.
"""
import unittest

from jaato_eval.verdict import BLOCKED, FAIL, PASS, STATES, Report, Verdict


class TestVerdict(unittest.TestCase):
    def test_state_set_is_exactly_three(self):
        self.assertEqual(set(STATES), {PASS, FAIL, BLOCKED})

    def test_unknown_state_rejected(self):
        with self.assertRaises(ValueError):
            Verdict(grader_id="g", claim="c", state="SKIPPED")

    def test_blocked_without_reason_rejected(self):
        """A BLOCKED verdict that does not say what was absent is a
        silent skip; the constructor must refuse it."""
        with self.assertRaises(ValueError):
            Verdict(grader_id="g", claim="c", state=BLOCKED)

    def test_blocked_with_reason_accepted(self):
        v = Verdict(grader_id="g", claim="c", state=BLOCKED,
                    blocked_reason="toolchain absent")
        self.assertIn("toolchain absent", v.render())


class TestReport(unittest.TestCase):
    def _v(self, state, **kw):
        if state == BLOCKED:
            kw.setdefault("blocked_reason", "r")
        return Verdict(grader_id="g", claim="c", state=state, **kw)

    def test_exit_codes_are_distinct(self):
        self.assertEqual(Report([self._v(PASS)]).exit_code(), 0)
        self.assertEqual(Report([self._v(FAIL)]).exit_code(), 1)
        self.assertEqual(Report([self._v(BLOCKED)]).exit_code(), 2)

    def test_fail_outranks_blocked(self):
        r = Report([self._v(BLOCKED), self._v(FAIL)])
        self.assertEqual(r.state(), FAIL)
        self.assertEqual(r.exit_code(), 1)

    def test_blocked_outranks_pass(self):
        """One grader that could not run means the claim is not
        established, however many others passed."""
        r = Report([self._v(PASS), self._v(BLOCKED)])
        self.assertEqual(r.state(), BLOCKED)
        self.assertEqual(r.exit_code(), 2)

    def test_empty_report_is_blocked_not_pass(self):
        """Nothing ran is not success — the vacuous pass, refused."""
        r = Report([])
        self.assertEqual(r.state(), BLOCKED)
        self.assertEqual(r.exit_code(), 2)


if __name__ == "__main__":
    unittest.main()
