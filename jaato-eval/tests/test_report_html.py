"""The per-arm document — jaato #777.

``render_markdown`` answers *which configuration won*.  These pin the
other artefact: one row per arm, shaped like a provider console's session
list, carrying the session id that joins it to the provider's own record.

The recurring assertion is that nothing prints as a measurement it is
not.  ``—`` is the only rendering of an unestablished value, and the
document says so in prose, because a reader who takes ``cost —`` for
``free`` or ``nudges —`` for ``none fired`` draws the opposite conclusion
from the sweep.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jaato_eval.report_html import (UNKNOWN, ReportDependencyError,
                                    budget_cell, finish_cell, grader_ids,
                                    nudges_cell, render_html, row_cells,
                                    write_html, write_pdf)
from jaato_eval.verdict import BLOCKED, FAIL, PASS


def _rec(**overrides):
    record = {
        "arm_id": "t/echo@openrouter_gpt5mini#0",
        "task_id": "t/echo",
        "profile_set": "openrouter_gpt5mini",
        "repeat": 0,
        "state": PASS,
        "blocked_reason": None,
        "error": None,
        "verdicts": [{"grader_id": "script:grep", "state": PASS}],
        "usage": {"cost_usd": 2.03, "spend_total_tokens": 4200},
        "duration_seconds": 61.5,
        "turns": 7,
        "finish_reason": "tool_use",
        "payload_hash": "abcdef0123456789" * 4,
        "session_id": "sess-19-c",
        "model": "openai/gpt-5-mini",
        "provider": "openrouter",
        "upstream_provider": None,
        "native_finish_reason": None,
        "completion_nudges": None,
        "budget_ceiling": None,
        "pool_limits": {"usd": 6.0},
        "pool_on_arrival": {"declared": True, "limits": {"usd": 6.0},
                            "usage_fraction": 0.635},
    }
    record.update(overrides)
    return record


class BudgetCellCase(unittest.TestCase):
    """Spend is only legible against what was allowed."""

    def test_pool_spend_shows_what_was_already_gone_on_arrival(self):
        """The run-19 arm.  From its own row a `budget_exhausted` kill
        reads as a model failure; this is the same arm described."""
        cell = budget_cell(_rec(usage={"cost_usd": 2.03}))
        self.assertEqual(cell, "$2.0300 / pool $6.0000 (64% consumed on arrival)")

    def test_a_profile_ceiling_names_itself_as_the_gate(self):
        """A session with its own budget_control does NOT draw on the
        pool; a ceiling shown without naming its pot reads as a pool that
        failed to bind."""
        cell = budget_cell(_rec(budget_ceiling={"usd": 0.5}))
        self.assertEqual(cell, "$2.0300 / own $0.5000")

    def test_no_ceiling_anywhere_is_a_dash(self):
        self.assertEqual(budget_cell(_rec(pool_limits=None)), UNKNOWN)

    def test_an_unread_pool_snapshot_does_not_claim_a_full_pool(self):
        """`declared: false` and 'we did not read it' both mean no
        fraction — 0% would assert a pool nobody saw."""
        cell = budget_cell(_rec(pool_on_arrival=None))
        self.assertEqual(cell, "$2.0300 / pool $6.0000")
        cell = budget_cell(_rec(pool_on_arrival={"declared": False}))
        self.assertEqual(cell, "$2.0300 / pool $6.0000")


class ColumnCase(unittest.TestCase):
    def test_finish_reason_carries_the_upstreams_own_word(self):
        """OpenRouter's normalisation is lossy exactly where it matters —
        Gemini's MALFORMED_FUNCTION_CALL arrives as a generic error."""
        self.assertEqual(
            finish_cell(_rec(finish_reason="error",
                             native_finish_reason="MALFORMED_FUNCTION_CALL")),
            "error (MALFORMED_FUNCTION_CALL)")

    def test_without_a_native_reason_no_empty_parenthetical(self):
        self.assertEqual(finish_cell(_rec(finish_reason="stop")), "stop")

    def test_nudges_show_the_ceiling_not_just_the_count(self):
        """An arm at 2/2 is one nudge from BLOCKED, and '2' does not say
        that."""
        self.assertEqual(nudges_cell(_rec(completion_nudges=2)), "2/2")
        self.assertEqual(nudges_cell(_rec(completion_nudges=0)), "0/2")

    def test_an_uncounted_nudge_is_not_zero(self):
        self.assertEqual(nudges_cell(_rec(completion_nudges=None)), UNKNOWN)

    def test_a_row_carries_the_session_id_verbatim(self):
        """The join key has to survive rendering unmangled — a truncated
        id does not find the session in the provider's console."""
        cells = row_cells(_rec(), ["script:grep"])
        self.assertIn("sess-19-c", cells)

    def test_one_column_per_grader_in_manifest_order(self):
        records = [_rec(verdicts=[{"grader_id": "script:grep", "state": PASS},
                                  {"grader_id": "judge:rubric", "state": FAIL}])]
        self.assertEqual(grader_ids(records), ["script:grep", "judge:rubric"])

    def test_a_grader_that_did_not_run_on_this_arm_is_unknown(self):
        """Not FAIL, and not blank — the arm has no verdict from it."""
        cells = row_cells(_rec(verdicts=[]), ["judge:rubric"])
        self.assertEqual(cells[-1], UNKNOWN)

    def test_blocked_reason_and_an_ungraded_sign_off_are_both_shown(self):
        """They are different facts: nothing to grade, versus evidence
        produced and a terminal anyway (jaato #773)."""
        cells = row_cells(_rec(state=BLOCKED, blocked_reason="429 from provider",
                               error="NudgeExhausted"), [])
        self.assertIn("429 from provider", cells[-1])
        self.assertIn("sign-off", cells[-1])


class DocumentCase(unittest.TestCase):
    def test_the_document_carries_both_the_pivot_and_the_per_arm_table(self):
        """Complementary, not a replacement: a document with only the
        per-arm table makes the reader re-derive which set won."""
        html = render_html([_rec()])
        self.assertIn("Which configuration won", html)
        self.assertIn("t/echo", html)
        self.assertIn("sess-19-c", html)

    def test_it_is_self_contained(self):
        """Sweep artefacts get emailed, committed and opened from file://
        on machines with no network."""
        html = render_html([_rec()])
        self.assertNotIn("<script", html)
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)
        self.assertIn("<style>", html)

    def test_it_carries_print_css_so_a_browser_is_the_pdf_renderer(self):
        self.assertIn("@media print", render_html([_rec()]))
        self.assertIn("@page", render_html([_rec()]))

    def test_the_dash_is_explained_rather_than_left_to_the_reader(self):
        html = render_html([_rec()])
        self.assertIn("does not mean free", html)
        self.assertIn("not that none fired", html)

    def test_an_empty_results_file_says_so(self):
        self.assertIn("No results.", render_html([]))

    def test_a_record_from_before_this_feature_renders_as_unknown(self):
        """An old results.jsonl has none of the new keys; every one of
        them must print as unestablished rather than raise."""
        legacy = {"task_id": "t", "profile_set": "s", "repeat": 0,
                  "state": PASS, "usage": {}, "verdicts": []}
        html = render_html([legacy])
        self.assertIn(UNKNOWN, html)

    def test_a_grader_id_carrying_a_quote_cannot_break_the_markup(self):
        """A `script` grader's identifier IS the command the manifest
        declared, quotes and ampersands included — it is data reaching an
        attribute, not markup."""
        record = _rec(verdicts=[{"grader_id": 'script:grep -qx "READY" a && x',
                                 "state": PASS}])
        html = render_html([record])
        self.assertNotIn('title="script:grep -qx "READY"', html)
        self.assertIn("&quot;READY&quot;", html)
        self.assertIn("&amp;&amp;", html)

    def test_html_content_is_escaped(self):
        """Blocked reasons quote provider prose, which is not trusted
        markup."""
        html = render_html([_rec(state=BLOCKED,
                                 blocked_reason="<script>alert(1)</script>")])
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertIn("&lt;script&gt;", html)


class OutputCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_write_html_needs_no_optional_dependency(self):
        path = write_html([_rec()], self.root / "nested" / "report.html")
        self.assertTrue(path.is_file())
        self.assertIn("sess-19-c", path.read_text(encoding="utf-8"))

    def test_pdf_without_the_extra_fails_loudly_with_the_install_line(self):
        """Not a silent HTML-only fallback: a sweep run unattended asked
        for a PDF and must not quietly produce something else."""
        try:
            import weasyprint  # noqa: F401
        except ImportError:
            with self.assertRaises(ReportDependencyError) as caught:
                write_pdf([_rec()], self.root / "report.pdf")
            self.assertIn("jaato-eval[report]", str(caught.exception))
        else:
            self.assertTrue(
                write_pdf([_rec()], self.root / "report.pdf").is_file())


if __name__ == "__main__":
    unittest.main()
