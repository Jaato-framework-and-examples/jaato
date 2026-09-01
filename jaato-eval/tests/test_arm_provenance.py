"""What each arm records about ITSELF — jaato #777.

The pivot answers which configuration won.  These pin the other half:
that an arm's row carries enough to go and look the arm up — the session
id the provider console groups by, the model and provider the daemon
actually bound, the nudges it drew, and the two budget ceilings it ran
under.

Every one of them is also a test that UNKNOWN stays unknown.  A ceiling
this engine could not resolve, a nudge count it could not read and a
model a cut arm never learned must not print as ``0`` or as a plausible
guess, because a reader acts on those differently.
"""
from __future__ import annotations

import textwrap
import unittest

from jaato_eval.verdict import BLOCKED, PASS

from tests.test_runner_integration import RunnerHarness

#: A session log shaped like the daemon's: the formatter is
#: ``%(asctime)s [%(levelname)s] %(name)s: %(message)s``, and the nudge is
#: announced by ``JaatoServer._trace``, which is ``logger.debug``.
NUDGE_LOG = textwrap.dedent("""\
    2026-08-30 10:00:00 [DEBUG] server.core: COMPLETION_NUDGE: agent ended \
its loop without signal_completion (nudge 1/2) — re-prompting
    2026-08-30 10:01:00 [DEBUG] server.core: COMPLETION_NUDGE: agent ended \
its loop without signal_completion (nudge 2/2) — re-prompting
""")

#: The same session, on a daemon logging at INFO.  The file exists and
#: carries no DEBUG record — which is "not recorded", not "none fired".
INFO_ONLY_LOG = "2026-08-30 10:00:00 [INFO] server.core: Session created\n"


class BindingCase(RunnerHarness):
    """The session id, model and provider land on the result."""

    def test_session_id_reaches_the_result(self):
        """The join key.  OpenRouter's console groups by exactly this id,
        so an arm without it cannot be joined to the provider's record —
        which is what the runner used to discard."""
        result = self._run({"writes": "READY\n", "payload": {"done": True}})
        self.assertEqual(result.state, PASS)
        self.assertEqual(result.session_id, "sid-1")

    def test_bound_model_and_provider_not_the_set_name(self):
        """``profile_set`` is a directory someone named; this is data."""
        result = self._run({"writes": "READY\n",
                            "model_name": "google/gemini-2.5-flash",
                            "model_provider": "openrouter"})
        self.assertEqual(result.model, "google/gemini-2.5-flash")
        self.assertEqual(result.provider, "openrouter")
        self.assertEqual(result.spec.profile_set, "cheap")

    def test_a_blocked_arm_still_records_which_session_it_was(self):
        """The arm a reader most needs to look up is the one that failed."""
        result = self._run({"agent_error": "ProviderError"})
        self.assertEqual(result.state, BLOCKED)
        self.assertEqual(result.session_id, "sid-1")
        self.assertEqual(result.provider, "openrouter")

    def test_an_unreported_binding_stays_none(self):
        """A cut arm may never have received its SessionInfoEvent, and
        naming a model it MIGHT have bound is worse than a blank."""
        result = self._run({"writes": "READY\n",
                            "model_name": "", "model_provider": ""})
        self.assertIsNone(result.model)
        self.assertIsNone(result.provider)


class NudgeCase(RunnerHarness):
    """Completion nudges, counted rather than grepped for afterwards."""

    def test_nudges_are_counted_from_the_session_log(self):
        """Three of one sweep's BLOCKED arms were explained by nothing
        else, and the count is trace-only — so it is read from the log
        the daemon writes into the arm's own workspace."""
        result = self._run({"writes": "READY\n", "session_log": NUDGE_LOG})
        self.assertEqual(result.completion_nudges, 2)

    def test_a_log_with_no_debug_records_reports_unknown_not_zero(self):
        """A daemon logging at INFO writes the file and never writes a
        nudge line.  Reporting that as 'no nudges' would be a fact this
        engine made up."""
        result = self._run({"writes": "READY\n", "session_log": INFO_ONLY_LOG})
        self.assertIsNone(result.completion_nudges)

    def test_a_debug_log_with_no_nudge_line_reports_zero(self):
        """Debug records present and no nudge among them IS evidence."""
        result = self._run({"writes": "READY\n",
                            "session_log": "x [DEBUG] server.core: hello\n"})
        self.assertEqual(result.completion_nudges, 0)

    def test_completion_gap_alone_puts_the_arm_at_the_ceiling(self):
        """The framework sets it exactly when it asked twice and gave up,
        so an arm carrying it is at the ceiling whether a log survives or
        not."""
        result = self._run({"writes": "READY\n",
                            "completion_gap": "not_signalled_after_nudges"})
        self.assertEqual(result.completion_nudges, 2)

    def test_no_log_at_all_is_unknown(self):
        result = self._run({"writes": "READY\n"})
        self.assertIsNone(result.completion_nudges)


CEILINGED = """
id: t/echo
environment:
  fixture: fixture
  config_root: cfg
input:
  prompt: write answer.txt containing READY
harness:
  profile: worker
budget:
  usd: 6.0
graders:
  - kind: script
    run: "grep -qx READY answer.txt"
"""


class BudgetCase(RunnerHarness):
    """Both gates, and the pool's state when the arm arrived."""

    task_yaml = CEILINGED

    def _profiles(self, base: str, in_set: str) -> None:
        profiles = self.root / "cfg" / "profiles" / "cheap"
        profiles.mkdir(parents=True)
        (profiles.parent / "_base_worker.yaml").write_text(base)
        (profiles / "worker.yaml").write_text(in_set)

    def test_the_arms_own_ceiling_is_resolved_from_the_profile_it_bound(self):
        self._profiles("name: _base_worker\n",
                       "name: worker\ninherits: [_base_worker]\n"
                       "budget_control:\n  limits:\n    usd: 2.0\n")
        result = self._run({"writes": "READY\n"})
        self.assertEqual(result.budget_ceiling, {"usd": 2.0})

    def test_a_child_may_only_tighten_an_inherited_ceiling(self):
        """Min-wins, the direction ``shared.budget_control.merge_limits``
        takes: child-replaces-parent on a resource ceiling would be an
        escape hatch, and a report printing the larger number would
        describe a ceiling the daemon never enforced."""
        self._profiles(
            "name: _base_worker\nbudget_control:\n  limits:\n"
            "    usd: 1.0\n    turns: 40\n",
            "name: worker\ninherits: [_base_worker]\n"
            "budget_control:\n  limits:\n    usd: 5.0\n")
        result = self._run({"writes": "READY\n"})
        self.assertEqual(result.budget_ceiling, {"usd": 1.0, "turns": 40.0})

    def test_no_profile_on_disk_is_unknown_not_unbudgeted(self):
        result = self._run({"writes": "READY\n"})
        self.assertIsNone(result.budget_ceiling)

    def test_the_task_pool_travels_with_the_arm(self):
        result = self._run({"writes": "READY\n"})
        self.assertEqual(result.pool_limits, {"usd": 6.0})

    def test_pool_state_on_arrival_is_recorded_when_supplied(self):
        """The column that makes a budget_exhausted arm legible: it says
        what was ALREADY gone when this arm started."""
        arrival = {"declared": True, "limits": {"usd": 6.0},
                   "usage_fraction": 0.635}
        result = self._run({"writes": "READY\n"}, pool_on_arrival=arrival)
        self.assertEqual(result.pool_on_arrival, arrival)

    def test_a_fixture_failure_still_reports_what_the_arm_was_allowed(self):
        """Ceilings are properties of what was ALLOWED, not of what ran,
        so an arm blocked before its session opened keeps them."""
        (self.root / "ws").mkdir(parents=True)
        (self.root / "ws" / "t_echo@cheap_0").mkdir()
        result = self._run({"writes": "READY\n"})
        self.assertEqual(result.state, BLOCKED)
        self.assertIn("fixture", result.blocked_reason)
        self.assertEqual(result.pool_limits, {"usd": 6.0})


class RecordCase(RunnerHarness):
    """The provenance block survives the trip into the results file."""

    def test_every_new_field_is_written_even_when_null(self):
        """A reader must be able to tell a field this engine could not
        establish from a field a newer engine added; an omitted key looks
        like the latter."""
        record = self._run({"writes": "READY\n"}).to_dict()
        for key in ("session_id", "model", "provider", "upstream_provider",
                    "native_finish_reason", "completion_nudges",
                    "budget_ceiling", "pool_limits", "pool_on_arrival"):
            self.assertIn(key, record, key)


if __name__ == "__main__":
    unittest.main()
