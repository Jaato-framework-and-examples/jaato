"""Resolving the one profile field no event reports — jaato #777.

The daemon announces the model and provider it bound
(``SessionInfoEvent``), so the engine never has to read those off a file.
It announces nothing about ``budget_control``, so the per-arm report's
budget column has to come from the profile the arm ran under.

Two framework rules are restated here and both are load-bearing:
set-directory-first (which is what makes the sweep's model axis an axis),
and min-wins limit merging (a child may only ever TIGHTEN a ceiling).
Getting the second backwards would print a ceiling larger than the one
the daemon actually enforced.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jaato_eval.profile import resolve_budget_ceiling


class CeilingCase(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "profiles").mkdir(parents=True)
        self.addCleanup(self.tmp.cleanup)

    def _write(self, relative, text):
        path = self.root / "profiles" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def test_a_profile_with_no_budget_control_resolves_to_none(self):
        """Unbudgeted is a real state: such a session draws on the task
        pool instead."""
        self._write("worker.yaml", "name: worker\n")
        self.assertIsNone(resolve_budget_ceiling(self.root, "worker", None))

    def test_a_missing_profile_is_none_rather_than_empty(self):
        """'this engine could not find it' must not read as
        'unbudgeted' — the two lead a reader to opposite conclusions."""
        self.assertIsNone(resolve_budget_ceiling(self.root, "worker", None))

    def test_the_set_directory_wins_over_the_base_tier(self):
        """The sweep's model axis: ``profiles/<set>/`` is scanned first
        and first-scanned wins, exactly as ``discover_profiles`` does."""
        self._write("worker.yaml",
                    "name: worker\nbudget_control:\n  limits:\n    usd: 9.0\n")
        self._write("cheap/worker.yaml",
                    "name: worker\nbudget_control:\n  limits:\n    usd: 1.0\n")
        self.assertEqual(resolve_budget_ceiling(self.root, "worker", "cheap"),
                         {"usd": 1.0})

    def test_without_a_set_the_base_tier_answers(self):
        self._write("worker.yaml",
                    "name: worker\nbudget_control:\n  limits:\n    usd: 9.0\n")
        self.assertEqual(resolve_budget_ceiling(self.root, "worker", None),
                         {"usd": 9.0})

    def test_limits_merge_min_wins_across_inheritance(self):
        """Not child-replaces-parent.  On a resource ceiling that would be
        an escape hatch, and the report would name a ceiling the daemon
        never enforced."""
        self._write("_base.yaml",
                    "name: _base\nbudget_control:\n  limits:\n"
                    "    usd: 1.0\n    turns: 40\n")
        self._write("cheap/worker.yaml",
                    "name: worker\ninherits: [_base]\n"
                    "budget_control:\n  limits:\n    usd: 5.0\n    tokens: 10\n")
        self.assertEqual(
            resolve_budget_ceiling(self.root, "worker", "cheap"),
            {"usd": 1.0, "turns": 40.0, "tokens": 10.0})

    def test_a_dimension_only_the_child_declares_is_kept(self):
        """An absent dimension is unbounded, so whichever layer declares
        one is strictly tighter."""
        self._write("_base.yaml", "name: _base\n")
        self._write("worker.yaml",
                    "name: worker\ninherits: _base\n"
                    "budget_control:\n  limits:\n    usd: 2.0\n")
        self.assertEqual(resolve_budget_ceiling(self.root, "worker", None),
                         {"usd": 2.0})

    def test_json_profiles_are_read_too(self):
        """``_scan_profiles_dir`` accepts .json alongside .yaml/.yml."""
        self._write("worker.json", json.dumps(
            {"name": "worker", "budget_control": {"limits": {"usd": 3.0}}}))
        self.assertEqual(resolve_budget_ceiling(self.root, "worker", None),
                         {"usd": 3.0})

    def test_a_cycle_terminates_instead_of_hanging(self):
        """Reporting a cycle is the daemon's job — an arm whose profiles
        do not resolve never runs.  This module's only duty is not to
        spin."""
        self._write("a.yaml", "name: a\ninherits: [b]\n"
                              "budget_control:\n  limits:\n    usd: 1.0\n")
        self._write("b.yaml", "name: b\ninherits: [a]\n")
        self.assertEqual(resolve_budget_ceiling(self.root, "a", None),
                         {"usd": 1.0})

    def test_a_non_numeric_ceiling_is_dropped_not_coerced(self):
        """The daemon would have rejected the profile; a number invented
        here would put a figure in the report that bounded nothing."""
        self._write("worker.yaml", "name: worker\nbudget_control:\n"
                                   "  limits:\n    usd: lots\n    turns: 5\n")
        self.assertEqual(resolve_budget_ceiling(self.root, "worker", None),
                         {"turns": 5.0})

    def test_unparseable_yaml_is_none_not_a_crash(self):
        self._write("worker.yaml", "name: [worker\n")
        self.assertIsNone(resolve_budget_ceiling(self.root, "worker", None))


if __name__ == "__main__":
    unittest.main()
