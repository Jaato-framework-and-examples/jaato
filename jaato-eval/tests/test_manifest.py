"""Manifest parsing fails loud rather than inferring."""
import tempfile
import unittest
from pathlib import Path

from jaato_eval.manifest import ManifestError, discover_tasks, load_manifest

GOOD = """
id: demo/task
description: A demo.
environment:
  fixture: fixture
  config_root: cfg
input:
  prompt: Do the thing.
  agent_params: {size: small}
harness:
  profile: worker
  profile_set: cheap
graders:
  - kind: script
    run: "true"
repeats: 2
"""


class ManifestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "fixture").mkdir()
        (self.root / "cfg").mkdir()
        self.addCleanup(self.tmp.cleanup)

    def write(self, text, name="task.yaml"):
        p = self.root / name
        p.write_text(text)
        return p

    def test_parses_a_good_manifest(self):
        m = load_manifest(self.write(GOOD))
        self.assertEqual(m.task_id, "demo/task")
        self.assertEqual(m.repeats, 2)
        self.assertEqual(m.harness.profile_set, "cheap")
        self.assertEqual(m.input.agent_params, {"size": "small"})
        self.assertEqual(m.graders[0].kind, "script")
        self.assertEqual(m.graders[0].identifier, "true")
        self.assertTrue(m.resolved_fixture().is_dir())

    def test_missing_file(self):
        with self.assertRaises(ManifestError):
            load_manifest(self.root / "nope.yaml")

    def test_missing_required_key(self):
        with self.assertRaises(ManifestError) as ctx:
            load_manifest(self.write(GOOD.replace("  profile: worker\n", "")))
        self.assertIn("profile", str(ctx.exception))

    def test_missing_fixture_directory_caught_before_any_run(self):
        """Existence is checked at parse time so a malformed dataset fails
        before provider tokens are spent."""
        with self.assertRaises(ManifestError) as ctx:
            load_manifest(self.write(GOOD.replace("fixture: fixture", "fixture: absent")))
        self.assertIn("does not exist", str(ctx.exception))

    def test_unknown_grader_kind(self):
        with self.assertRaises(ManifestError) as ctx:
            load_manifest(self.write(GOOD.replace("kind: script", "kind: vibes")))
        self.assertIn("unknown kind", str(ctx.exception))

    def test_empty_prompt_rejected(self):
        with self.assertRaises(ManifestError):
            load_manifest(self.write(GOOD.replace("Do the thing.", "   ")))

    def test_empty_graders_rejected(self):
        bad = GOOD.split("graders:")[0] + "graders: []\n"
        with self.assertRaises(ManifestError):
            load_manifest(self.write(bad))

    def test_duplicate_task_ids_rejected(self):
        """Two tasks sharing an id would overwrite each other in the pivot."""
        self.write(GOOD)
        sub = self.root / "other"
        (sub / "fixture").mkdir(parents=True)
        (sub / "cfg").mkdir()
        (sub / "task.yaml").write_text(GOOD)
        with self.assertRaises(ManifestError) as ctx:
            discover_tasks(self.root)
        self.assertIn("duplicate task id", str(ctx.exception))


DRIVER = """
id: demo/backtest
description: One backtest cell.
environment:
  fixture: fixture
  config_root: cfg
input:
  params: {TICKER: NVDA, TRADE_DATE: "2026-08-28"}
harness:
  kind: driver
  run: python -m ta_cascade analyze
  profile_set: cheap
graders:
  - kind: script
    run: "true"
repeats: 5
"""


class HarnessKindCase(ManifestCase):
    """``harness.kind`` selects the variant, and each variant refuses the
    other's keys by name (jaato #1110)."""

    def test_absent_kind_is_a_session(self):
        m = load_manifest(self.write(GOOD))
        self.assertEqual(m.harness.kind, "session")
        self.assertFalse(m.harness.is_driver)
        self.assertIsNone(m.harness.run)
        self.assertEqual(m.input.params, {})
        self.assertEqual(m.input.grader_params, {"size": "small"})

    def test_a_driver_parses(self):
        m = load_manifest(self.write(DRIVER))
        self.assertTrue(m.harness.is_driver)
        self.assertEqual(m.harness.run, "python -m ta_cascade analyze")
        self.assertIsNone(m.harness.profile)
        self.assertEqual(m.harness.profile_set, "cheap")
        self.assertIsNone(m.input.prompt)
        self.assertEqual(m.input.params, {"TICKER": "NVDA", "TRADE_DATE": "2026-08-28"})
        self.assertEqual(m.input.grader_params, m.input.params)
        self.assertEqual(m.repeats, 5)

    def test_unknown_kind_names_the_vocabulary(self):
        with self.assertRaises(ManifestError) as ctx:
            load_manifest(self.write(DRIVER.replace("kind: driver", "kind: python")))
        self.assertIn("harness.kind", str(ctx.exception))
        self.assertIn("driver", str(ctx.exception))

    def test_a_driver_without_run_is_refused(self):
        with self.assertRaises(ManifestError) as ctx:
            load_manifest(self.write(DRIVER.replace("  run: python -m ta_cascade analyze\n", "")))
        self.assertIn("run", str(ctx.exception))

    def test_an_empty_run_is_refused(self):
        with self.assertRaises(ManifestError):
            load_manifest(self.write(DRIVER.replace("run: python -m ta_cascade analyze", "run: '  '")))

    def test_a_driver_with_a_prompt_is_refused_naming_the_variant(self):
        """A prompt on a driver arm reaches nothing; its author believes
        it was sent."""
        bad = DRIVER.replace("input:\n", "input:\n  prompt: do the thing\n")
        with self.assertRaises(ManifestError) as ctx:
            load_manifest(self.write(bad))
        self.assertIn("input.prompt", str(ctx.exception))
        self.assertIn("harness.kind: driver", str(ctx.exception))

    def test_a_driver_with_a_profile_is_refused(self):
        bad = DRIVER.replace("  kind: driver\n", "  kind: driver\n  profile: worker\n")
        with self.assertRaises(ManifestError) as ctx:
            load_manifest(self.write(bad))
        self.assertIn("harness.profile", str(ctx.exception))

    def test_a_session_with_run_is_refused(self):
        bad = GOOD.replace("  profile: worker\n", "  profile: worker\n  run: python x.py\n")
        with self.assertRaises(ManifestError) as ctx:
            load_manifest(self.write(bad))
        self.assertIn("harness.run", str(ctx.exception))
        self.assertIn("harness.kind: session", str(ctx.exception))

    def test_a_session_with_params_is_refused(self):
        bad = GOOD.replace("  agent_params: {size: small}\n",
                           "  agent_params: {size: small}\n  params: {x: 1}\n")
        with self.assertRaises(ManifestError) as ctx:
            load_manifest(self.write(bad))
        self.assertIn("input.params", str(ctx.exception))

    def test_a_driver_may_have_no_input_block(self):
        """A driver whose whole input is its command line has nothing to
        put there."""
        no_input = DRIVER.replace(
            'input:\n  params: {TICKER: NVDA, TRADE_DATE: "2026-08-28"}\n', "")
        m = load_manifest(self.write(no_input))
        self.assertTrue(m.harness.is_driver)
        self.assertEqual(m.input.params, {})

    def test_a_param_collision_is_refused_before_any_arm_runs(self):
        """``issue-id`` and ``issue_id`` both want one variable; the
        driver and its graders would read whichever won."""
        bad = DRIVER.replace('params: {TICKER: NVDA, TRADE_DATE: "2026-08-28"}',
                             'params: {issue-id: 1, issue_id: 2}')
        with self.assertRaises(ManifestError) as ctx:
            load_manifest(self.write(bad))
        self.assertIn("JAATO_EVAL_PARAM_ISSUE_ID", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
