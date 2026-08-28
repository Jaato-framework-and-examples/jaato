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


if __name__ == "__main__":
    unittest.main()
