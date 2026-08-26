"""Workspaces are hermetic, and refuse to be reused."""
import tempfile
import unittest
from pathlib import Path

from jaato_eval.fixture import FixtureError, discard, materialise


class FixtureCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.src = self.root / "fixture"
        (self.src / "sub").mkdir(parents=True)
        (self.src / "sub" / "a.txt").write_text("original")
        (self.src / ".git").mkdir()
        (self.src / ".git" / "HEAD").write_text("ref: refs/heads/main")
        self.addCleanup(self.tmp.cleanup)

    def test_copies_tree_and_writes_profile_set(self):
        ws = materialise(self.src, self.root / "ws1", profile_set="cheap")
        self.assertEqual((ws.path / "sub" / "a.txt").read_text(), "original")
        self.assertIn("JAATO_PROFILE_SET=cheap", ws.env_file.read_text())

    def test_git_is_not_copied(self):
        ws = materialise(self.src, self.root / "ws2")
        self.assertFalse((ws.path / ".git").exists())

    def test_arms_do_not_contaminate_each_other(self):
        a = materialise(self.src, self.root / "wsa")
        (a.path / "sub" / "a.txt").write_text("mutated by arm a")
        b = materialise(self.src, self.root / "wsb")
        self.assertEqual((b.path / "sub" / "a.txt").read_text(), "original")

    def test_refuses_existing_destination(self):
        materialise(self.src, self.root / "ws3")
        with self.assertRaises(FixtureError) as ctx:
            materialise(self.src, self.root / "ws3")
        self.assertIn("refusing to reuse", str(ctx.exception))

    def test_missing_fixture_raises(self):
        with self.assertRaises(FixtureError):
            materialise(self.root / "absent", self.root / "ws4")

    def test_extra_env_written(self):
        ws = materialise(self.src, self.root / "ws5", profile_set="s",
                         env={"VLLM_HOST": "http://x:8000"})
        text = ws.env_file.read_text()
        self.assertIn("VLLM_HOST=http://x:8000", text)

    def test_discard_removes(self):
        ws = materialise(self.src, self.root / "ws6")
        discard(ws)
        self.assertFalse(ws.path.exists())


if __name__ == "__main__":
    unittest.main()
