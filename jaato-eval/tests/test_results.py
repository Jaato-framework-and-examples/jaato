"""Result store survives partial writes and supports resume."""
import tempfile
import unittest
from pathlib import Path

from jaato_eval.arm import ArmResult, ArmSpec
from jaato_eval.results import ResultStore, canonical_hash


class _FakeTask:
    task_id = "t"


class ResultsCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "nested" / "results.jsonl"
        self.addCleanup(self.tmp.cleanup)

    def _result(self, repeat=0, pset="cheap"):
        spec = ArmSpec(task=_FakeTask(), profile_set=pset, repeat=repeat)
        return ArmResult(spec=spec, blocked_reason="not run")

    def test_append_creates_parents_and_round_trips(self):
        store = ResultStore(self.path)
        store.append(self._result())
        records = store.read()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["arm_id"], "t@cheap#0")
        self.assertEqual(records[0]["state"], "BLOCKED")

    def test_missing_file_reads_empty(self):
        self.assertEqual(ResultStore(self.path).read(), [])

    def test_truncated_trailing_line_skipped(self):
        """A sweep killed mid-write must not make the file unreadable."""
        store = ResultStore(self.path)
        store.append(self._result())
        with self.path.open("a") as fh:
            fh.write('{"arm_id": "partial"')
        self.assertEqual(len(store.read()), 1)

    def test_completed_arm_ids_for_resume(self):
        store = ResultStore(self.path)
        store.append(self._result(repeat=0))
        store.append(self._result(repeat=1))
        self.assertEqual(store.completed_arm_ids(), {"t@cheap#0", "t@cheap#1"})

    def test_canonical_hash_is_key_order_independent(self):
        self.assertEqual(canonical_hash({"a": 1, "b": 2}),
                         canonical_hash({"b": 2, "a": 1}))

    def test_canonical_hash_distinguishes_content(self):
        self.assertNotEqual(canonical_hash({"a": 1}), canonical_hash({"a": 2}))


if __name__ == "__main__":
    unittest.main()
