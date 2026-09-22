"""The results file is a CONTRACT, and the contract's producer half.

``docs/eval-results.md`` declares what a reader outside this package may rely
on.  The first such reader is jaato's dossier generator
(``jaato-server/jaato_server/shared/scaffold/eval_results.py``), which renders an arm's
numbers into the accuracy section of an Annex IV dossier (jaato #1124) and
may NOT import this engine -- ``jaato_eval`` imports ``jaato_sdk`` and
nothing else from that tree, so a consumer importing the producer would run
that rule backwards.

Two fields therefore exist for that reader and for nobody here, and nothing
inside this package would notice if either stopped being written.  These
tests are what notices.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from types import SimpleNamespace

from jaato_eval.arm import ArmResult, ArmSpec
from jaato_eval.results import ResultStore
from jaato_eval.results_format import (
    GRADER_CAVEATS, RESULTS_FORMAT_VERSION, caveats_for)
from jaato_eval.verdict import PASS, Verdict


def _result(*grader_ids: str) -> ArmResult:
    # A stand-in manifest: ``ArmSpec`` and ``to_dict`` read ``task_id`` and
    # nothing else, and building a whole ``TaskManifest`` here would assert a
    # dependency this contract does not have.
    spec = ArmSpec(task=SimpleNamespace(task_id="t/one"), profile_set="s",
                   repeat=0)
    return ArmResult(
        spec=spec,
        verdicts=[Verdict(grader_id=g, claim="c", state=PASS)
                  for g in grader_ids],
    )


class CaveatsFor(unittest.TestCase):

    def test_the_judge_declares_its_limit(self):
        self.assertIn("judge", GRADER_CAVEATS)
        self.assertTrue(GRADER_CAVEATS["judge"].strip())

    def test_a_kind_with_no_declared_limit_asserts_nothing(self):
        """Absence is 'we have not measured a limit worth stating'.

        Never 'this instrument is calibrated' -- so an unknown kind
        contributes no caveat rather than a reassuring one.
        """
        self.assertEqual(caveats_for(["script:x", "processor:y"]), [])

    def test_one_limit_is_stated_once(self):
        """Two judge rubrics on one arm state the judge's limit once.

        De-duplicated so a reader's diff stays stable and the section does
        not repeat a paragraph per grader.
        """
        self.assertEqual(
            caveats_for(["judge:a", "script:x", "judge:b"]),
            [GRADER_CAVEATS["judge"]])

    def test_a_bare_kind_still_resolves(self):
        self.assertEqual(caveats_for(["judge"]), [GRADER_CAVEATS["judge"]])


class TheRecord(unittest.TestCase):

    def test_every_record_declares_its_version(self):
        """Without it a reader cannot refuse a file it does not understand,
        and a half-understood accuracy table looks like a complete one."""
        record = _result("script:x").to_dict()
        self.assertEqual(record["results_version"], RESULTS_FORMAT_VERSION)

    def test_a_judge_graded_arm_carries_the_judge_caveat(self):
        record = _result("judge:rubric").to_dict()
        self.assertEqual(record["caveats"], [GRADER_CAVEATS["judge"]])

    def test_an_arm_with_no_caveated_grader_carries_an_empty_list(self):
        """An empty list, never a missing key: a consumer distinguishes
        'measured nothing to say' from 'an older engine wrote this'."""
        record = _result("script:x").to_dict()
        self.assertEqual(record["caveats"], [])

    def test_both_fields_survive_the_round_trip_through_the_store(self):
        with TemporaryDirectory() as tmp:
            store = ResultStore(Path(tmp) / "r.jsonl")
            store.append(_result("judge:rubric"))
            [record] = store.read()
        self.assertEqual(record["results_version"], RESULTS_FORMAT_VERSION)
        self.assertEqual(record["caveats"], [GRADER_CAVEATS["judge"]])

    def test_the_record_is_json_serialisable_as_written(self):
        json.dumps(_result("judge:rubric").to_dict(), sort_keys=True,
                   default=str)


class TheContractIsDocumented(unittest.TestCase):

    def test_the_declaration_names_this_version(self):
        """``docs/eval-results.md`` is the agreement both halves are written
        against; a version it does not mention is one no reader was told to
        expect."""
        doc = (Path(__file__).resolve().parents[2]
               / "docs" / "eval-results.md")
        if not doc.is_file():          # installed sdist: the doc is not shipped
            self.skipTest("docs/eval-results.md is not in this tree")
        text = doc.read_text(encoding="utf-8")
        self.assertIn("results_version", text)
        self.assertIn(f"**`\"{RESULTS_FORMAT_VERSION}\"`**", text)
        self.assertIn("verbatim", text)


if __name__ == "__main__":       # pragma: no cover
    unittest.main()
