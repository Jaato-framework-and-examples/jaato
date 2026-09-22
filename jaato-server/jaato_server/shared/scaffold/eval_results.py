"""Render a ``jaato-eval`` results file into the dossier's accuracy section.

Article 15(3) asks that accuracy levels and the relevant accuracy metrics be
declared in the instructions for use, and 9(8) asks for testing against
prior-defined metrics.  ``jaato-eval`` already measures; the dossier already
has the section; this is the joint (#1124).

IT READS A FORMAT, IT DOES NOT IMPORT AN ENGINE
===============================================

``jaato_eval`` imports ``jaato_sdk`` and nothing else from this tree -- the
rule :mod:`jaato_eval.sign_off` records -- so a consumer inside ``shared`` that
imported it would run that rule backwards, and the one import this module makes
would decide whether a dossier can be generated at all on a machine where the
harness is not installed.  So the two sides are joined by a declared contract:
``docs/eval-results.md``, whose producer half is
``jaato-eval/jaato_eval/results_format.py``.  Nothing here imports the engine,
and the file is parsed as JSON lines.

THREE REFUSALS, EACH BECAUSE THE ALTERNATIVE LOOKS COMPLETE
===========================================================

An accuracy section is the part of a dossier a reader quotes.  Every failure
here therefore REFUSES by name rather than rendering what it could make sense
of, because a half-rendered table is indistinguishable from a full one:

* a file that is not there, or holds no record;
* a record whose ``results_version`` this reader does not know -- **an absent
  version is an unknown version**, not a version 1 record, since the field has
  been written since the contract was declared;
* a file whose records disagree about their version.

THE CAVEAT IS NOT OURS TO WRITE
===============================

``caveats`` carries what the graders cannot tell you, in the words of the
harness that measured the limit, and every one is rendered **verbatim** above
the numbers it qualifies.  This module never summarises one and never supplies
one of its own: a hand-written warning about LLM judges here would be a second
copy of a fact, and the copy that rots is the one nothing executes -- it would
go on being quoted after the limit was fixed, or miss a limit added later.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

#: The results-format versions this reader understands.  Declared here rather
#: than imported, deliberately: this is a statement about what THIS reader can
#: make sense of, and reading it from the producer would make every file
#: readable by construction -- which is the check disappearing, not passing.
SUPPORTED_VERSIONS: Tuple[str, ...] = ("1",)

#: The state vocabulary of version 1.  ``BLOCKED`` is excluded from a pass-rate
#: denominator: nothing was exercised, so it is neither a pass nor a failure of
#: the thing under test.
PASS, FAIL, BLOCKED = "PASS", "FAIL", "BLOCKED"


class EvalResultsError(Exception):
    """A results file this reader will not render, and why."""


def _records(path: Path) -> List[Dict[str, Any]]:
    """Every JSON object in the file, in order.

    A truncated trailing line is skipped -- a sweep killed mid-write leaves
    one, and the engine's own reader tolerates it, so refusing the whole file
    over it would refuse a file the harness considers readable.  Any other
    malformed line is a refusal: it is evidence this is not the file we think.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvalResultsError(f"cannot read {path}: {exc}") from exc

    lines = [ln for ln in raw.splitlines() if ln.strip()]
    out: List[Dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1:
                continue  # the known, benign truncated tail
            raise EvalResultsError(
                f"{path} line {index + 1} is not JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise EvalResultsError(
                f"{path} line {index + 1} is not a results record")
        out.append(record)
    return out


def _check_version(path: Path, records: Iterable[Dict[str, Any]]) -> str:
    """The file's one declared version, or a refusal naming what was found."""
    seen: List[str] = []
    for record in records:
        declared = record.get("results_version")
        label = "none" if declared is None else str(declared)
        if label not in seen:
            seen.append(label)

    if len(seen) > 1:
        raise EvalResultsError(
            f"{path} mixes results_version {', '.join(repr(s) for s in seen)}; "
            f"a file whose records disagree about their format is not one "
            f"this reader can render")

    version = seen[0]
    if version not in SUPPORTED_VERSIONS:
        known = ", ".join(repr(v) for v in SUPPORTED_VERSIONS)
        found = ("declares no results_version, so it predates the format "
                 "contract" if version == "none" else
                 f"declares results_version {version!r}")
        raise EvalResultsError(
            f"{path} {found}; this reader knows {known}. Refusing rather than "
            f"rendering the fields it recognises: a dossier section filled "
            f"from a guessed format reads as complete. See docs/eval-results.md")
    return version


# ------------------------------------------------------------- aggregation

class _Group:
    """One (task, profile set) cell: the unit the harness reports on.

    Never an average across cells.  Repeats disagreeing IS the measurement,
    and a mean over two repeats of a coin flip reads as a model difference.
    """

    def __init__(self, task_id: str, profile_set: str) -> None:
        self.task_id = task_id
        self.profile_set = profile_set
        self.passed = 0
        self.failed = 0
        self.blocked = 0
        self.graders: List[str] = []

    @property
    def exercised(self) -> int:
        return self.passed + self.failed

    @property
    def arms(self) -> int:
        return self.exercised + self.blocked

    @property
    def pass_rate(self) -> Optional[float]:
        """``None`` when nothing was exercised -- not ``0.0``.

        Zero would say "it always failed"; the truth is "we never found out".
        """
        return (self.passed / self.exercised) if self.exercised else None

    def add(self, record: Dict[str, Any]) -> None:
        state = str(record.get("state", "")).upper()
        if state == PASS:
            self.passed += 1
        elif state == FAIL:
            self.failed += 1
        else:
            self.blocked += 1
        for verdict in record.get("verdicts") or []:
            if not isinstance(verdict, dict):
                continue
            kind = str(verdict.get("grader_id", "")).split(":", 1)[0]
            if kind and kind not in self.graders:
                self.graders.append(kind)


def _group(records: Iterable[Dict[str, Any]]) -> List[_Group]:
    cells: "OrderedDict[Tuple[str, str], _Group]" = OrderedDict()
    for record in records:
        key = (str(record.get("task_id") or "(unnamed task)"),
               str(record.get("profile_set") or "default"))
        cell = cells.get(key)
        if cell is None:
            cell = cells[key] = _Group(*key)
        cell.add(record)
    return list(cells.values())


def _caveats(records: Iterable[Dict[str, Any]]) -> List[str]:
    """Every caveat the FILE carries, de-duplicated, order preserved."""
    out: List[str] = []
    for record in records:
        for caveat in record.get("caveats") or []:
            text = str(caveat).strip()
            if text and text not in out:
                out.append(text)
    return out


def _provenance(records: List[Dict[str, Any]]) -> List[str]:
    """Which code produced these numbers, from the file's own block."""
    seen: List[str] = []
    for record in records:
        block = record.get("provenance")
        if not isinstance(block, dict):
            continue
        line = (f"`jaato-sdk {block.get('jaato_sdk_version') or 'unknown'}` "
                f"from `{block.get('jaato_sdk_path') or 'unknown'}`")
        if line not in seen:
            seen.append(line)
    return seen


# ---------------------------------------------------------------- render

def render_section(path_str: str) -> List[str]:
    """The accuracy section's body, as markdown lines.

    Raises:
        EvalResultsError: the file is absent, empty, malformed, or declares a
            format this reader does not know.  The caller renders the refusal
            rather than an accuracy section, because there is no honest
            partial answer here.
    """
    path = Path(path_str)
    records = _records(path)
    if not records:
        raise EvalResultsError(
            f"{path} holds no readable results record; an accuracy section "
            f"with no arms behind it would be an assertion, not a "
            f"measurement. (A single unparsable line reads as the truncated "
            f"tail a killed sweep leaves, so it is skipped rather than "
            f"reported as malformed.)")
    version = _check_version(path, records)

    lines: List[str] = [
        f"Measured by `jaato-eval` from `{path}` "
        f"(results format `{version}`, {len(records)} arm"
        f"{'' if len(records) == 1 else 's'}).",
        "",
    ]

    for caveat in _caveats(records):
        # Verbatim, on the harness's authority.  Above the table rather than
        # below it, because a reader who stops at the numbers must have met
        # the limit of the instrument first.
        lines.extend([f"> {caveat}", ""])

    lines.extend([
        "| Task | Profile set | Graders | Arms | Exercised | Pass rate "
        "| Pass | Fail | Blocked | Threshold |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ])
    for cell in _group(records):
        rate = "—" if cell.pass_rate is None else f"{cell.pass_rate * 100:.0f}%"
        graders = ", ".join(f"`{g}`" for g in cell.graders) or "—"
        lines.append(
            f"| {cell.task_id} | {cell.profile_set} | {graders} | {cell.arms} "
            f"| {cell.exercised} | {rate} | {cell.passed} | {cell.failed} "
            f"| {cell.blocked} | **TODO** |")

    lines.extend([
        "",
        "_Pass rate excludes blocked arms from its denominator: a blocked arm "
        "was never exercised, so it is neither a pass nor a failure of the "
        "system under test. A rate of `—` means every arm in that cell "
        "blocked, which is not `0%`._",
        "",
        _threshold_todo(),
    ])

    provenance = _provenance(records)
    if provenance:
        lines.extend(["", "Produced by: " + "; ".join(provenance) + "."])
    return lines


def _threshold_todo() -> str:
    """The one thing the framework must not fill in for a provider."""
    return (
        "> **TODO — the framework cannot supply this.** Every row's "
        "**Threshold** is the accuracy level you declare as appropriate to "
        "the intended purpose (Article 15(3)); choosing it, and choosing "
        "which metrics are the relevant ones, is the provider's call and not "
        "a measurement this harness can make. State the threshold beside each "
        "metric and say what happens when a run falls below it.\n>\n"
        "> Left in rather than omitted: an absent section in a legal document "
        "reads as *nothing to declare*.")
