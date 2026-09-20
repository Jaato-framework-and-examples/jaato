"""The signal a consumer is TOLD to watch is the one that is emitted.

THE DEFECT (#771).  ``TurnCompletedEvent.completion_gap``'s docstring made
two claims about the nudge-exhaustion path and both were false:

  1. *"This is the only signal a consumer gets in that state"* -- the field
     does not reach a consumer on that path at all.
  2. *"no SessionTerminatedEvent fires (quiescence is gated on
     signal_completion having been called)"* -- one does, carrying
     ``error_type="NudgeExhausted"``.

The matching daemon trace said ``no terminal event will fire for this
session`` roughly twenty lines above the code that fires one.

WHY THE EXISTING GUARD DID NOT CATCH IT.
``test_a_missing_completion_is_not_silent.py`` asserts the field round-trips
through ``to_dict()`` and that a constructed event carries it.  Both pass,
and neither asks whether a daemon ever EMITS one carrying the value -- which
is the only claim a consumer depends on.  That is the recurring shape the
issue names: *a guard that tests the carrier rather than the delivery.*

WHAT THIS MODULE ASSERTS.  Three facts, each the evidence for one sentence
of the corrected docstring:

  * the nudge-exhaustion branch really does emit a ``NudgeExhausted``
    terminal -- the signal consumers are now pointed at;
  * ``completion_gap`` has exactly ONE writer in the server, which is why it
    is reliably ``None`` (the write happens after the turn event was built
    and cleared, and the session then ends); and
  * the docstring itself no longer makes either false claim, and names the
    event that does fire.

The third is deliberately scoped to the ``#:`` block attached to the field
rather than to the file: a test grepping a whole module for a word that
appears elsewhere in it passes for the wrong reason.
"""

import ast
import pathlib
import re
from typing import List

from shared.tests.reversion import Reversion

#: Two reversions, one per claim the corrected docstring makes.
REVERSIONS = [
    Reversion(
        target="jaato-server/server/core.py",
        find="""                    server._emit_error_termination(
                        error_type="NudgeExhausted",
                        error_summary=nudge_exhaust_summary,
                    )""",
        replace="""                    pass  # terminal removed""",
        test="TestTheTerminalIsReal::test_the_nudge_exhaust_branch_emits_a_terminal",
        because="the terminal the docstring now tells consumers to watch",
    ),
    Reversion(
        target="jaato-sdk/jaato_sdk/events.py",
        find=(
            "    #: **No daemon in this tree delivers that value, and none "
            "is expected\n    #: to.  Watch the terminal instead** (#771):"
        ),
        replace=(
            "    #: This is the only signal a consumer gets in that state. "
            "No\n    #: ``SessionTerminatedEvent`` fires."
        ),
        test="TestTheDocstringSaysWhatIsTrue::test_the_field_docstring_does_not_claim_it_is_the_only_signal",
        because="the docstring claiming delivery it does not get",
    ),
]

ROOT = pathlib.Path(__file__).resolve().parents[3]
CORE = ROOT / "jaato-server" / "server" / "core.py"
EVENTS = ROOT / "jaato-sdk" / "jaato_sdk" / "events.py"

GAP_VALUE = "not_signalled_after_nudges"
TERMINAL = "NudgeExhausted"


def _terminal_emissions() -> List[str]:
    """Every call in core.py passing ``error_type="NudgeExhausted"``.

    Read from the AST rather than by substring, so a mention inside a
    comment or a docstring cannot stand in for a live call.
    """
    tree = ast.parse(CORE.read_text(encoding="utf-8"))
    found: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg != "error_type":
                continue
            if isinstance(kw.value, ast.Constant) and kw.value.value == TERMINAL:
                func = node.func
                name = getattr(func, "attr", None) or getattr(func, "id", "?")
                found.append(str(name))
    return found


def _gap_writers() -> List[int]:
    """Line numbers of every assignment of the gap value in the server."""
    server_dir = ROOT / "jaato-server" / "server"
    lines: List[int] = []
    for path in sorted(server_dir.rglob("*.py")):
        if "/tests/" in path.as_posix():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not (isinstance(node.value, ast.Constant)
                    and node.value.value == GAP_VALUE):
                continue
            lines.append(node.lineno)
    return lines


def field_doc_comment() -> str:
    """The ``#:`` block attached to ``completion_gap`` -- and only it.

    Scoping matters: this guard asserts what the FIELD's own documentation
    says, and ``events.py`` is a large module where every word it checks
    for appears somewhere else.  A file-wide grep would pass for the wrong
    reason, which is the failure mode the issue is about.
    """
    src = EVENTS.read_text(encoding="utf-8").splitlines()
    anchor = None
    for i, line in enumerate(src):
        if re.match(r"\s*completion_gap:\s*Optional\[str\]", line):
            anchor = i
            break
    assert anchor is not None, (
        "could not find the completion_gap field declaration in events.py; "
        "this guard is measuring nothing"
    )
    block: List[str] = []
    i = anchor - 1
    while i >= 0 and src[i].lstrip().startswith("#:"):
        block.append(src[i].lstrip()[2:].strip())
        i -= 1
    return "\n".join(reversed(block))


class TestTheTerminalIsReal:
    def test_the_nudge_exhaust_branch_emits_a_terminal(self):
        """#771 claim 2: a terminal DOES fire on this path."""
        emissions = _terminal_emissions()
        assert emissions, (
            f"no call in server/core.py passes error_type={TERMINAL!r}. The "
            f"completion_gap docstring tells consumers to watch for that "
            f"terminal; if nothing emits it, they are being pointed at "
            f"nothing."
        )
        assert "_emit_error_termination" in emissions, (
            f"nothing calls _emit_error_termination(error_type={TERMINAL!r}); "
            f"an ErrorEvent alone is not a SessionTerminatedEvent, and the "
            f"docstring names both. Found: {emissions}"
        )


class TestTheGapDoesNotArrive:
    def test_the_gap_has_exactly_one_writer(self):
        """The evidence for the docstring's explanation.

        The corrected docstring says the value is written in exactly one
        place, after the turn event has been built and cleared.  If a
        second writer appears, that explanation is no longer true and the
        docstring needs revisiting -- possibly because someone made the
        field work, which would be a welcome reason to fail.
        """
        writers = _gap_writers()
        assert len(writers) == 1, (
            f"expected exactly one writer of {GAP_VALUE!r} in jaato-server/"
            f"server/, found {len(writers)} at lines {writers}. The "
            f"completion_gap docstring explains its own None-ness by there "
            f"being one writer that runs too late; re-check that claim."
        )


class TestTheDocstringSaysWhatIsTrue:
    def test_the_field_docstring_does_not_claim_it_is_the_only_signal(self):
        """#771 claim 1, and claim 2, in the text that makes them."""
        doc = field_doc_comment().lower()
        assert doc, "completion_gap carries no #: documentation at all"
        assert "only signal a consumer gets" not in doc, (
            "the completion_gap docstring still claims to be the only signal "
            "a consumer gets. It does not reach a consumer on the "
            "nudge-exhaustion path at all (#771)."
        )
        assert "no ``sessionterminatedevent`` fires" not in doc, (
            "the completion_gap docstring still claims no SessionTerminated"
            "Event fires. One does, carrying error_type='NudgeExhausted'."
        )

    def test_the_field_docstring_names_the_event_that_does_fire(self):
        """A correction that removes a lie and offers nothing is half done."""
        doc = field_doc_comment()
        assert TERMINAL in doc, (
            f"the completion_gap docstring does not name {TERMINAL!r}. A "
            f"consumer told the field is unreliable still needs to be told "
            f"what to watch instead."
        )
