"""A scaffolded driver must handle the session ending under its turn verb.

WHY THIS EXISTS.  #1007/#1044 changed what the facade's verbs do when a
terminal arrives mid-turn: ``ask`` and ``stream`` raise ``SessionEnded``
(carrying ``reason`` and ``details``) unless the reason is one of
``CLEAN_TERMINAL_REASONS``, and all three verbs record ``Session.terminus``.
For two releases nothing in ``shared/scaffold`` mentioned any of it (#1063):
``explain clients`` still printed the pre-#1007 rule as fact, and the
templates -- the text ``jaato-scaffold new`` writes INTO a builder's first
driver -- caught ``ConnectionError``, ``SessionCreateFailed`` and
``AgentError`` and nothing else.

So a budget ceiling in a scaffolded client ended in a traceback, and a
scaffolded cascade printed ``completed`` for a stage a ceiling had stopped
-- dropping the reason in exactly the place #1007 made it available.

That is a drift defect, not a typo: the templates are prose about the SDK,
and prose does not fail when the SDK changes underneath it.  This test is
the tie.

WHAT THIS ASSERTS.  For every rendered template body:

* a body that calls a TURN VERB (``.ask(``/``.stream(``) also catches
  ``SessionEnded`` -- a terminal cutting the turn short is the one outcome
  those verbs signal by raising rather than by returning;
* a body that calls ``.complete(`` reads ``.terminus`` -- ``complete``
  returns ``None`` for several different endings and the payload cannot
  tell them apart, so a driver that never reads the terminus has thrown
  the distinction away.

WHAT THIS DOES NOT ASSERT.  Not that the handler does anything sensible
with what it caught, and not that the generated script runs -- the
emit-then-compile check in ``build`` covers syntax, and the conformance
suites cover behaviour.  This is the narrow property whose absence was the
defect: that the template acknowledges the outcome at all.

A body that uses no turn verb (the observer, which subscribes) is silent
here rather than exempted by name, so a template added later is covered by
what it does rather than by whether someone remembered to list it.

WHY IT LIVES IN ``shared/tests`` RATHER THAN BESIDE THE SCAFFOLD SUITE.
Its subject is ``shared/scaffold``, so ``shared/scaffold/tests`` is where it
belongs by topic -- and the reversion meta-guard walks only ``shared/tests``
and ``server/tests``, so a ``REVERSIONS`` list declared there is collected
by nothing and passes by never running (#1097).  Measured while writing
this: the meta-guard collected six of the eight reversions added in this
change and silently skipped the two in the scaffold package.  Until #1097
widens that walk, a guard that declares reversions has to sit where they
are exercised.
"""

from __future__ import annotations

import re
from typing import Dict, List

import pytest

from shared.scaffold import _client_templates as templates

#: Rendered template bodies, by the attribute that holds each.  Read from
#: the module rather than listed, so a template added later is checked
#: without an edit here -- the same reason #1063 happened is the reason not
#: to hardcode the set.
_TEMPLATE_ATTRS = tuple(
    name
    for name in dir(templates)
    if name.endswith("_TEMPLATE") and isinstance(getattr(templates, name), str)
)

#: ``s.ask(`` / ``await s.ask(`` / ``stage.stream(`` -- any attribute call.
_TURN_VERB_RE = re.compile(r"\.(?:ask|stream)\s*\(")
_COMPLETE_RE = re.compile(r"\.complete\s*\(")
_CATCHES_ENDED_RE = re.compile(r"except\s+SessionEnded\b")
_READS_TERMINUS_RE = re.compile(r"\.terminus\b")


def _bodies() -> Dict[str, str]:
    return {name: getattr(templates, name) for name in _TEMPLATE_ATTRS}


def _strip_comments_and_docstrings(body: str) -> str:
    """Drop ``#`` comments so PROSE about a verb is not read as a call.

    The templates discuss these verbs at length -- the client template's own
    comment says to switch to ``await s.complete(PROMPT)`` for a gated
    profile -- and a substring match over the raw text would require that
    template to catch a verb it deliberately does not call.  Same reasoning
    as ``build._referenced_names``, which reads the parse tree for exactly
    this hazard.
    """
    return "\n".join(
        line.split("#", 1)[0] if "#" in line else line
        for line in body.splitlines()
    )


def test_there_are_templates_to_check():
    """A rename that emptied the set would make every test below vacuous."""
    assert len(_TEMPLATE_ATTRS) >= 4, (
        f"found only {_TEMPLATE_ATTRS} template bodies in _client_templates "
        "-- if the naming convention changed, update _TEMPLATE_ATTRS; a "
        "green run over an empty set proves nothing"
    )


def test_at_least_one_template_uses_each_verb_shape():
    """Both rules below must bind on something, or they are decoration."""
    bodies = {n: _strip_comments_and_docstrings(b) for n, b in _bodies().items()}
    turn = [n for n, b in bodies.items() if _TURN_VERB_RE.search(b)]
    complete = [n for n, b in bodies.items() if _COMPLETE_RE.search(b)]
    assert turn, "no template calls .ask()/.stream() -- the first rule binds on nothing"
    assert complete, "no template calls .complete() -- the second rule binds on nothing"


@pytest.mark.parametrize("name", _TEMPLATE_ATTRS)
def test_a_turn_verb_template_catches_session_ended(name: str):
    """#1063: a budget ceiling in a scaffolded client ended in a traceback."""
    body = _strip_comments_and_docstrings(getattr(templates, name))
    if not _TURN_VERB_RE.search(body):
        pytest.skip(f"{name} calls no turn verb")
    assert _CATCHES_ENDED_RE.search(body), (
        f"{name} calls .ask()/.stream() and never catches SessionEnded. "
        "Since #1007 a plain turn CAN end the session -- a budget_control "
        "ceiling does -- and the verb then RAISES rather than returning "
        "text, so a scaffolded driver ends in a traceback. Catch it and "
        "report exc.reason / exc.details; a clean ending (see "
        "CLEAN_TERMINAL_REASONS) still returns normally."
    )


@pytest.mark.parametrize("name", _TEMPLATE_ATTRS)
def test_a_complete_template_reads_the_terminus(name: str):
    """``complete`` returns None for endings only the terminus distinguishes."""
    body = _strip_comments_and_docstrings(getattr(templates, name))
    if not _COMPLETE_RE.search(body):
        pytest.skip(f"{name} does not call .complete()")
    assert _READS_TERMINUS_RE.search(body), (
        f"{name} calls .complete() and never reads Session.terminus. "
        "None means the profile declared no completion schema, OR the "
        "model never signalled, OR a budget ceiling stopped the stage -- "
        "three endings with one return value. The terminus is what tells "
        "them apart, and it dies with the session object (#1007)."
    )


# ---------------------------------------------------------------------------
# Reversions -- the meta-suite
# (test_every_guard_detects_its_own_reversion) discovers this list by
# name and asserts each one makes the NAMED test fail.  A guard that
# cannot notice its own reversion is not evidence.
# ---------------------------------------------------------------------------
from shared.tests.reversion import (  # noqa: E402
    Reversion,
)

REVERSIONS = [
    Reversion(
        target="jaato-server/shared/scaffold/_client_templates.py",
        find='''    except SessionEnded as exc:
        # ask() raises when a terminal cuts the turn short''',
        replace='''    except _RemovedByReversion as exc:
        # ask() raises when a terminal cuts the turn short''',
        because=(
            "the host-tools template calls ask(), so dropping its "
            "SessionEnded handler is #1063 exactly: a budget ceiling in a "
            "scaffolded driver ends in a traceback instead of a named "
            "outcome"
        ),
        test="test_a_turn_verb_template_catches_session_ended",
    ),
    Reversion(
        target="jaato-server/shared/scaffold/_client_templates.py",
        find="        return payload, stage.terminus",
        replace="        return payload, None",
        because=(
            "dropping the terminus from the cascade stage throws away the "
            "one thing that tells a budget stop apart from a profile with "
            "no completion schema -- the loss #1007 made avoidable, in the "
            "template that taught it"
        ),
        test="test_a_complete_template_reads_the_terminus",
    ),
]
