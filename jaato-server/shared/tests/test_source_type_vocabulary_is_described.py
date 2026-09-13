"""Every ``SourceType`` a caller may send is named where callers read.

WHAT WENT WRONG.  ``session.offer_message`` derives the set of accepted
``source_type`` strings from :class:`shared.message_queue.SourceType` itself --
``SourceType(source_type_str)``, with a rejection message that renders
``sorted(s.value for s in SourceType)``.  So the daemon has always accepted all
six.  The four places that DESCRIBE the vocabulary were each hand-typed, and
each was wrong about the same member:

===============================================  ==========================
surface                                          said
===============================================  ==========================
``SourceType``'s own class docstring             idle-only is "CHILD, PEER"
                                                 -- there is no ``PEER``
                                                 member; the comment three
                                                 lines below even warns that
                                                 ``EventType.PEER_*`` is a
                                                 different thing
``InjectPromptRequest``'s docstring              five values, no ``sibling``
``InjectPromptRequest.source_type``'s comment    five values, no ``sibling``
``IPCClient.inject_prompt``'s ``Args:``          five values, no ``sibling``
===============================================  ==========================

``sibling`` is not decorative: it is the one idle-only tier that is NOT
``child``, drained by its own ``get_sibling_message`` /
``has_sibling_messages`` path, and the framework itself sends it
(``session_manager`` injects ``SourceType.SIBLING`` for cascade-peer
coordination).  A builder reading the SDK could not learn it exists, so the
reachable mistake is reaching for ``"child"`` -- a different drain path -- to
do a sibling's job.

WHY THE ASSERTION IS SHAPED LIKE THIS.  The enum is the authority the daemon
already consults, so each surface is checked AGAINST IT rather than against a
literal repeated here -- a literal would be a fifth copy of the thing that
drifted.  A member added to ``SourceType`` tomorrow fails every test below
until the surfaces that teach the vocabulary mention it.

``shared`` may not be imported from ``jaato-sdk``, which is why the SDK cannot
simply render this list at runtime and why the copies exist at all.  This test
lives on the server side, where both are importable.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

from shared.message_queue import (
    HIGH_PRIORITY_SOURCES,
    IDLE_ONLY_SOURCES,
    SourceType,
)
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

REVERSIONS = [
    Reversion(
        target="jaato-server/shared/message_queue.py",
        find="    - Low priority (idle-only): CHILD, SIBLING",
        replace="    - Low priority (idle-only): CHILD, PEER",
        test="test_the_enum_docstring_names_every_member",
        because="SourceType's own docstring naming a member that does not exist",
    ),
    Reversion(
        target="jaato-sdk/jaato_sdk/events.py",
        find='    * ``"sibling"`` — SIBLING priority (idle-only, like ``"child"``,',
        replace='    * ``"nonesuch"`` — not a member (idle-only, like ``"child"``,',
        test="test_the_inject_request_docstring_names_every_value",
        because="the InjectPromptRequest docstring omitting an accepted value",
    ),
    Reversion(
        target="jaato-sdk/jaato_sdk/events.py",
        find='    # "user" | "child" | "sibling" | "system" | "event" | "parent"\n'
             '    source_type: str = "user"',
        replace='    source_type: str = "user"  '
                '# "user" | "child" | "system" | "event" | "parent"',
        test="test_the_source_type_field_comment_names_every_value",
        because="the field's own comment omitting an accepted value",
    ),
    Reversion(
        target="jaato-sdk/jaato_sdk/client/ipc.py",
        find='                ``"child"`` (follow-up), ``"sibling"`` (idle-only',
        replace='                ``"child"`` (follow-up), ``"nobody"`` (idle-only',
        test="test_the_inject_prompt_method_docstring_names_every_value",
        because="IPCClient.inject_prompt's Args omitting an accepted value",
    ),
]

#: Repository root -- this file is <root>/jaato-server/shared/tests/<me>.py
ROOT = Path(__file__).resolve().parents[3]

#: Every value the daemon will accept, read from the enum it reads.
VALUES = tuple(m.value for m in SourceType)

#: Every member NAME, for the surface that speaks in names rather than values.
NAMES = tuple(m.name for m in SourceType)


def _missing(haystack: str, needles) -> list:
    """The needles absent from *haystack*, in declaration order."""
    return [n for n in needles if n not in haystack]


def _tier_line(doc: str, label: str) -> set:
    """The member names the docstring's ``label`` tier line lists."""
    m = re.search(rf"^\s*-\s*{label}[^:]*:\s*(.+)$", doc, re.M)
    assert m, (f"SourceType's docstring no longer carries a '{label}' tier "
               f"line -- the guard's anchor moved")
    return {w.strip() for w in m.group(1).split(",") if w.strip()}


def test_the_enum_docstring_names_every_member():
    """SourceType's two tier lines must agree with the two frozensets.

    Asserting merely that each name appears SOMEWHERE in the docstring is
    decorative -- the prose below the tier lines mentions ``SIBLING`` by name,
    so the tier line could go on saying ``PEER`` and the check would pass.
    (It did: the meta-guard rejected exactly that draft.)  The claim worth
    guarding is the PARTITION: the docstring says which members may interrupt
    a turn, and ``HIGH_PRIORITY_SOURCES`` / ``IDLE_ONLY_SOURCES`` are what
    actually decide it.
    """
    doc = inspect.getdoc(SourceType) or ""
    high = {m.name for m in HIGH_PRIORITY_SOURCES}
    idle = {m.name for m in IDLE_ONLY_SOURCES}

    assert _tier_line(doc, "High priority") == high, (
        "SourceType's docstring lists the mid-turn tier as "
        f"{sorted(_tier_line(doc, 'High priority'))}, but "
        f"HIGH_PRIORITY_SOURCES is {sorted(high)}.  Tier membership is an "
        "authority statement, so a docstring that disagrees with the "
        "frozenset misstates who may interrupt a turn."
    )
    assert _tier_line(doc, "Low priority") == idle, (
        "SourceType's docstring lists the idle-only tier as "
        f"{sorted(_tier_line(doc, 'Low priority'))}, but IDLE_ONLY_SOURCES "
        f"is {sorted(idle)}."
    )
    assert high | idle == set(NAMES), (
        f"the two frozensets do not partition SourceType: "
        f"{sorted(set(NAMES) - (high | idle))} sit in neither tier and would "
        f"never drain."
    )


def test_the_inject_request_docstring_names_every_value():
    """The wire event's docstring is the protocol-level description."""
    from jaato_sdk.events import InjectPromptRequest
    doc = inspect.getdoc(InjectPromptRequest) or ""
    assert not _missing(doc, VALUES), (
        "InjectPromptRequest's docstring does not mention "
        f"{_missing(doc, VALUES)}.  The daemon accepts every SourceType "
        "value (it calls SourceType(...) on the string), so an omitted one "
        "is a working queue priority the SDK hides."
    )


def test_the_source_type_field_comment_names_every_value():
    """The comment on the field itself -- what an IDE shows at the call site."""
    src = (ROOT / "jaato-sdk/jaato_sdk/events.py").read_text()
    m = re.search(r"^([^\n]*)\n\s*source_type: str = \"user\"", src, re.M)
    assert m, ("could not find InjectPromptRequest.source_type's declaration "
               "in jaato_sdk/events.py -- the guard's anchor moved")
    comment = m.group(1)
    assert not _missing(comment, VALUES), (
        f"the comment on InjectPromptRequest.source_type omits "
        f"{_missing(comment, VALUES)}.  It reads {comment.strip()!r}; the "
        f"accepted set is {list(VALUES)}."
    )


def test_the_inject_prompt_method_docstring_names_every_value():
    """The method a builder actually calls, and the last place to say it."""
    from jaato_sdk.client.ipc import IPCClient
    doc = inspect.getdoc(IPCClient.inject_prompt) or ""
    assert not _missing(doc, VALUES), (
        "IPCClient.inject_prompt's docstring does not mention "
        f"{_missing(doc, VALUES)}.  This is the surface a builder reads "
        "before choosing a queue priority; an omitted value is one they "
        "cannot choose without reading server source."
    )
