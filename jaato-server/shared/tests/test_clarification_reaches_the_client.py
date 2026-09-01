"""A clarification the daemon asks for must be answerable by the client.

WHY THIS EXISTS.  ``request_clarification`` blocks its tool call until a
client answers.  So a clarification event the daemon emits and no client
dispatches on is not a cosmetic gap — it is a turn that never ends.  That
shipped: the daemon emitted ``ClarificationBatchEvent`` for every
runner-tier session and ``jaato-tui/rich_client.py`` had no branch for
it, so the questions arrived, were never rendered, and the session had to
be abandoned (#704).  Ctrl+C does not cancel a turn blocked inside a tool
call, and later input queues behind it, so the only recovery was killing
the client and restarting the daemon.

Nothing caught it because both halves were individually correct: the
server emitted a well-formed event that the SDK defines and serialises,
and the client's ``elif isinstance(...)`` chain simply fell through.  A
silent drop has no failing assertion anywhere — which is what a guard is
for.

WHAT IT CHECKS.

1. Every ``Clarification*Event`` the daemon CONSTRUCTS in production code
   is classified in :data:`OBLIGATION` below.  A new emitter forces a
   decision instead of defaulting to "ignored by everyone".
2. Every classification is live — an entry for an event nothing emits any
   more is stale and must be dropped, the same discipline the complexity
   ratchet and the session-env audit apply to their tables.
3. Every :data:`PROMPT` event — one the daemon blocks on — is dispatched
   by every shipped client, meaning the client's event loop names it in
   an ``isinstance`` check.  Importing the symbol is NOT enough: the
   original defect had the client importing the sibling events it did
   handle while the batch event fell through, and a guard that matched on
   the import would have passed over exactly that.

WHAT IT DELIBERATELY DOES NOT CHECK.  Whether the handler is *correct* —
that a rendered question is legible, that the answers map back to the
right questions.  This guard draws the line at the failure mode that has
no other symptom: a prompt that is never shown, and a turn that never
ends.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Set

import pytest


ROOT = Path(__file__).resolve().parents[3]

#: Production trees whose event constructions count as "the daemon emits
#: this".  Test files are excluded: a fixture building an event is not the
#: daemon deciding to send one.
SERVER_TREE = ROOT / "jaato-server"

#: Every client shipped from this repository, by the module that owns its
#: event loop.  A client that is not listed is a client this guard does
#: not protect.
CLIENTS = (
    "jaato-tui/rich_client.py",
    "jaato-tui/headless_mode.py",
)

PROMPT = "prompt"    #: the daemon BLOCKS until a client answers this
NOTICE = "notice"    #: informational; a client may ignore it

#: What each emitted clarification event obliges a client to do.
#:
#: ``PROMPT`` is not a style preference — it is the statement that a tool
#: call is parked on this event and stays parked until the client replies.
#: Marking a new event ``NOTICE`` is a claim that nothing waits on it;
#: pair it with a comment saying what makes that true.
OBLIGATION: Dict[str, str] = {
    # server/core.py, per question, daemon-local sessions: the QueueChannel
    # is sitting on the input queue when this goes out.
    "ClarificationInputModeEvent": PROMPT,
    # server/core.py (preview, batch_only=False) and
    # runner_rpc_handlers/clarification_relay.py (batch_only=True, the whole
    # request).  The relay's RPC future is unresolved until the client sends
    # ClarificationBatchResponseEvent back — this is #704's event.
    "ClarificationBatchEvent": PROMPT,
    # server/core.py, after the answers are in.  Reports the Q&A summary so
    # a client can close out its tool tree; nothing waits on it, and a
    # client with nothing to redraw (headless) ignores it.
    "ClarificationResolvedEvent": NOTICE,
}


def _python_sources(root: Path) -> list[Path]:
    """Production ``.py`` files under *root* — no tests, no caches."""
    return [
        p for p in root.rglob("*.py")
        if "tests" not in p.parts
        and not p.name.startswith("test_")
        and "__pycache__" not in p.parts
    ]


def _constructed_event_names(path: Path) -> Set[str]:
    """``Clarification*Event`` classes instantiated in *path*."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - not our file
        return set()
    found: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else (
            func.attr if isinstance(func, ast.Attribute) else "")
        if name.startswith("Clarification") and name.endswith("Event"):
            found.add(name)
    return found


def _emitted_by_the_daemon() -> Set[str]:
    """Every clarification event production server code constructs."""
    emitted: Set[str] = set()
    for path in _python_sources(SERVER_TREE):
        emitted |= _constructed_event_names(path)
    return emitted


def _dispatched_names(path: Path) -> Set[str]:
    """Class names this client branches on via ``isinstance``.

    Reads the second argument of every ``isinstance`` call, unpacking the
    tuple form, so a client that folds several event types into one branch
    counts for each of them.  An imported-but-unbranched name does not
    count — that is precisely the state #704 shipped in.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "isinstance"):
            continue
        if len(node.args) < 2:
            continue
        candidates = (node.args[1].elts
                      if isinstance(node.args[1], ast.Tuple)
                      else [node.args[1]])
        names |= {c.id for c in candidates if isinstance(c, ast.Name)}
    return names


def test_the_scan_finds_the_emitters():
    """Anchor: a scan that matches nothing would pass everything below.

    The guards this repository has already caught reporting green were
    all exercising an empty set.  Assert the shape of the result, not
    just that the scan ran.
    """
    emitted = _emitted_by_the_daemon()
    assert "ClarificationBatchEvent" in emitted, (
        "the emitter scan no longer sees ClarificationBatchEvent, which "
        "server/runner_rpc_handlers/clarification_relay.py emits for every "
        "runner-tier clarification. Either the scan broke or the emitter "
        "moved; either way this guard is not checking what it claims to."
    )
    assert "ClarificationInputModeEvent" in emitted


def test_every_emitted_clarification_event_is_classified():
    """A new emitter must be classified, not silently ignored by clients."""
    emitted = _emitted_by_the_daemon()
    unclassified = sorted(emitted - set(OBLIGATION))
    assert not unclassified, (
        f"clarification events emitted by the daemon with no entry in "
        f"OBLIGATION: {unclassified}. Add one. If a tool call waits for a "
        f"client reply to it, it is PROMPT and every client in CLIENTS "
        f"must branch on it; if nothing waits, it is NOTICE — say what "
        f"makes that true in a comment."
    )


def test_no_classification_is_stale():
    """An entry for an event nothing emits any more must be dropped."""
    emitted = _emitted_by_the_daemon()
    stale = sorted(set(OBLIGATION) - emitted)
    assert not stale, (
        f"OBLIGATION classifies events the daemon no longer emits: {stale}. "
        f"Drop the entries. A table that outlives its subject stops being "
        f"read, and then stops being true."
    )


@pytest.mark.parametrize("client", CLIENTS)
def test_every_prompting_event_is_dispatched_by_every_client(client):
    """The #704 assertion: no PROMPT event may fall through a client.

    A client that does not branch on a prompting event leaves the tool
    call — and the turn behind it — blocked with no way out, because
    cancelling a turn does not interrupt a tool that is waiting.
    """
    path = ROOT / client
    assert path.is_file(), (
        f"{client} is listed in CLIENTS but does not exist. A client this "
        f"guard cannot read is a client it is not protecting."
    )
    dispatched = _dispatched_names(path)
    required = {n for n, kind in OBLIGATION.items() if kind == PROMPT}
    missing = sorted(required - dispatched)
    assert not missing, (
        f"{client} never branches on {missing}. The daemon blocks a tool "
        f"call on each of these until the client answers, so an unhandled "
        f"one hangs the session with no recovery short of restarting the "
        f"daemon (#704). Importing the symbol is not handling it: add an "
        f"isinstance branch that prompts and replies."
    )


# ---------------------------------------------------------------------------
# Self-certification for test_every_guard_detects_its_own_reversion.
# ---------------------------------------------------------------------------

from shared.tests.test_every_guard_detects_its_own_reversion import Reversion  # noqa: E402


REVERSIONS = [
    Reversion(
        target="jaato-tui/rich_client.py",
        find="elif isinstance(event, (ClarificationInputModeEvent, ClarificationBatchEvent)):",
        replace="elif isinstance(event, ClarificationInputModeEvent):",
        because=(
            "the TUI stops branching on ClarificationBatchEvent — #704 "
            "exactly: the import stays, the questions still arrive, and "
            "the turn blocks forever with nothing rendered."
        ),
        test="test_every_prompting_event_is_dispatched_by_every_client"
             "[jaato-tui/rich_client.py]",
    ),
]
