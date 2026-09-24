"""A session waiting on a human says so on the listing (#1138).

One session raises a permission ASK or a ``request_clarification`` while you
are working in another.  The turn is BLOCKED, and nothing tells you: prompt
events are delivered to ``session.attached_clients`` and
``_client_to_session`` is 1:1, so a client attached to session A is not in
B's set and B's ``PermissionRequestedEvent`` never reaches it.
``SessionManager.broadcast_event`` is not the escape hatch either -- its
docstring reserves it for events that are NOT tied to a specific session.

So the fact rides the listing every client already polls:
``RuntimeSessionInfo.awaiting`` (``"permission"`` / ``"clarification"`` /
``None``) plus ``awaiting_since``.

THE FINDING THIS MODULE EXISTS TO PIN
======================================

The issue reads the daemon's two fields --
``JaatoServer._pending_permission_request_id`` and
``_pending_clarification_request_id`` -- and calls the change "the same read,
one field over" from ``session.server.is_processing``.

**Those two fields are never written on the default path.**  They are set by
the DAEMON-side hooks in ``_setup_permission_hooks`` /
``_setup_clarification_hooks``, and ``permission``, ``clarification`` and
``references`` are all ``PLUGIN_TIER = "runner"`` -- so on a runner-served
session the plugin that raises the prompt lives in the runner process and the
daemon-side hook is not in the loop.  ``PromptOperatorHandler``'s own comments
say it outright: *"that path is dead post-§7c since the runner-side permission
plugin is the one in the loop, and its ASK arrives HERE via the PromptOperator
RPC instead"*.

Implemented literally, therefore, ``awaiting`` would have reported ``None``
for exactly the sessions the issue is about -- a field resolved, carried,
rendered and armed on nobody (#1133), and the #735 shape of a cap that
silently does not apply.  ``test_the_field_the_issue_named_is_empty_on_the_
default_path`` is that measurement, kept as a test so the reason both holders
are read cannot quietly stop being true.

WHAT EACH CASE PINS
===================

``test_a_runner_served_permission_ask_is_visible``
``test_a_runner_served_clarification_is_visible``
    The DEFAULT path, driven through the real relay handlers' ``handle``
    coroutine rather than by poking their dicts -- a test that registered
    the future by hand would pass against a handler that never stamped.

``test_a_daemon_local_ask_is_visible``
    The embedded / standalone-WS path, and the legacy fallback
    ``respond_to_permission`` still documents as Path 2.  A SEPARATE door
    from the one above: neutralising either branch leaves the other's case
    green, which is what stops the pair being one check on one door.

``test_a_relayed_ask_is_dated``
    Open question 2, decided YES.  The listing is a POLL, so a client that
    derived the wait from when IT first saw the flag would under-report
    every wait that predates it and reset to zero on every reconnect.

``test_the_oldest_pending_prompt_wins``
    Both kinds can be in flight at once -- tool execution is 8-wide, so one
    tool can hit a permission gate while another asks a question -- and the
    pair has room for one.  Oldest-first, because that is the question a
    reader is asking and because it needs no invented ranking between kinds.

``test_the_listing_reports_a_blocked_session`` (``list_sessions``)
``test_the_listing_row_carries_awaiting`` (the ``session.list`` wire dict)
    Two layers, because a value computed and not delivered is #1133.

``test_is_processing_stays_true_while_the_session_waits``
    The issue's own argument for why ``is_processing`` could not carry this.

``test_a_persisted_only_row_never_claims_to_be_waiting``
    In-memory branch only: a session that is not loaded has no ``server``
    and cannot be waiting on anything.

``test_an_unknown_kind_never_reaches_the_wire``
``test_a_server_that_answers_nothing_degrades_to_silence``
    The vocabulary is closed, and it is closed AT THE BOUNDARY because
    ``awaiting_of`` accepts any duck-typed server (the ``_turns_ran_snapshot``
    precedent, #881).  A listing that raised because one session's server
    answered oddly would be a worse failure than the fact it was reporting.

``test_the_relays_read_a_snapshot_not_the_live_dict`` (AST)
    The read and the write are on DIFFERENT threads: ``list_sessions`` runs
    on a transport thread while ``handle`` registers and drops entries on
    the daemon's asyncio loop.  #938's invariant applies unchanged -- every
    read path iterates a snapshot, never the live container -- and it is
    pinned at source level because a threaded version would be a timing bet
    and the property is textual.

NO CASE HERE SLEEPS.  The relay stamps ``time.time()``, so the dating case
asserts a bound around the instant it recorded itself rather than waiting for
a clock to move.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import pathlib
import threading
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jaato_sdk.events import SessionListEvent

from jaato_server.server.awaiting import (
    AWAITING_CLARIFICATION,
    AWAITING_KINDS,
    AWAITING_PERMISSION,
    PendingPrompt,
    awaiting_fields,
    awaiting_of,
    resolve_awaiting,
)
from jaato_server.server.command_router import CommandRouter
from jaato_server.server.core import JaatoServer
from jaato_server.server.runner_rpc_handlers.clarification_relay import ClarificationRelayHandler
from jaato_server.server.runner_rpc_handlers.prompt_operator import PromptOperatorHandler
from jaato_server.server.session_manager import RuntimeSessionInfo, Session, SessionManager
from jaato_server.shared.tests.reversion import Reversion


_AWAITING = "jaato-server/jaato_server/server/awaiting.py"
_MANAGER = "jaato-server/jaato_server/server/session_manager.py"
_ROUTER = "jaato-server/jaato_server/server/command_router.py"
_PROMPT_OPERATOR = "jaato-server/jaato_server/server/runner_rpc_handlers/prompt_operator.py"


REVERSIONS = [
    Reversion(
        target=_AWAITING,
        find="""    if relay is not None:
        probe = getattr(relay, "has_pending_prompt", None)
        if callable(probe) and probe():""",
        replace="""    if relay is not None:
        probe = getattr(relay, "has_pending_prompt", None)
        if False:""",
        test="test_a_runner_served_permission_ask_is_visible",
        because=(
            "the runner-RPC relay is not consulted, so on the DEFAULT path -- "
            "where the permission plugin is runner-tier and the daemon-side "
            "fields are never written -- a blocked session reports nothing, "
            "which is the state the issue is about"
        ),
    ),
    Reversion(
        target=_AWAITING,
        find="""    if local_request_id:
        return PendingPrompt(kind, local_since)""",
        replace="""    if False:
        return PendingPrompt(kind, local_since)""",
        test="test_a_daemon_local_ask_is_visible",
        because=(
            "the daemon-local holder is not consulted, so an embedded or "
            "standalone-WS session -- and the legacy Path 2 fallback -- goes "
            "back to being invisible while the runner path is covered"
        ),
    ),
    Reversion(
        target=_AWAITING,
        find="        return min(dated, key=lambda p: (p.since, _RANK[p.kind]))",
        replace="        return max(dated, key=lambda p: (p.since, _RANK[p.kind]))",
        test="test_the_oldest_pending_prompt_wins",
        because=(
            "with two prompts in flight the pair describes the NEWEST, so the "
            "one that has been waiting longest is the one nobody is told about"
        ),
    ),
    Reversion(
        target=_PROMPT_OPERATOR,
        find="""        self._pending[payload.request_id] = fut
        self._raised_at[payload.request_id] = time.time()""",
        replace="""        self._pending[payload.request_id] = fut""",
        test="test_a_relayed_ask_is_dated",
        because=(
            "nothing dates the ASK at the raise site, so awaiting_since is "
            "absent and a client is back to timing the wait from when it "
            "first happened to poll"
        ),
    ),
    Reversion(
        target=_PROMPT_OPERATOR,
        find="""        stamps = self._raised_at
        return oldest([
            stamp for stamp in
            (stamps.get(request_id) for request_id in list(self._pending))
            if stamp is not None
        ])""",
        replace="""        return oldest([
            self._raised_at[request_id]
            for request_id in self._pending
            if request_id in self._raised_at
        ])""",
        test="test_the_relays_read_a_snapshot_not_the_live_dict",
        because=(
            "the listing thread iterates a dict the daemon's asyncio loop is "
            "mutating, so a prompt resolving mid-listing raises "
            "'dictionary changed size during iteration' INSIDE the listing "
            "-- #938's shape, and for a fact the listing is only reporting"
        ),
    ),
    Reversion(
        target=_MANAGER,
        find="""                    awaiting=awaiting,
                    awaiting_since=awaiting_since,""",
        replace="""                    awaiting=None,
                    awaiting_since=None,""",
        test="test_the_listing_reports_a_blocked_session",
        because=(
            "list_sessions computes the fact and drops it, so every row "
            "reports an idle-looking session however long it has been blocked"
        ),
    ),
    Reversion(
        target=_ROUTER,
        find="""            "awaiting": s.awaiting,
            "awaiting_since": s.awaiting_since,""",
        replace="""            "_awaiting_withheld": None,""",
        test="test_the_listing_row_carries_awaiting",
        because=(
            "the value is resolved and reaches no client -- #1133's shape, "
            "and the one a test of list_sessions alone would not notice"
        ),
    ),
]


# ------------------------------------------------------------------ harness

def _server(
    *,
    permission_relay=None,
    clarification_relay=None,
    permission_id=None,
    permission_since=None,
    clarification_id=None,
    clarification_since=None,
    running=True,
) -> JaatoServer:
    """A ``JaatoServer`` carrying only what ``awaiting_prompt`` reads.

    ``__new__`` rather than the real constructor -- the pattern
    ``test_respond_to_permission_routing_step7_3.py`` already uses on this
    class, whose ``__init__`` stands up a registry, a provider and a
    formatter pipeline, none of which this read touches.

    Args:
        permission_relay: A ``PromptOperatorHandler``, or ``None`` for a
            session with no runner (the embedded path).
        clarification_relay: A ``ClarificationRelayHandler``, likewise.
        permission_id: The daemon-local pending permission id.
        permission_since: Its stamp, epoch seconds.
        clarification_id: The daemon-local pending clarification id.
        clarification_since: Its stamp, epoch seconds.
        running: What ``is_processing`` answers.
    """
    srv = JaatoServer.__new__(JaatoServer)
    srv._prompt_operator_handler = permission_relay
    srv._clarification_relay_handler = clarification_relay
    srv._pending_permission_request_id = permission_id
    srv._pending_permission_since = permission_since
    srv._pending_clarification_request_id = clarification_id
    srv._pending_clarification_since = clarification_since
    srv._model_running = running
    # The rest of what the listing's in-memory branch reads off the server.
    # Set here rather than mocked so ``list_sessions`` runs for real: the
    # case under test is that ONE more read joined the ones already there.
    srv._model_provider = "echo"
    srv._model_name = "echo-1"
    srv.get_history = lambda: []
    return srv


async def _settle() -> None:
    """Let a just-created ``handle()`` task register and emit.

    Two ticks rather than one: ``handle`` awaits nothing before registering,
    but the task has to be scheduled first and then run to its own await.
    """
    await asyncio.sleep(0)
    await asyncio.sleep(0)


async def _raise_permission(
    handler: PromptOperatorHandler, request_id: str = "req-1",
) -> "asyncio.Task":
    """Drive a real relayed ASK and leave it unanswered."""
    task = asyncio.create_task(handler.handle({
        "request_id": request_id,
        "tool_name": "writeNewFile",
        "tool_args": {"path": "/tmp/x"},
        "response_options": [{"key": "y", "label": "yes", "action": "allow"}],
        "agent_id": "main",
    }))
    await _settle()
    return task


async def _raise_clarification(
    handler: ClarificationRelayHandler, request_id: str = "clar-1",
) -> "asyncio.Task":
    """Drive a real relayed clarification batch and leave it unanswered."""
    task = asyncio.create_task(handler.handle({
        "request_id": request_id,
        "tool_name": "request_clarification",
        "questions": [{"text": "which branch?", "type": "free_text"}],
        "agent_id": "main",
    }))
    await _settle()
    return task


async def _abandon(*tasks: "asyncio.Task") -> None:
    """Tear in-flight prompts down so no case leaks a pending task."""
    for task in tasks:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


def _sm() -> SessionManager:
    """A ``SessionManager`` with only what ``list_sessions`` reads.

    Same construction pattern as ``test_unload_grace_1106.py``: the real
    ``__init__`` stands up transports, a workspace index on the real
    ``~/.jaato`` and a plugin registry, and the listing reads none of them.
    """
    sm = SessionManager.__new__(SessionManager)
    sm._sessions = {}
    sm._lock = threading.RLock()
    sm._client_config = {}
    sm._orphan_since = {}
    sm._ever_attached = set()
    sm._normalize_workspace = lambda p: p
    sm._session_workspace_index = MagicMock()
    sm._session_workspace_index.workspaces.return_value = []
    # No persisted rows: this module is about the in-memory overlay, and a
    # real workspace scan would reach the developer's own ``~/.jaato``.
    sm._get_persisted_sessions = lambda workspace_path=None: []
    sm._session_config = SimpleNamespace(storage_path=".jaato/sessions")
    return sm


def _session(sid: str, server: JaatoServer) -> Session:
    session = Session(
        session_id=sid,
        name=sid,
        server=server,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    session.workspace_path = "/ws"
    return session


def _row(sm: SessionManager, sid: str) -> RuntimeSessionInfo:
    return {info.session_id: info for info in sm.list_sessions()}[sid]


def _router(rows):
    """A ``CommandRouter`` whose manager answers with *rows*."""
    manager = MagicMock()
    manager.list_sessions.return_value = rows
    sink = MagicMock(spec=["send_event", "get_client_user",
                           "get_client_workspace", "set_client_session"])
    sink.get_client_user.return_value = None
    router = CommandRouter.__new__(CommandRouter)
    router._session_manager = manager
    router._event_sink = sink
    return router, sink


def _wire_rows(sink):
    event = [
        call[0][1] for call in sink.send_event.call_args_list
        if isinstance(call[0][1], SessionListEvent)
    ][0]
    return event.sessions


# ------------------------------------------------- the default (runner) path

async def test_a_runner_served_permission_ask_is_visible():
    """The path the issue's own two fields do not cover."""
    relay = PromptOperatorHandler(emit_event=lambda _event: None)
    server = _server(permission_relay=relay)

    assert awaiting_of(server) == (None, None)

    task = await _raise_permission(relay)
    try:
        kind, _since = awaiting_of(server)
        assert kind == AWAITING_PERMISSION, (
            "a runner-served session blocked on a permission ASK reported "
            f"{kind!r}; the ASK is pending in the relay, not in "
            "_pending_permission_request_id"
        )
    finally:
        await _abandon(task)


async def test_a_runner_served_clarification_is_visible():
    relay = ClarificationRelayHandler(emit_event=lambda _event: None)
    server = _server(clarification_relay=relay)

    assert awaiting_of(server) == (None, None)

    task = await _raise_clarification(relay)
    try:
        assert awaiting_of(server)[0] == AWAITING_CLARIFICATION
    finally:
        await _abandon(task)


async def test_the_field_the_issue_named_is_empty_on_the_default_path():
    """The measurement the whole shape of this change rests on.

    The relay holds the pending ASK and ``_pending_permission_request_id``
    stays ``None`` -- which is why ``awaiting_prompt`` reads BOTH holders
    rather than the one the issue named.  If this ever fails because the
    daemon-side field started being written on the runner path too, the
    relay read has become redundant and the reason recorded here should be
    revisited rather than the assertion relaxed.
    """
    relay = PromptOperatorHandler(emit_event=lambda _event: None)
    server = _server(permission_relay=relay)
    task = await _raise_permission(relay)
    try:
        assert relay.has_pending_prompt(), "the relay did not register the ASK"
        assert server._pending_permission_request_id is None, (
            "the daemon-side field IS written on the runner path now"
        )
        # And the listing still answers, because it does not read that field.
        assert awaiting_of(server)[0] == AWAITING_PERMISSION
    finally:
        await _abandon(task)


async def test_an_answered_prompt_stops_being_reported():
    relay = PromptOperatorHandler(emit_event=lambda _event: None)
    server = _server(permission_relay=relay)
    task = await _raise_permission(relay)
    assert awaiting_of(server)[0] == AWAITING_PERMISSION

    relay.resolve_response("req-1", "y")
    await task

    assert awaiting_of(server) == (None, None)


async def test_a_shutdown_leaves_no_stale_claim():
    """A torn-down relay must not keep a session looking blocked."""
    relay = PromptOperatorHandler(emit_event=lambda _event: None)
    server = _server(permission_relay=relay)
    task = await _raise_permission(relay)
    try:
        relay.shutdown()
        assert relay.pending_since() is None
        assert awaiting_of(server) == (None, None)
    finally:
        await _abandon(task)


# ------------------------------------------------------- the daemon-local path

def test_a_daemon_local_ask_is_visible():
    """The embedded / standalone-WS path, and the legacy Path 2 fallback."""
    server = _server(permission_id="req-legacy", permission_since=1_000.0)
    assert awaiting_of(server) == (
        AWAITING_PERMISSION, "1970-01-01T00:16:40+00:00")


def test_a_daemon_local_clarification_is_visible():
    server = _server(clarification_id="clar-legacy")
    kind, since = awaiting_of(server)
    assert kind == AWAITING_CLARIFICATION
    assert since is None, "an undated holder must report NOT MEASURED"


async def test_the_relay_outranks_the_daemon_local_field_when_both_are_set():
    """Both holders populated: the live one wins, and it is dated.

    Not a contrived state -- a session can carry a relay AND have taken the
    daemon-local fallback ``respond_to_permission`` documents.
    """
    relay = PromptOperatorHandler(emit_event=lambda _event: None)
    server = _server(permission_relay=relay, permission_id="stale",
                     permission_since=1_000.0)
    task = await _raise_permission(relay)
    try:
        _kind, since = awaiting_of(server)
        assert since is not None
        assert since != "1970-01-01T00:16:40+00:00", (
            "the stale daemon-local stamp was reported over the live relay's"
        )
    finally:
        await _abandon(task)


# --------------------------------------------------------------- the clock

async def test_a_relayed_ask_is_dated():
    """Open question 2, decided YES -- and dated at the RAISE site."""
    before = datetime.now(timezone.utc)
    relay = PromptOperatorHandler(emit_event=lambda _event: None)
    server = _server(permission_relay=relay)
    task = await _raise_permission(relay)
    after = datetime.now(timezone.utc)
    try:
        _kind, since = awaiting_of(server)
        assert since is not None, (
            "awaiting_since is absent, so a client can only date the wait "
            "from its own first sighting"
        )
        raised = datetime.fromisoformat(since)
        assert raised.tzinfo is not None, "the stamp must carry its zone"
        assert before <= raised <= after, (
            f"{raised} is not the instant the ASK was raised "
            f"({before}..{after})"
        )
    finally:
        await _abandon(task)


def test_the_stamp_is_wall_clock_so_a_browser_can_subtract_it():
    """Not monotonic, though the daemon's own bounds are.

    A monotonic instant means nothing outside the process that produced it,
    and this number's only consumer is a client rendering "waiting 4 min".
    """
    _kind, since = awaiting_fields(PendingPrompt(AWAITING_PERMISSION, 0.0))
    assert since == "1970-01-01T00:00:00+00:00"


def test_awaiting_since_is_a_second_field_not_a_widening():
    """The degradation argument depends on ``awaiting`` staying a scalar."""
    kind, since = awaiting_fields(PendingPrompt(AWAITING_CLARIFICATION, 5.0))
    assert isinstance(kind, str)
    assert isinstance(since, str)
    fields = {f.name for f in RuntimeSessionInfo.__dataclass_fields__.values()}
    assert {"awaiting", "awaiting_since"} <= fields


# ------------------------------------------------------------- precedence

def test_the_oldest_pending_prompt_wins():
    """Both kinds in flight: the one waiting longest is the one reported."""
    winner = resolve_awaiting([
        PendingPrompt(AWAITING_PERMISSION, 200.0),
        PendingPrompt(AWAITING_CLARIFICATION, 100.0),
    ])
    assert winner.kind == AWAITING_CLARIFICATION
    assert winner.since == 100.0


def test_an_undated_pair_still_answers_deterministically():
    """Two dicts must not decide a UI marker by iteration order."""
    forwards = resolve_awaiting([
        PendingPrompt(AWAITING_PERMISSION, None),
        PendingPrompt(AWAITING_CLARIFICATION, None),
    ])
    backwards = resolve_awaiting([
        PendingPrompt(AWAITING_CLARIFICATION, None),
        PendingPrompt(AWAITING_PERMISSION, None),
    ])
    assert forwards == backwards == PendingPrompt(AWAITING_PERMISSION, None)


def test_a_dated_prompt_outranks_an_undated_one():
    """An absent stamp is NOT MEASURED, so it cannot win an age contest."""
    winner = resolve_awaiting([
        PendingPrompt(AWAITING_PERMISSION, None),
        PendingPrompt(AWAITING_CLARIFICATION, 100.0),
    ])
    assert winner == PendingPrompt(AWAITING_CLARIFICATION, 100.0)


def test_nothing_pending_is_nothing_reported():
    assert resolve_awaiting([]) is None
    assert awaiting_fields(None) == (None, None)
    assert awaiting_of(_server()) == (None, None)


# ------------------------------------------------------------- the listing

def test_the_listing_reports_a_blocked_session():
    sm = _sm()
    blocked = _server(permission_id="req-1", permission_since=1_000.0)
    idle = _server()
    sm._sessions = {
        "blocked": _session("blocked", blocked),
        "idle": _session("idle", idle),
    }

    assert _row(sm, "blocked").awaiting == AWAITING_PERMISSION
    assert _row(sm, "blocked").awaiting_since == "1970-01-01T00:16:40+00:00"
    assert _row(sm, "idle").awaiting is None
    assert _row(sm, "idle").awaiting_since is None


def test_is_processing_stays_true_while_the_session_waits():
    """Why ``is_processing`` could not have carried this.

    A session blocked on a prompt is still processing; one boolean cannot
    tell WORKING from WAITING ON YOU, which is the issue's whole argument
    for a second field.
    """
    sm = _sm()
    server = _server(permission_id="req-1", running=True)
    sm._sessions = {"blocked": _session("blocked", server)}

    row = _row(sm, "blocked")
    assert row.is_processing is True
    assert row.awaiting == AWAITING_PERMISSION


def test_a_persisted_only_row_never_claims_to_be_waiting():
    """In-memory branch only: a cold session has no ``server`` to ask."""
    row = RuntimeSessionInfo(
        session_id="cold", name="cold", description=None,
        created_at="", last_activity="", model_provider="", model_name="",
        is_processing=False, is_loaded=False, client_count=0, turn_count=0,
    )
    assert row.awaiting is None
    assert row.awaiting_since is None


def test_the_listing_row_carries_awaiting():
    """The wire dict, not only the dataclass (#1133)."""
    rows = [
        SimpleNamespace(
            session_id="blocked", name="blocked", description="",
            model_provider="", model_name="", is_loaded=True, client_count=0,
            turn_count=0, workspace_path="/ws", created_by=None,
            orphaned=False, runner=None, inbox_pending=0,
            awaiting=AWAITING_PERMISSION,
            awaiting_since="1970-01-01T00:16:40+00:00",
        ),
    ]
    router, sink = _router(rows)
    router._handle_session_list("c1", None)

    row = _wire_rows(sink)[0]
    assert row["awaiting"] == AWAITING_PERMISSION, (
        "the listing resolved the fact and no client was told"
    )
    assert row["awaiting_since"] == "1970-01-01T00:16:40+00:00"


# ------------------------------------------------ the boundary is enforced

def test_an_unknown_kind_never_reaches_the_wire():
    """The vocabulary is closed, and something has to close it."""
    assert awaiting_fields(PendingPrompt("reference", 1.0)) == (None, None)
    assert resolve_awaiting([PendingPrompt("reference", 1.0)]) is None


def test_a_server_that_answers_nothing_degrades_to_silence():
    """An out-of-tree or duck-typed session server (the #881 precedent)."""
    assert awaiting_of(object()) == (None, None)
    assert awaiting_of(SimpleNamespace()) == (None, None)
    # A bare MagicMock answers every attribute with another mock, which is
    # what a listing would choke on if the boundary did not judge the shape.
    assert awaiting_of(MagicMock()) == (None, None)


def test_a_nonsense_stamp_loses_the_clock_not_the_kind():
    """Absent is 'not measured'; it must never take the kind down with it."""
    assert awaiting_fields(PendingPrompt(AWAITING_PERMISSION, "soon")) == (
        AWAITING_PERMISSION, None)
    assert awaiting_fields(PendingPrompt(AWAITING_PERMISSION, True)) == (
        AWAITING_PERMISSION, None)
    assert awaiting_fields(PendingPrompt(AWAITING_PERMISSION, 1e30)) == (
        AWAITING_PERMISSION, None)


def test_the_vocabulary_is_the_one_the_issue_decided():
    assert AWAITING_KINDS == ("permission", "clarification")


# ------------------------------------------------ concurrency at the read

_RELAY_MODULES = (
    "jaato-server/jaato_server/server/runner_rpc_handlers/prompt_operator.py",
    "jaato-server/jaato_server/server/runner_rpc_handlers/clarification_relay.py",
)


def _iterated_names(function: ast.AST):
    """Every expression ``function`` iterates over, as source snippets."""
    iterated = []
    for node in ast.walk(function):
        if isinstance(node, (ast.For, ast.AsyncFor)):
            iterated.append(node.iter)
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp,
                               ast.GeneratorExp)):
            iterated.extend(gen.iter for gen in node.generators)
    return iterated


@pytest.mark.parametrize("module", _RELAY_MODULES)
def test_the_relays_read_a_snapshot_not_the_live_dict(module):
    """``pending_since`` is read from a DIFFERENT thread than it is written.

    ``list_sessions`` runs on a transport thread; ``handle`` registers and
    drops entries on the daemon's asyncio loop.  Iterating the live dict is
    #938's shape -- ``RuntimeError: dictionary changed size during
    iteration`` raised inside a listing, about a prompt that had just been
    answered.  ``list(self._pending)`` is one C-level copy that does not
    release the GIL, which is the same invariant ``PluginRegistry`` states:
    every read path iterates a snapshot, never the live container.

    Source-level rather than a race: a threaded version would be a timing
    bet, and the property being pinned is textual -- the next read path
    added here must take a snapshot too.
    """
    root = pathlib.Path(__file__).resolve().parents[4]
    tree = ast.parse((root / module).read_text(encoding="utf-8"))
    functions = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "pending_since"
    ]
    assert functions, f"{module} has no pending_since to check"

    live = {"_pending", "_raised_at"}
    for function in functions:
        for iterated in _iterated_names(function):
            # Only a DIRECT ``self._x`` as the iterable is the defect.  An
            # expression that merely mentions one (the generator feeding the
            # comprehension) is judged by its own iterable, which this walk
            # reaches separately.
            if not isinstance(iterated, ast.Attribute):
                continue
            if not (isinstance(iterated.value, ast.Name)
                    and iterated.value.id == "self"):
                continue
            assert iterated.attr not in live, (
                f"{module}: pending_since iterates the live container "
                f"`{ast.unparse(iterated)}` -- take a snapshot"
            )


# --------------------------------------------------------- the declarations

@pytest.mark.parametrize("reversion", REVERSIONS, ids=lambda r: r.test)
def test_each_reversion_anchor_is_unique(reversion):
    """A ``find`` matching twice, or not at all, reports BLOCKED.

    Checked here rather than only in the meta-guard so the failure names the
    anchor at the point an author can fix it.
    """
    root = pathlib.Path(__file__).resolve().parents[4]
    source = (root / reversion.target).read_text()
    assert source.count(reversion.find) == 1, (
        f"{reversion.target} contains {source.count(reversion.find)} copies "
        f"of the anchor for {reversion.test}"
    )
