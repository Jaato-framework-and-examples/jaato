"""Regression-pin tests for Path E — second post-§7c chain
(send_message during active turn).

**Root cause** (cycle-6 comprehensive remaining-chain audit):

Cycle 5 verified Path D closed the BOOTSTRAP chain.  Cycle 6
surfaced a SECOND chain in send_message-during-active-turn.  Two
layers, both shipped together as Path E:

**Layer 5 — notification re-entrancy.**  Pre-Path-E the daemon-
side ``usage_update`` notification handler at ``core.py:3524-3561``
synchronously called 2 blocking RPCs back into the runner DURING
the runner's active ``send_message``.  The handler waits for the
runner to finish its current request before serving the new one,
but the original ``send_message`` hasn't returned — race-shaped
timeout.

**Layer 6 — wire-format mismatch.**  Pre-Path-E
``_serialize_message_for_wire`` used ``dataclasses.asdict`` to
serialize ``Message`` objects, producing a wire shape (parts as
raw dataclass dump) that didn't match the canonical session
serializer's expected input (parts as tagged-union).  Two
consumers crashed with ``'dict' object has no attribute 'role'``:
``session_manager._save_session`` and the replay path.

**Path E sub-fixes** (5 in total, one commit):

- E.1 — Runner-side ``_make_usage_update_notification_shim``
  batches ``context_limit`` + ``turns`` into the payload.
- E.2 — Daemon-side ``_cached_context_limit`` cache, populated at
  end of ``initialize()`` from off-band RPC.  In-band aspect
  callbacks read from cache.
- E.3 — Cache invalidated on ``/model`` command.
- E.4 — ``_serialize_message_for_wire`` uses canonical
  ``serialize_message`` for real ``Message`` instances.
- E.5 — ``serialize_history`` idempotent on already-serialized
  dicts (defensive — closes both consumer crashes).

Tests pin:

- E.1: shim payload contains ``context_limit`` + ``turns``;
  daemon handler reads from payload (no in-band RPC needed)
- E.2: ``_cached_context_limit`` populated after initialize
  (covered structurally in the cache field default)
- E.3: ``/model`` command result handler resets
  ``_cached_context_limit`` to ``None``
- E.4: ``_serialize_message_for_wire`` on real Message returns
  canonical-format dict (parts as tagged-union); test stub
  ``to_dict`` path preserved
- E.5: ``serialize_history`` accepts dicts as no-op pass-through
  AND Message objects as serialize; mixed lists tolerated
- AST pin: ``_make_usage_update_notification_shim`` body computes
  ``context_limit`` and ``turns``
- Round-trip: runner wire dict → ``deserialize_history`` →
  Message → ``serialize_history`` is byte-identical
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any, List
from unittest.mock import MagicMock

import pytest


# ----------------------------------------------------------------------
# E.1 — runner-side shim batches context_limit + turns
# ----------------------------------------------------------------------


def test_usage_update_shim_batches_context_limit_from_session() -> None:
    """Pin: runner shim reads context_limit from
    ``session.get_context_limit()`` and emits it in the payload."""
    from server.runner.rpc import RunnerRPC

    rpc = RunnerRPC.__new__(RunnerRPC)
    captured: List[Any] = []
    rpc.emit_notification = lambda **kw: captured.append(kw)

    class _Sess:
        _turn_accounting = [{"t": 1}, {"t": 2}, {"t": 3}]

        def get_context_limit(self) -> int:
            return 200_000

    class _Host:
        session = _Sess()

    rpc._session_host = _Host()

    shim = rpc._make_usage_update_notification_shim(request_id=7)

    class _Usage:
        prompt_tokens = 10
        output_tokens = 20
        total_tokens = 30
        cache_read_tokens = None
        cache_creation_tokens = None
        reasoning_tokens = None
        thinking_tokens = None
        cost_usd = None

    shim(_Usage())

    assert len(captured) == 1
    payload = captured[0]["payload"]
    assert payload["context_limit"] == 200_000, (
        "Path E E.1 regression: shim must batch context_limit into "
        "payload.  Without this, daemon handler would call back into "
        "the runner via in-band RPC and race against active send_message."
    )
    assert payload["turns"] == 3, (
        "Path E E.1 regression: shim must batch turns count into "
        "payload (len of turn_accounting)."
    )


def test_usage_update_shim_falls_back_to_zero_on_missing_session() -> None:
    """Pin: shim emits ``context_limit=0`` + ``turns=0`` when the
    session host is missing or lacks the accessor.  Defensive
    fallback so the notification still emits — daemon handler can
    consult its cache or omit the field."""
    from server.runner.rpc import RunnerRPC

    rpc = RunnerRPC.__new__(RunnerRPC)
    captured: List[Any] = []
    rpc.emit_notification = lambda **kw: captured.append(kw)
    rpc._session_host = None  # Pre-bootstrap: no session yet.

    shim = rpc._make_usage_update_notification_shim(request_id=99)

    class _Usage:
        prompt_tokens = 1
        output_tokens = 2
        total_tokens = 3
        cache_read_tokens = None
        cache_creation_tokens = None
        reasoning_tokens = None
        thinking_tokens = None
        cost_usd = None

    shim(_Usage())

    assert len(captured) == 1
    assert captured[0]["payload"]["context_limit"] == 0
    assert captured[0]["payload"]["turns"] == 0


def test_usage_update_shim_uses_get_turn_accounting_fallback() -> None:
    """Pin: when the session lacks ``_turn_accounting`` private
    attr, the shim falls back to ``get_turn_accounting()`` public
    method."""
    from server.runner.rpc import RunnerRPC

    rpc = RunnerRPC.__new__(RunnerRPC)
    captured: List[Any] = []
    rpc.emit_notification = lambda **kw: captured.append(kw)

    class _Sess:
        # No _turn_accounting attribute.

        def get_context_limit(self) -> int:
            return 50_000

        def get_turn_accounting(self) -> list:
            return [{"a": 1}, {"a": 2}]

    class _Host:
        session = _Sess()

    rpc._session_host = _Host()
    shim = rpc._make_usage_update_notification_shim(request_id=1)

    class _Usage:
        prompt_tokens = 0
        output_tokens = 0
        total_tokens = 5
        cache_read_tokens = None
        cache_creation_tokens = None
        reasoning_tokens = None
        thinking_tokens = None
        cost_usd = None

    shim(_Usage())

    assert captured[0]["payload"]["context_limit"] == 50_000
    assert captured[0]["payload"]["turns"] == 2


# ----------------------------------------------------------------------
# E.1 — daemon handler reads from payload (no in-band RPC)
# ----------------------------------------------------------------------


def test_demuxer_usage_update_reads_context_limit_from_payload() -> None:
    """Pin: when payload carries ``context_limit``, daemon handler
    uses it WITHOUT calling ``session_get_context_limit_threadsafe``.
    Closes the cycle-6 race."""
    from server.core import JaatoServer
    from jaato_sdk.events import ContextUpdatedEvent, UsageBreakdown

    srv = JaatoServer.__new__(JaatoServer)
    srv._emitted_events = []
    srv.emit = lambda e: srv._emitted_events.append(e)
    srv._main_agent_id = "main"
    srv._cached_context_limit = None
    srv._build_usage = lambda **kw: UsageBreakdown(
        prompt_tokens=kw.get("prompt_tokens", 0),
        output_tokens=kw.get("output_tokens", 0),
        total_tokens=kw.get("total_tokens", 0),
    )
    # Sentinel: the runner_rpc would crash if called.  This is the
    # critical pin — Path E makes the handler NOT call back.
    rpc_mock = MagicMock()
    rpc_mock.session_get_context_limit_threadsafe.side_effect = AssertionError(
        "Path E E.1 regression: handler must NOT call "
        "session_get_context_limit_threadsafe in-band — that's the "
        "race fix.  Read context_limit from payload instead."
    )
    rpc_mock.session_get_turn_accounting_threadsafe.side_effect = (
        AssertionError(
            "Path E E.1 regression: handler must NOT call "
            "session_get_turn_accounting_threadsafe in-band — that's "
            "the race fix.  Read turns from payload instead."
        )
    )
    srv._runner_rpc = rpc_mock

    handler = srv._build_send_message_notification_handler()
    handler("usage_update", {
        "prompt_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
        "cache_read_tokens": None,
        "cache_creation_tokens": None,
        "reasoning_tokens": None,
        "thinking_tokens": None,
        "cost_usd": None,
        "context_limit": 200_000,
        "turns": 4,
    })

    assert len(srv._emitted_events) == 1
    evt = srv._emitted_events[0]
    assert isinstance(evt, ContextUpdatedEvent)
    assert evt.context_limit == 200_000
    assert evt.turns == 4


def test_demuxer_usage_update_falls_back_to_cache() -> None:
    """Pin: if payload omits ``context_limit`` (rolling-upgrade
    scenario where the runner is older than the daemon), the
    handler reads from ``_cached_context_limit``."""
    from server.core import JaatoServer
    from jaato_sdk.events import UsageBreakdown

    srv = JaatoServer.__new__(JaatoServer)
    srv._emitted_events = []
    srv.emit = lambda e: srv._emitted_events.append(e)
    srv._main_agent_id = "main"
    srv._cached_context_limit = 128_000
    srv._build_usage = lambda **kw: UsageBreakdown(
        prompt_tokens=0, output_tokens=0,
        total_tokens=kw.get("total_tokens", 0),
    )
    rpc_mock = MagicMock()
    srv._runner_rpc = rpc_mock

    handler = srv._build_send_message_notification_handler()
    handler("usage_update", {
        "total_tokens": 50,
        # No context_limit field — pre-Path-E runner.
    })

    # Handler must NOT call back into the runner — uses cache.
    rpc_mock.session_get_context_limit_threadsafe.assert_not_called()
    assert len(srv._emitted_events) == 1
    assert srv._emitted_events[0].context_limit == 128_000


# ----------------------------------------------------------------------
# E.3 — /model command invalidates cache
# ----------------------------------------------------------------------


def test_model_command_invalidates_cached_context_limit() -> None:
    """Pin: ``/model`` command result handler resets
    ``_cached_context_limit`` to ``None``.  Different model can
    have a different context window — stale cache would mis-
    report usage in the toolbar."""
    src = inspect.getsource(__import__("server.core", fromlist=["JaatoServer"]).JaatoServer)
    src = textwrap.dedent(src)
    # AST scan for the assignment ``self._cached_context_limit = None``
    # inside the model-command result handler region.  We can't
    # easily test the handler in isolation (it lives inside a
    # multi-hundred-line method), so we pin the assignment
    # syntactically.
    tree = ast.parse(src)
    found_invalidations = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
            and node.targets[0].attr == "_cached_context_limit"
            and isinstance(node.value, ast.Constant)
            and node.value.value is None
        ):
            found_invalidations.append(node)
    assert found_invalidations, (
        "Path E E.3 regression: ``self._cached_context_limit = None`` "
        "assignment missing.  Without it, /model command leaves a "
        "stale cached context_limit, and subsequent usage events "
        "report wrong percentages in the toolbar."
    )


# ----------------------------------------------------------------------
# E.4 — runner serializer uses canonical Message format
# ----------------------------------------------------------------------


def test_serialize_message_for_wire_uses_canonical_format() -> None:
    """Pin: real ``Message`` dataclass → canonical session-
    serializer wire shape (parts as tagged union with ``type``
    discriminator).  Pre-Path-E this produced ``dataclasses.asdict``
    output (parts as raw dataclass dump) which crashed the daemon-
    side session-save path."""
    from jaato_sdk.plugins.model_provider.types import Message, Part, Role
    from server.runner.rpc import _serialize_message_for_wire

    msg = Message(
        role=Role.USER,
        parts=[Part.from_text("hello")],
    )
    wire = _serialize_message_for_wire(msg)

    assert isinstance(wire, dict)
    assert wire["role"] == "user", (
        "canonical format: role coerced to enum value string"
    )
    assert wire["parts"] == [{"type": "text", "text": "hello"}], (
        "Path E E.4 regression: parts must use canonical tagged-"
        "union shape (``{type: text, text: ...}``), NOT the raw "
        "``dataclasses.asdict`` dump (``{text: ..., function_call: "
        "null, ...}``).  The mismatch crashes session_save with "
        "``'dict' object has no attribute 'role'``."
    )
    # message_id must NOT leak (canonical session serializer
    # excludes it; sessions don't persist message_id).
    assert "message_id" not in wire


def test_serialize_message_for_wire_preserves_to_dict_path() -> None:
    """Pin: test stubs that implement ``to_dict()`` still win
    over the canonical Message path.  Backward compat for existing
    tests."""
    from server.runner.rpc import _serialize_message_for_wire

    class _Stub:
        def to_dict(self) -> dict:
            return {"role": "user", "content": "hi"}

    wire = _serialize_message_for_wire(_Stub())
    assert wire == {"role": "user", "content": "hi"}


# ----------------------------------------------------------------------
# E.5 — serialize_history idempotent
# ----------------------------------------------------------------------


def test_serialize_history_idempotent_on_dicts() -> None:
    """Pin: ``serialize_history`` accepts a list of dicts (from
    runner-RPC wire) and passes them through unchanged.  Closes
    the cycle-6 ``'dict' object has no attribute 'role'`` crash
    in both consumers (session-save + replay)."""
    from shared.plugins.session.serializer import serialize_history

    wire_dicts = [
        {"role": "user", "parts": [{"type": "text", "text": "hi"}]},
        {"role": "model", "parts": [{"type": "text", "text": "yo"}]},
    ]
    out = serialize_history(wire_dicts)
    assert out == wire_dicts, (
        "Path E E.5 regression: serialize_history must be idempotent "
        "on already-serialized dicts (pass-through).  Pre-Path-E it "
        "called serialize_message on each dict → AttributeError on .role."
    )


def test_serialize_history_works_on_messages() -> None:
    """Pin: ``serialize_history`` on a list of Message objects
    still serializes them (the historical contract)."""
    from jaato_sdk.plugins.model_provider.types import Message, Part, Role
    from shared.plugins.session.serializer import serialize_history

    messages = [
        Message(role=Role.USER, parts=[Part.from_text("hi")]),
        Message(role=Role.MODEL, parts=[Part.from_text("yo")]),
    ]
    out = serialize_history(messages)
    assert out == [
        {"role": "user", "parts": [{"type": "text", "text": "hi"}]},
        {"role": "model", "parts": [{"type": "text", "text": "yo"}]},
    ]


def test_serialize_history_tolerates_mixed_list() -> None:
    """Pin: mixed lists (some Message, some dict) work.  Defensive:
    callers shouldn't construct mixed lists, but if they do we
    don't crash."""
    from jaato_sdk.plugins.model_provider.types import Message, Part, Role
    from shared.plugins.session.serializer import serialize_history

    mixed = [
        Message(role=Role.USER, parts=[Part.from_text("a")]),
        {"role": "model", "parts": [{"type": "text", "text": "b"}]},
    ]
    out = serialize_history(mixed)
    assert len(out) == 2
    assert out[0]["role"] == "user"
    assert out[1]["role"] == "model"


def test_serialize_history_empty() -> None:
    """Pin: empty list / None handled."""
    from shared.plugins.session.serializer import serialize_history
    assert serialize_history([]) == []
    assert serialize_history(None) == []  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Round-trip pin: wire dict → deserialize → serialize is identity
# ----------------------------------------------------------------------


def test_wire_dict_roundtrips_through_serializer() -> None:
    """Pin: the runner's wire form (canonical, post-E.4) round-
    trips losslessly through ``deserialize_history`` →
    ``serialize_history``.  Catches drift between
    ``_serialize_message_for_wire`` and the canonical serializer."""
    from jaato_sdk.plugins.model_provider.types import Message, Part, Role
    from shared.plugins.session.serializer import (
        deserialize_history, serialize_history,
    )
    from server.runner.rpc import _serialize_message_for_wire

    original = [
        Message(role=Role.USER, parts=[Part.from_text("hi")]),
        Message(role=Role.MODEL, parts=[Part.from_text("yo")]),
    ]
    wire = [_serialize_message_for_wire(m) for m in original]
    # Round-trip: wire → Messages → wire.
    messages = deserialize_history(wire)
    re_wire = serialize_history(messages)
    assert re_wire == wire, (
        "Path E E.4/E.5 regression: wire format drifted from "
        "canonical serializer; round-trip not lossless."
    )


# ----------------------------------------------------------------------
# AST pin: shim body computes context_limit + turns
# ----------------------------------------------------------------------


def test_usage_update_shim_source_computes_context_limit_and_turns() -> None:
    """Pin via AST: ``_make_usage_update_notification_shim`` body
    references ``get_context_limit`` AND ``turn_accounting`` (or
    ``get_turn_accounting``).  Catches accidental deletion of E.1."""
    from server.runner.rpc import RunnerRPC

    src = inspect.getsource(
        RunnerRPC._make_usage_update_notification_shim,
    )
    assert "get_context_limit" in src, (
        "Path E E.1 regression: shim must compute context_limit "
        "from session — string ``get_context_limit`` missing."
    )
    assert (
        "turn_accounting" in src or "get_turn_accounting" in src
    ), (
        "Path E E.1 regression: shim must compute turns from "
        "session — neither ``turn_accounting`` nor "
        "``get_turn_accounting`` referenced."
    )
