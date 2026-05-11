"""Regression-pin tests for Phase 4 §4.4 — description-callback bridge
(Finding 2 closure).

**Root cause** (Phase 3 closure recap Finding 2 +
`phase4_implementation_audits.md` Audit 1):

Pre-§4.4 state (post-Path-D, post-§7c):

- ``shared/plugins/session/__init__.py`` declared
  ``PLUGIN_TIER = "daemon"``.
- Path D's runner discover used ``tier_filter="runner"`` → daemon-tier
  session plugin excluded from the runner registry.
- ``session_describe`` tool was unreachable from the runner-side
  model loop.
- Daemon-side ``_setup_session_plugin`` wired
  ``set_description_callback`` on a daemon-side instance whose
  ``set_description`` was never invoked.
- Net effect: ``SessionDescriptionUpdatedEvent`` never fired post-§7c.

**Phase 4 §4.4 fix** (4 sub-actions in one commit, per audit doc):

A. Tier-flip ``session`` plugin → ``runner``.
B. Add ``description_updated`` NotificationFrame event_type + install
   runner-side shim that emits it; mirror restore.
C. Daemon-side demuxer branch routes ``description_updated`` →
   ``SessionDescriptionUpdatedEvent.emit``.
D. Delete the daemon-side ``set_description_callback`` wiring (dead
   code post-flip; the daemon-side instance's ``set_description`` is
   still never invoked).

Tests pin:

1. ``PLUGIN_TIER`` is ``"runner"`` in the session plugin __init__.
2. ``_NOTIF_DESCRIPTION_UPDATED`` constant exists on
   ``RunnerRPC``.
3. Install function wires the description shim on the runner-side
   session plugin via the registry.
4. Shim emits ``description_updated`` NotificationFrame.
5. Daemon-side demuxer routes the frame to
   ``SessionDescriptionUpdatedEvent``.
6. Restore reverts the description callback to its pre-install value.
7. AST pin: ``_setup_session_plugin`` no longer wires
   ``set_description_callback`` (catches regression of the dead-code
   removal).
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from server.runner.rpc import RunnerRPC


# ----------------------------------------------------------------------
# A — tier-flip
# ----------------------------------------------------------------------


def test_session_plugin_is_runner_tier() -> None:
    """Pin §4.4-A: ``shared/plugins/session/__init__.py`` declares
    ``PLUGIN_TIER = "runner"`` so Path D's runner discover loads it.

    Pre-§4.4: ``"daemon"``, which made ``session_describe``
    unreachable from the model loop.  Catches accidental revert.
    """
    from shared.plugins import session as session_pkg

    assert hasattr(session_pkg, "PLUGIN_TIER")
    assert session_pkg.PLUGIN_TIER == "runner", (
        f"Phase 4 §4.4 regression: session plugin PLUGIN_TIER is "
        f"{session_pkg.PLUGIN_TIER!r}; should be 'runner' so Path D's "
        f"runner discover loads it.  Reverting to 'daemon' breaks "
        f"session_describe (unreachable from runner-side model loop)."
    )


# ----------------------------------------------------------------------
# B — NotificationFrame event_type + shim install
# ----------------------------------------------------------------------


def test_notif_description_updated_constant_exists() -> None:
    """Pin §4.4-B: ``_NOTIF_DESCRIPTION_UPDATED`` constant exists
    on ``RunnerRPC`` for shim → demuxer wire alignment."""
    assert hasattr(RunnerRPC, "_NOTIF_DESCRIPTION_UPDATED")
    assert RunnerRPC._NOTIF_DESCRIPTION_UPDATED == "description_updated"


def test_install_wires_description_callback_on_runner_session_plugin() -> None:
    """Pin §4.4-B: ``_install_session_notification_callbacks`` looks
    up the session plugin via ``session._runtime.registry.get_plugin``
    and calls ``set_description_callback`` on it.

    Without this wiring, the model-invoked ``session_describe``
    fires its callback into the void.
    """
    rpc = RunnerRPC.__new__(RunnerRPC)
    rpc.emit_notification = lambda **kw: None

    captured_callbacks: List[Any] = []

    class _FakeSessionPlugin:
        _on_description_changed = None

        def set_description_callback(self, cb):
            captured_callbacks.append(cb)
            self._on_description_changed = cb

    class _FakeRegistry:
        def get_plugin(self, name):
            if name == "session":
                return _FakeSessionPlugin()
            return None

    class _FakeRuntime:
        registry = _FakeRegistry()

    class _FakeSession:
        _runtime = _FakeRuntime()

    sess = _FakeSession()
    originals = rpc._install_session_notification_callbacks(
        sess, request_id=1,
    )

    assert len(captured_callbacks) == 1, (
        "Phase 4 §4.4 regression: install function didn't call "
        "set_description_callback on the runner-side session plugin."
    )
    assert "description_callback" in originals, (
        "Phase 4 §4.4 regression: install must record the pre-shim "
        "callback under key 'description_callback' so restore can "
        "put it back."
    )


def test_shim_emits_description_updated_notification() -> None:
    """Pin §4.4-B: the description-callback shim emits a
    ``description_updated`` NotificationFrame with the right
    payload shape."""
    rpc = RunnerRPC.__new__(RunnerRPC)
    captured: List[Dict[str, Any]] = []
    rpc.emit_notification = lambda **kw: captured.append(kw)

    class _FakeSessionPlugin:
        _on_description_changed = None
        _captured_callback = None

        def set_description_callback(self, cb):
            self._on_description_changed = cb
            self._captured_callback = cb

    plugin = _FakeSessionPlugin()

    class _FakeRegistry:
        def get_plugin(self, name):
            return plugin if name == "session" else None

    class _FakeRuntime:
        registry = _FakeRegistry()

    class _FakeSession:
        _runtime = _FakeRuntime()

    rpc._install_session_notification_callbacks(_FakeSession(), request_id=42)

    # Fire the shim like the session plugin would.
    plugin._captured_callback("sess-abc", "My description")

    assert len(captured) == 1
    assert captured[0]["request_id"] == 42
    assert captured[0]["event_type"] == "description_updated"
    assert captured[0]["payload"] == {
        "session_id": "sess-abc",
        "description": "My description",
    }


# ----------------------------------------------------------------------
# C — daemon-side demuxer branch
# ----------------------------------------------------------------------


def test_demuxer_routes_description_updated_to_event() -> None:
    """Pin §4.4-C: daemon-side
    ``_build_send_message_notification_handler`` handles the
    ``description_updated`` event_type and emits a
    ``SessionDescriptionUpdatedEvent``."""
    from server.core import JaatoServer
    from jaato_sdk.events import SessionDescriptionUpdatedEvent

    srv = JaatoServer.__new__(JaatoServer)
    srv._emitted_events = []
    srv.emit = lambda e: srv._emitted_events.append(e)
    srv._main_agent_id = "main"
    srv._cached_context_limit = 0
    srv._traces = []
    srv._trace = lambda m: srv._traces.append(m)

    handler = srv._build_send_message_notification_handler()
    handler("description_updated", {
        "session_id": "sess-1",
        "description": "Test description",
    })

    desc_events = [
        e for e in srv._emitted_events
        if isinstance(e, SessionDescriptionUpdatedEvent)
    ]
    assert len(desc_events) == 1, (
        "Phase 4 §4.4 regression: daemon demuxer didn't route "
        "description_updated → SessionDescriptionUpdatedEvent; the "
        "TUI's session-picker won't refresh."
    )
    assert desc_events[0].session_id == "sess-1"
    assert desc_events[0].description == "Test description"


def test_demuxer_description_updated_handles_missing_fields() -> None:
    """Pin §4.4-C: when payload is malformed (missing session_id
    or description), the demuxer emits an event with empty-string
    defaults — no crash."""
    from server.core import JaatoServer
    from jaato_sdk.events import SessionDescriptionUpdatedEvent

    srv = JaatoServer.__new__(JaatoServer)
    srv._emitted_events = []
    srv.emit = lambda e: srv._emitted_events.append(e)
    srv._main_agent_id = "main"
    srv._cached_context_limit = 0
    srv._traces = []
    srv._trace = lambda m: srv._traces.append(m)

    handler = srv._build_send_message_notification_handler()
    handler("description_updated", {})

    desc_events = [
        e for e in srv._emitted_events
        if isinstance(e, SessionDescriptionUpdatedEvent)
    ]
    assert len(desc_events) == 1
    assert desc_events[0].session_id == ""
    assert desc_events[0].description == ""


# ----------------------------------------------------------------------
# B (restore counterpart)
# ----------------------------------------------------------------------


def test_restore_reverts_description_callback() -> None:
    """Pin §4.4-B: ``_restore_session_notification_callbacks``
    reverts the description callback to its pre-install value
    (typically None for fresh runner-side instances)."""
    rpc = RunnerRPC.__new__(RunnerRPC)
    rpc.emit_notification = lambda **kw: None

    class _FakeSessionPlugin:
        _on_description_changed = None

        def set_description_callback(self, cb):
            self._on_description_changed = cb

    plugin = _FakeSessionPlugin()

    class _FakeRegistry:
        def get_plugin(self, name):
            return plugin if name == "session" else None

    class _FakeRuntime:
        registry = _FakeRegistry()

    class _FakeSession:
        _runtime = _FakeRuntime()

    sess = _FakeSession()
    originals = rpc._install_session_notification_callbacks(sess, request_id=1)
    # Shim installed.
    assert plugin._on_description_changed is not None
    # Restore.
    rpc._restore_session_notification_callbacks(sess, originals)
    assert plugin._on_description_changed is None, (
        "Phase 4 §4.4 regression: restore didn't revert the "
        "description callback; leaked shim references would fire "
        "against a stale request_id on the next call."
    )


# ----------------------------------------------------------------------
# D — dead-code removal (daemon-side wiring deleted)
# ----------------------------------------------------------------------


def test_setup_session_plugin_does_not_wire_description_callback() -> None:
    """Pin §4.4-D: ``_setup_session_plugin`` source no longer
    references ``set_description_callback``.  Catches regression
    where someone re-adds the daemon-side wiring (which would
    silently double-fire SessionDescriptionUpdatedEvent if both
    paths somehow became live, or — more likely — restore the
    pre-§4.4 dead-letter state)."""
    from server.core import JaatoServer

    src = inspect.getsource(JaatoServer._setup_session_plugin)
    src = textwrap.dedent(src)

    # The string "set_description_callback" must NOT appear as a
    # method call.  It MAY appear in comments referencing the
    # historical context (the deletion note).
    tree = ast.parse(src)
    found_call = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "set_description_callback"
        ):
            found_call = True
            break

    assert not found_call, (
        "Phase 4 §4.4-D regression: _setup_session_plugin re-wires "
        "set_description_callback on the daemon-side instance.  "
        "Post-§4.4 this is dead code — the daemon-side instance's "
        "set_description is never invoked.  Keep the wiring deleted."
    )
