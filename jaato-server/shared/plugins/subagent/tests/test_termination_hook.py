"""Tests for the §3.11 + peer-review M4 subagent-termination hook.

When a subagent finishes — normal ``signal_completion``, error
termination, or operator ``cancel_subagent`` — its session-id-keyed
state in shared plugin instances must be dropped.  Without this,
a long-lived parent session accumulates unbounded entries from
completed subagents in plugins like ``reliability``
(``(session_id, tool_name)`` success counters), ``memory``
(per-session caches), and the runner-side ``permission`` plugin
(per-session policy state once §3.7's runner-side migration lands).

The hook works as follows:

- Plugins opt in by implementing
  ``on_subagent_terminated(agent_id, session_id)``.
- :meth:`SubagentPlugin.set_runtime` walks the runtime's plugin
  registry and auto-registers any plugin with that method as a
  termination callback.
- :meth:`SubagentPlugin._close_session_unlocked` invokes registered
  callbacks AFTER pulling the agent_id out of ``_active_sessions``
  but with the JaatoSession's id passed through so callbacks can
  drop their session-id-keyed entries.

These tests pin the contract:

- Auto-discovery: plugins with ``on_subagent_terminated`` are picked
  up by :meth:`set_runtime`.
- Plugins without the method are NOT registered.
- Manual registration via
  :meth:`register_termination_callback` works.
- Idempotent registration: registering the same callback twice
  doesn't fire it twice.
- Callback invocation: closing a subagent fires every registered
  callback with ``(agent_id, session_id)``.
- Callback ordering: callbacks fire AFTER the agent_id is removed
  from ``_active_sessions``.
- Best-effort: a callback raising doesn't block other callbacks.
- ``session_id`` is ``None`` when the JaatoSession lacks one (early
  failure) — callbacks see ``None`` rather than a crash.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import pytest

from shared.plugins.subagent.plugin import SubagentPlugin


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


class _FakeSession:
    """Stand-in for a JaatoSession; only the ``session_id`` lookup
    matters for the M4 contract."""

    def __init__(self, session_id: Optional[str]) -> None:
        self.session_id = session_id


class _FakeRegistry:
    """Stand-in for PluginRegistry exposing list_exposed +
    get_plugin — the surface SubagentPlugin.set_runtime walks."""

    def __init__(self, plugins: dict) -> None:
        self._plugins = dict(plugins)

    def list_exposed(self) -> List[str]:
        return list(self._plugins)

    def get_plugin(self, name: str) -> Any:
        return self._plugins.get(name)


class _FakeRuntime:
    """Stand-in for JaatoRuntime exposing ``registry``."""

    def __init__(self, registry: _FakeRegistry) -> None:
        self.registry = registry


class _RecordingPlugin:
    """Plugin that records each on_subagent_terminated call.  The
    duck-typed handler signature mirrors what real plugins like
    reliability implement."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Optional[str]]] = []

    def on_subagent_terminated(
        self,
        agent_id: str,
        session_id: Optional[str],
    ) -> None:
        self.calls.append((agent_id, session_id))


class _PluginWithoutHook:
    """Plugin that does NOT implement on_subagent_terminated; must
    be skipped during auto-discovery."""

    pass


def _install_subagent(agent_id: str, session_id: Optional[str]) -> SubagentPlugin:
    """Build a SubagentPlugin with one fake active session installed
    so ``_close_session_unlocked`` has something to close."""
    plugin = SubagentPlugin()
    session = _FakeSession(session_id)
    plugin._active_sessions[agent_id] = {
        'session': session,
        'profile': None,
        'display_name': agent_id,
        'agent_id': agent_id,
        'owner_id': 0,
        'created_at': None,
        'last_activity': None,
        'turn_count': 5,
        'max_turns': 10,
    }
    return plugin


# ----------------------------------------------------------------------
# Auto-discovery via set_runtime
# ----------------------------------------------------------------------


def test_set_runtime_auto_registers_plugins_with_hook() -> None:
    recorder = _RecordingPlugin()
    runtime = _FakeRuntime(_FakeRegistry({"reliability": recorder}))

    plugin = SubagentPlugin()
    plugin.set_runtime(runtime)

    assert recorder.on_subagent_terminated in plugin._termination_callbacks


def test_set_runtime_skips_plugins_without_hook() -> None:
    recorder = _RecordingPlugin()
    no_hook = _PluginWithoutHook()
    runtime = _FakeRuntime(_FakeRegistry({
        "reliability": recorder,
        "noisy": no_hook,
    }))

    plugin = SubagentPlugin()
    plugin.set_runtime(runtime)

    # Only reliability's hook is registered.
    assert plugin._termination_callbacks == [recorder.on_subagent_terminated]


def test_set_runtime_idempotent_for_same_handler() -> None:
    """Calling set_runtime twice with the same plugin doesn't
    register the same handler twice."""
    recorder = _RecordingPlugin()
    runtime = _FakeRuntime(_FakeRegistry({"reliability": recorder}))

    plugin = SubagentPlugin()
    plugin.set_runtime(runtime)
    plugin.set_runtime(runtime)

    # Only one callback entry — duplicate registration is filtered.
    assert plugin._termination_callbacks.count(
        recorder.on_subagent_terminated
    ) == 1


def test_set_runtime_no_registry_is_safe() -> None:
    """Runtime without a ``registry`` attribute (very early bootstrap)
    must not crash."""

    class _BareRuntime:
        pass

    plugin = SubagentPlugin()
    plugin.set_runtime(_BareRuntime())  # type: ignore[arg-type]
    assert plugin._termination_callbacks == []


# ----------------------------------------------------------------------
# Manual registration
# ----------------------------------------------------------------------


def test_register_termination_callback_appends() -> None:
    plugin = SubagentPlugin()
    calls: List[Tuple[str, Optional[str]]] = []

    def _cb(agent_id: str, session_id: Optional[str]) -> None:
        calls.append((agent_id, session_id))

    plugin.register_termination_callback(_cb)
    assert plugin._termination_callbacks == [_cb]


def test_register_termination_callback_idempotent() -> None:
    plugin = SubagentPlugin()

    def _cb(agent_id: str, session_id: Optional[str]) -> None:
        pass

    plugin.register_termination_callback(_cb)
    plugin.register_termination_callback(_cb)

    assert plugin._termination_callbacks == [_cb]


# ----------------------------------------------------------------------
# Invocation on session close
# ----------------------------------------------------------------------


def test_close_session_fires_callbacks_with_session_id() -> None:
    plugin = _install_subagent("sa-1", "sess-abc")
    recorder = _RecordingPlugin()
    plugin.register_termination_callback(recorder.on_subagent_terminated)

    plugin._close_session_unlocked("sa-1")

    assert recorder.calls == [("sa-1", "sess-abc")]


def test_close_session_fires_with_none_session_id_when_session_has_none() -> None:
    """Subagent that failed before the session had an id assigned —
    the callback still fires, with session_id=None."""
    plugin = _install_subagent("sa-2", None)
    recorder = _RecordingPlugin()
    plugin.register_termination_callback(recorder.on_subagent_terminated)

    plugin._close_session_unlocked("sa-2")

    assert recorder.calls == [("sa-2", None)]


def test_close_session_fires_after_active_sessions_cleared() -> None:
    """Callback fires AFTER the agent_id is removed from
    ``_active_sessions`` — proves the registry is consistent
    when callbacks observe it."""
    plugin = _install_subagent("sa-3", "sess-3")
    seen_active_keys: List[List[str]] = []

    def _cb(agent_id: str, session_id: Optional[str]) -> None:
        seen_active_keys.append(list(plugin._active_sessions.keys()))

    plugin.register_termination_callback(_cb)
    plugin._close_session_unlocked("sa-3")

    # Callback observed an empty active-sessions registry.
    assert seen_active_keys == [[]]


def test_close_session_fires_all_callbacks_in_registration_order() -> None:
    plugin = _install_subagent("sa-4", "sess-4")
    order: List[str] = []

    def _first(agent_id: str, session_id: Optional[str]) -> None:
        order.append("first")

    def _second(agent_id: str, session_id: Optional[str]) -> None:
        order.append("second")

    plugin.register_termination_callback(_first)
    plugin.register_termination_callback(_second)

    plugin._close_session_unlocked("sa-4")
    assert order == ["first", "second"]


def test_close_session_isolates_callback_failures() -> None:
    """A buggy callback that raises must not block subsequent
    callbacks (M4 cleanup is best-effort)."""
    plugin = _install_subagent("sa-5", "sess-5")
    fired: List[str] = []

    def _bad(agent_id: str, session_id: Optional[str]) -> None:
        fired.append("bad")
        raise RuntimeError("boom")

    def _good(agent_id: str, session_id: Optional[str]) -> None:
        fired.append("good")

    plugin.register_termination_callback(_bad)
    plugin.register_termination_callback(_good)

    # Should not raise.
    plugin._close_session_unlocked("sa-5")
    assert fired == ["bad", "good"]


def test_close_session_noop_for_unknown_agent_id() -> None:
    """Closing an agent that isn't in ``_active_sessions`` is a no-op
    — no callbacks fire."""
    plugin = SubagentPlugin()
    fired: List[str] = []

    def _cb(agent_id: str, session_id: Optional[str]) -> None:
        fired.append(agent_id)

    plugin.register_termination_callback(_cb)
    plugin._close_session_unlocked("never-existed")
    assert fired == []
