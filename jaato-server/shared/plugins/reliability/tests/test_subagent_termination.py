"""Tests for ``ReliabilityPlugin.on_subagent_terminated`` (Phase 3
§3.11 + peer-review M4).

The reliability plugin keys per-session tool-success counters by
``(session_id, tool_name)``.  When a subagent finishes, its
session-id-keyed entries must be dropped or a long-lived parent
session accumulates unbounded counters from completed subagents.

These tests pin the cleanup contract:

- Entries for the terminating session are dropped.
- Entries for OTHER sessions are preserved.
- Empty session_id (early-failure) is a safe no-op.
- ``None`` session_id is a safe no-op.
- The method is duck-type compatible with the SubagentPlugin
  auto-discovery contract (callable as ``on_subagent_terminated(
  agent_id, session_id)``).
"""

from __future__ import annotations

from shared.plugins.reliability.plugin import ReliabilityPlugin


def _populate_session_state(
    plugin: ReliabilityPlugin,
    session_id: str,
    tools: list,
) -> None:
    """Seed _session_tool_successes with one entry per (sid, tool)."""
    for i, tool in enumerate(tools, start=1):
        plugin._session_tool_successes[(session_id, tool)] = i


def test_on_terminated_drops_session_entries() -> None:
    plugin = ReliabilityPlugin()
    _populate_session_state(plugin, "sess-A", ["tool_a", "tool_b", "tool_c"])

    plugin.on_subagent_terminated("subagent-1", "sess-A")

    assert plugin._session_tool_successes == {}


def test_on_terminated_preserves_other_sessions() -> None:
    plugin = ReliabilityPlugin()
    _populate_session_state(plugin, "sess-A", ["tool_a", "tool_b"])
    _populate_session_state(plugin, "sess-B", ["tool_a", "tool_c"])

    plugin.on_subagent_terminated("subagent-1", "sess-A")

    # Only sess-B entries remain.
    remaining = set(plugin._session_tool_successes)
    assert remaining == {("sess-B", "tool_a"), ("sess-B", "tool_c")}


def test_on_terminated_with_empty_session_id_is_noop() -> None:
    plugin = ReliabilityPlugin()
    _populate_session_state(plugin, "sess-A", ["tool_a"])
    _populate_session_state(plugin, "", ["tool_x"])

    plugin.on_subagent_terminated("subagent-1", "")

    # Both entries preserved — empty string is treated as "no session"
    # rather than as a key, so nothing matches.
    assert ("sess-A", "tool_a") in plugin._session_tool_successes
    assert ("", "tool_x") in plugin._session_tool_successes


def test_on_terminated_with_none_session_id_is_noop() -> None:
    plugin = ReliabilityPlugin()
    _populate_session_state(plugin, "sess-A", ["tool_a"])

    plugin.on_subagent_terminated("subagent-1", None)

    # Entries unchanged.
    assert plugin._session_tool_successes == {("sess-A", "tool_a"): 1}


def test_on_terminated_when_no_entries_for_session() -> None:
    """Closing a subagent whose session never recorded successes is
    a clean no-op — no errors, no spurious mutations."""
    plugin = ReliabilityPlugin()
    _populate_session_state(plugin, "sess-A", ["tool_a"])

    plugin.on_subagent_terminated("subagent-1", "sess-never-recorded")

    assert plugin._session_tool_successes == {("sess-A", "tool_a"): 1}


def test_on_terminated_method_signature_matches_subagent_contract() -> None:
    """Sanity check that the method can be called with the exact
    signature SubagentPlugin's auto-discovery uses:
    ``on_subagent_terminated(agent_id: str, session_id: Optional[str])``.
    """
    plugin = ReliabilityPlugin()
    # Callable with positional args, both str.
    plugin.on_subagent_terminated("agent-1", "sess-1")
    # Callable with positional args, session_id=None.
    plugin.on_subagent_terminated("agent-2", None)
