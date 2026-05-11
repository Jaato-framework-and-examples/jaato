"""Phase 4 §D regression pins: agent_params → runner propagation.

Third post-§7c envelope-propagation gap (after §B env vars and §C
plugin_configs).  Pre-§D the ``runner_spawn.build_session_envelope``
helper hard-coded ``agent_params={}`` on the wire envelope, so the
originating ``client.create_session(agent_params={...})`` call from a
SDK client never reached the runner-side ``JaatoSession._agent_params``.
The runner's prefetch scripts (``{{!py:...}}`` placeholders rendered
during ``JaatoSession.configure()``) saw an empty
``context.agent_params`` and emitted their "[prefetch error: missing
required keys]" block — which caused the documenter agent in the
self-documenting harness to hallucinate tmux pane id ``0`` instead of
the harness-provided ``7:1``, dumping its keystrokes into the
operator's interactive panel.

Fix shape: forward agent_params through the existing channels.

1. ``BootstrapEnvelope`` (daemon-internal) gains an ``agent_params``
   field; ``_create_session_impl`` populates it from the IPC request.
2. ``_construct_and_initialize_server`` stashes a copy on the
   per-session JaatoServer instance (``server._agent_params``).  This
   is per-session, not centralized daemon state — the JaatoServer
   lives only for the duration of the session.
3. ``build_session_envelope`` reads ``server._agent_params`` and
   passes them to ``SessionInitEnvelope(agent_params=...)``.
4. Runner-side: zero changes.  The pre-existing
   ``runtime.create_session(agent_params=envelope.agent_params or
   None)`` call (``runner/session.py:489``) was already wiring the
   field through to ``JaatoSession._agent_params``; it just never had
   anything to forward.

These tests pin the writer side (build_session_envelope reads from
server._agent_params) and the wire-envelope shape end-to-end.

The integration counterpart (rerun documenter harness, expect
``tmux send-keys -t 7:1 ...`` to land in the harness window instead
of the operator's pane) lives in the harness workspace; see
``docs/design/phase4_agent_params_passdown_audit.md``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from shared.session_envelope import SessionInitEnvelope
from server.runner_spawn import build_session_envelope


def _stub_profile(**overrides: Any) -> SimpleNamespace:
    base = dict(
        provider="anthropic",
        model="m",
        plugins=[],
        plugin_configs={},
        preloaded_plugins=set(),
        system_instructions=None,
        gc=None,
        env={},
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _stub_server(profile: Any, agent_params: Any = None) -> SimpleNamespace:
    """Mirror the JaatoServer-shape that build_session_envelope reads.

    Phase 4 §D adds ``_agent_params`` to the attribute set.
    SessionManager._construct_and_initialize_server stashes the
    originating create_session.agent_params here at envelope-build
    time."""
    return SimpleNamespace(
        _profile=profile,
        config_root=None,
        _agent_params=dict(agent_params or {}),
    )


def test_envelope_agent_params_empty_when_server_has_no_stash() -> None:
    """Server with no ``_agent_params`` attribute (or empty dict) →
    envelope.agent_params is ``{}``.  This is the pre-§D behavior
    preserved for sessions created without agent_params (e.g. profile-
    only, no agent persona)."""
    server = SimpleNamespace(_profile=None, config_root=None)
    env = build_session_envelope(
        server=server,
        session_id="s",
        workspace_path="/tmp/ws",
        profile_name=None,
    )
    assert env.agent_params == {}


def test_envelope_agent_params_forwarded_from_server_stash() -> None:
    """Server with stashed ``_agent_params`` → envelope.agent_params
    carries the same dict (deep-copied so mutations don't leak)."""
    profile = _stub_profile()
    server = _stub_server(profile=profile, agent_params={
        "feature_id": "exit-menu",
        "tmux_pane": "7:1",
        "feature_title": "Exit menu",
    })
    env = build_session_envelope(
        server=server,
        session_id="s",
        workspace_path="/tmp/ws",
        profile_name="documenter",
    )
    assert env.agent_params == {
        "feature_id": "exit-menu",
        "tmux_pane": "7:1",
        "feature_title": "Exit menu",
    }


def test_envelope_agent_params_dict_is_independent_copy() -> None:
    """Mutating the envelope's agent_params must NOT mutate the
    server's stash.  Defensive copy prevents accidental aliasing
    between daemon-side and runner-side views."""
    original = {"feature_id": "exit-menu", "tmux_pane": "7:1"}
    profile = _stub_profile()
    server = _stub_server(profile=profile, agent_params=original)
    env = build_session_envelope(
        server=server,
        session_id="s",
        workspace_path="/tmp/ws",
        profile_name="documenter",
    )
    env.agent_params["mutation_test"] = "added"
    assert "mutation_test" not in original


def test_session_init_envelope_agent_params_round_trip() -> None:
    """The wire-format ``to_dict``/``from_dict`` survives the new
    field's contents.  Already covered by older v1 fixtures, but
    pinning the §D-typical shape here makes the integration story
    visible alongside the writer-side tests."""
    env = SessionInitEnvelope(
        session_id="s",
        workspace_path=None,
        profile_name="documenter",
        provider_name="openrouter",
        model_name="nvidia/nemotron-3-super-120b-a12b:free",
        plugins=[],
        agent_params={"feature_id": "exit-menu", "tmux_pane": "7:1"},
    )
    d = env.to_dict()
    assert d["agent_params"] == {"feature_id": "exit-menu", "tmux_pane": "7:1"}
    env2 = SessionInitEnvelope.from_dict(d)
    assert env2.agent_params == {"feature_id": "exit-menu", "tmux_pane": "7:1"}
