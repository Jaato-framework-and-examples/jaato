"""Tests for ``SessionManager._bootstrap_session`` (Phase 3 §3.12.0).

The helper is the unified construction point for the four
session-bootstrap paths.  These tests pin its contract using a
patched-JaatoServer fixture so the test doesn't actually load
plugins, talk to providers, or spawn runners — it exercises the
helper's own logic (envelope decomposition, hook ordering,
sandbox_mode read-back, Session-record assembly).

Coverage:

- Successful bootstrap returns ``(JaatoServer, Session)`` populated
  from the envelope's fields.
- Failed ``server.initialize()`` returns ``(None, None)``.
- Pre-initialize hooks fire BEFORE ``initialize()`` (the §3.12.0
  spec's lifecycle ordering).
- Pre-initialize hooks see the constructed server, the envelope's
  ``session_id`` / ``workspace_path`` / ``client_id``.
- ``config_root`` lands on the server BEFORE ``initialize()``.
- ``planned_sandbox_mode`` is read from the server's stash slot
  AFTER ``initialize()`` (so the IPC apparmor pre-init hook's
  side-effect surfaces on the Session record).
- ``envelope.sandbox_mode`` overrides the server stash when
  populated (disk-restore path's pre-known mode).
- ``timestamp`` propagation: explicit envelope timestamp lands on
  ``Session.created_at``; absent → fallback to ``datetime.now``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest

from shared.session_envelope import BootstrapEnvelope


class _FakeJaatoServer:
    """Stand-in for :class:`JaatoServer` capturing the kwargs the
    bootstrap helper passes in and exposing a controllable
    ``initialize`` outcome.  Phase 3 §3.13 removed the
    ``_planned_sandbox_mode`` slot — apparmor mode now flows via
    the SessionManager helper's return value."""

    instances: List["_FakeJaatoServer"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)
        # Default to success; tests override.
        self._initialize_outcome = True
        self.config_root: Optional[str] = None
        self.registry = None  # session_ops wiring is post-helper
        # Track call order so tests can assert lifecycle ordering.
        self.lifecycle_log: List[str] = []
        self.event_callbacks: List[Any] = []
        self._session_env: dict = {}
        _FakeJaatoServer.instances.append(self)

    def initialize(self) -> bool:
        self.lifecycle_log.append("initialize")
        return self._initialize_outcome

    def set_event_callback(self, cb: Any) -> None:
        self.event_callbacks.append(cb)

    # Phase 4 §B: SessionManager invokes these pre-spawn so the runner
    # fork inherits resolved secret URIs via os.environ.  The fake
    # records them in lifecycle_log + provides a no-op _with_session_env
    # context manager (real implementation mutates os.environ; the fake
    # spawn doesn't fork so behaviour is irrelevant).
    def _resolve_session_env(self) -> None:
        self.lifecycle_log.append("resolve_session_env")

    def _with_session_env(self):
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            self.lifecycle_log.append("with_session_env:enter")
            try:
                yield
            finally:
                self.lifecycle_log.append("with_session_env:exit")
        return _noop()

    # Server 0.6.131+ (PR-148): the helper creates the plugin registry
    # BEFORE composing the apparmor profile, so the composer can query
    # each plugin's ``get_apparmor_rules``.  That block runs inside
    # ``_in_workspace()`` so discovery resolves paths against the
    # session's workspace rather than the daemon's cwd.  Both are
    # recorded here because the ordering IS the contract these tests
    # pin — see ``test_lifecycle_ordering``.
    def _in_workspace(self):
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            self.lifecycle_log.append("in_workspace:enter")
            try:
                yield
            finally:
                self.lifecycle_log.append("in_workspace:exit")
        return _noop()

    def create_registry_and_discover(self) -> None:
        self.lifecycle_log.append("create_registry_and_discover")


class _FakeSessionManager:
    """Standalone test double exercising only _bootstrap_session.

    Reuses the real implementation by importing ``SessionManager``
    and binding the method to this instance — keeps the helper
    under test rather than reimplementing it.

    Phase 3 §3.13: also stubs
    ``_provision_ipc_apparmor_and_spawn_runner`` (the relocated
    inline IPC apparmor logic) so tests can drive its return value
    without exercising the real apparmor / spawn machinery.
    """

    def __init__(self) -> None:
        self.pre_init_hook_calls: List[Tuple[Any, str, Optional[str], Optional[str]]] = []
        # Capture lifecycle order across helper invocations.
        self.lifecycle_log: List[str] = []
        # §3.13 stub: return value from
        # _provision_ipc_apparmor_and_spawn_runner.  Tests set this
        # to drive the apparmor branch without real apparmor calls.
        self.ipc_provision_return: Optional[str] = None
        self.ipc_provision_calls: List[Tuple[Any, str, Optional[str], Optional[str]]] = []
        # Keyword-only overrides captured per call (see the stub below).
        self.ipc_provision_kwargs: List[Dict[str, Any]] = []

    def _run_pre_initialize_hooks(
        self,
        server: Any,
        session_id: str,
        workspace_path: Optional[str],
        client_id: Optional[str] = None,
    ) -> None:
        self.pre_init_hook_calls.append(
            (server, session_id, workspace_path, client_id),
        )
        if isinstance(server, _FakeJaatoServer):
            server.lifecycle_log.append("pre_init_hooks")
        self.lifecycle_log.append("pre_init_hooks")

    def _provision_ipc_apparmor_and_spawn_runner(
        self,
        server: Any,
        session_id: str,
        workspace_path: Optional[str],
        client_id: Optional[str],
        *,
        apparmor_override: Optional[bool] = None,
        config_root_override: Optional[str] = None,
        env_file_override: Optional[str] = None,
        apparmor_fragments_override: Optional[List[str]] = None,
        cascade_driver_id: Optional[str] = None,
    ) -> Optional[str]:
        # Keyword-only overrides are mirrored EXPLICITLY rather than
        # swallowed by ``**kwargs`` so that a future divergence from the
        # real signature fails here loudly instead of silently accepting
        # an argument the production method would reject.
        self.ipc_provision_calls.append(
            (server, session_id, workspace_path, client_id),
        )
        self.ipc_provision_kwargs.append({
            "apparmor_override": apparmor_override,
            "config_root_override": config_root_override,
            "env_file_override": env_file_override,
            "apparmor_fragments_override": apparmor_fragments_override,
            "cascade_driver_id": cascade_driver_id,
        })
        if isinstance(server, _FakeJaatoServer):
            server.lifecycle_log.append("ipc_apparmor_provision")
        self.lifecycle_log.append("ipc_apparmor_provision")
        return self.ipc_provision_return


@pytest.fixture(autouse=True)
def _patch_jaato_server():
    """Replace the ``JaatoServer`` symbol used by
    ``SessionManager._bootstrap_session`` with the fake."""
    _FakeJaatoServer.instances.clear()
    with patch("server.session_manager.JaatoServer", _FakeJaatoServer):
        yield


def _bind_helper(fake_mgr: _FakeSessionManager):
    """Bind the real ``_bootstrap_session`` to *fake_mgr* so the
    helper runs against the fake's pre-init hook + the fake
    JaatoServer.

    Phase 3 §3.12 disk-restore migration extracted
    ``_construct_and_initialize_server`` as the inner helper that
    ``_bootstrap_session`` delegates to.  Bind both so the fake_mgr
    invocation chain stays intact (without this, the
    ``self._construct_and_initialize_server(envelope)`` call inside
    ``_bootstrap_session`` would raise AttributeError on the fake)."""
    from server.session_manager import SessionManager
    fake_mgr._construct_and_initialize_server = (
        SessionManager._construct_and_initialize_server.__get__(
            fake_mgr, SessionManager,
        )
    )
    # Same reason as above: the funnel wires the manager into every plugin
    # that asks for it, so an unbound fake raises AttributeError mid-helper.
    # Binding the REAL method (rather than stubbing it) keeps the wiring
    # under test — it is the line whose absence on the disk-restore path
    # cost re-attached sessions their sibling verbs.
    fake_mgr._wire_session_manager_into_plugins = (
        SessionManager._wire_session_manager_into_plugins.__get__(
            fake_mgr, SessionManager,
        )
    )
    return SessionManager._bootstrap_session.__get__(fake_mgr, SessionManager)


# ----------------------------------------------------------------------
# Successful bootstrap → (JaatoServer, Session)
# ----------------------------------------------------------------------


def test_successful_bootstrap_returns_server_and_session() -> None:
    fake_mgr = _FakeSessionManager()
    helper = _bind_helper(fake_mgr)
    envelope = BootstrapEnvelope(
        session_id="sess-1",
        workspace_path="/tmp/ws",
        name="my-session",
    )
    server, session = helper(envelope)

    assert server is not None
    assert session is not None
    assert isinstance(server, _FakeJaatoServer)
    # Session record fields propagate from envelope.
    assert session.session_id == "sess-1"
    assert session.name == "my-session"
    assert session.workspace_path == "/tmp/ws"
    assert session.is_dirty is True


def test_jaato_server_constructed_with_envelope_fields() -> None:
    fake_mgr = _FakeSessionManager()
    helper = _bind_helper(fake_mgr)
    envelope = BootstrapEnvelope(
        session_id="sess-2",
        workspace_path="/tmp/ws",
        name="cfg",
        env_file="/tmp/.env",
        agent_name="researcher",
        env_overrides={"FOO": "bar"},
        suppress_base_instructions=True,
    )
    server, _ = helper(envelope)

    assert server is not None
    assert server.kwargs["env_file"] == "/tmp/.env"
    assert server.kwargs["agent_name"] == "researcher"
    assert server.kwargs["env_overrides"] == {"FOO": "bar"}
    assert server.kwargs["suppress_base_instructions"] is True
    assert server.kwargs["session_id"] == "sess-2"
    assert server.kwargs["workspace_path"] == "/tmp/ws"


# ----------------------------------------------------------------------
# Lifecycle ordering: config_root → pre_init_hooks → initialize
# ----------------------------------------------------------------------


def test_pre_init_hooks_fire_before_initialize() -> None:
    """Lifecycle ordering: Phase 4 §B pre-spawn env resolution +
    with_session_env wrap around the inline IPC apparmor provisioning,
    then pre-init hooks (WS + third-party), then ``server.initialize()``.

    The Phase 4 §B additions are:
    - ``resolve_session_env`` before the with_session_env wrap so
      ``self._session_env`` is populated when ``_with_session_env``
      reads it.
    - ``with_session_env:enter`` + ``with_session_env:exit`` flanking
      the inline IPC apparmor provisioning (and the runner spawn it
      drives), so fork() inherits the resolved env via ``os.environ``.
    """
    fake_mgr = _FakeSessionManager()
    helper = _bind_helper(fake_mgr)
    envelope = BootstrapEnvelope(
        session_id="sess-3",
        workspace_path="/tmp/ws",
        name="lc",
        client_id="client-A",
    )
    server, _ = helper(envelope)
    assert server is not None
    assert server.lifecycle_log == [
        "resolve_session_env",
        # PR-148: registry creation + discovery, inside the session's
        # workspace, BEFORE apparmor composition — the composer walks
        # ``profile.plugins`` via ``registry.get_plugin`` and silently
        # contributes zero plugin rules if the registry does not exist yet.
        "with_session_env:enter",
        "in_workspace:enter",
        "create_registry_and_discover",
        "in_workspace:exit",
        "with_session_env:exit",
        # then the apparmor provision + spawn, as before
        "with_session_env:enter",
        "ipc_apparmor_provision",
        "with_session_env:exit",
        "pre_init_hooks",
        "initialize",
    ]


def test_config_root_set_before_pre_init_hooks() -> None:
    """config_root must land on the server BEFORE pre-init hooks
    so plugins discovered during initialize see the override on
    their first set_config_root notification."""
    fake_mgr = _FakeSessionManager()
    helper = _bind_helper(fake_mgr)
    envelope = BootstrapEnvelope(
        session_id="sess-4",
        workspace_path="/tmp/ws",
        name="cr",
        config_root="/custom/.jaato",
    )
    server, _ = helper(envelope)
    assert server is not None
    assert server.config_root == "/custom/.jaato"
    # And the hook captured the server with config_root already set.
    hook_call = fake_mgr.pre_init_hook_calls[0]
    assert hook_call[0].config_root == "/custom/.jaato"


def test_pre_init_hooks_receive_session_id_workspace_client_id() -> None:
    fake_mgr = _FakeSessionManager()
    helper = _bind_helper(fake_mgr)
    envelope = BootstrapEnvelope(
        session_id="sess-5",
        workspace_path="/tmp/ws",
        name="hook",
        client_id="client-X",
    )
    helper(envelope)
    assert fake_mgr.pre_init_hook_calls
    _, sid, ws, cid = fake_mgr.pre_init_hook_calls[0]
    assert sid == "sess-5"
    assert ws == "/tmp/ws"
    assert cid == "client-X"


# ----------------------------------------------------------------------
# Failed initialize → (None, None)
# ----------------------------------------------------------------------


def test_failed_initialize_returns_none_pair() -> None:
    fake_mgr = _FakeSessionManager()
    helper = _bind_helper(fake_mgr)

    # Patch _FakeJaatoServer's __init__ to mark the next instance
    # as failing-to-initialize.
    original_init = _FakeJaatoServer.__init__

    def _failing_init(self: _FakeJaatoServer, **kwargs: Any) -> None:
        original_init(self, **kwargs)
        self._initialize_outcome = False

    with patch.object(_FakeJaatoServer, "__init__", _failing_init):
        envelope = BootstrapEnvelope(
            session_id="sess-fail",
            workspace_path=None,
            name="no",
        )
        server, session = helper(envelope)

    assert server is None
    assert session is None


# ----------------------------------------------------------------------
# sandbox_mode resolution (Phase 3 §3.13)
#
# After §3.13 the IPC apparmor logic lives inline in
# ``SessionManager._provision_ipc_apparmor_and_spawn_runner``,
# called from ``_bootstrap_session`` BEFORE pre-init hooks fire.
# The Session record's ``sandbox_mode`` resolves with priority:
#   1. ``envelope.sandbox_mode`` (disk-restore path's pre-known value)
#   2. The IPC provisioning method's return value
#   3. None (no opt-in / non-confined)
# The Phase 2 ``server._planned_sandbox_mode`` slot was removed.
# ----------------------------------------------------------------------


def test_ipc_provisioning_apparmor_lands_on_session() -> None:
    """When the inline IPC apparmor provisioning returns
    ``"apparmor"``, the Session record reflects it."""
    fake_mgr = _FakeSessionManager()
    fake_mgr.ipc_provision_return = "apparmor"
    helper = _bind_helper(fake_mgr)

    envelope = BootstrapEnvelope(
        session_id="sess-6",
        workspace_path="/tmp/ws",
        name="sandbox-test",
        client_id="client-A",
    )
    _, session = helper(envelope)
    assert session is not None
    assert session.sandbox_mode == "apparmor"
    # Provision was called with the envelope's identity fields.
    assert len(fake_mgr.ipc_provision_calls) == 1
    _, sid, ws, cid = fake_mgr.ipc_provision_calls[0]
    assert sid == "sess-6"
    assert ws == "/tmp/ws"
    assert cid == "client-A"


def test_ipc_provisioning_soft_lands_on_session() -> None:
    """When apparmor is requested but unavailable / provisioning
    fails, the IPC method returns ``"soft"`` and the Session
    record reflects the downgrade."""
    fake_mgr = _FakeSessionManager()
    fake_mgr.ipc_provision_return = "soft"
    helper = _bind_helper(fake_mgr)

    envelope = BootstrapEnvelope(
        session_id="sess-soft",
        workspace_path="/tmp/ws",
        name="soft-fallback",
        client_id="client-B",
    )
    _, session = helper(envelope)
    assert session is not None
    assert session.sandbox_mode == "soft"


def test_envelope_sandbox_mode_overrides_ipc_provision() -> None:
    """Disk-restore path: envelope.sandbox_mode is the authoritative
    pre-restart value; the helper must prefer it over whatever the
    inline IPC provisioning returns.  This pins the precedence
    documented on ``_bootstrap_session``."""
    fake_mgr = _FakeSessionManager()
    fake_mgr.ipc_provision_return = "apparmor"  # would normally win
    helper = _bind_helper(fake_mgr)

    envelope = BootstrapEnvelope(
        session_id="sess-7",
        workspace_path="/tmp/ws",
        name="disk-restore",
        sandbox_mode="kernel",  # authoritative
    )
    _, session = helper(envelope)
    assert session is not None
    assert session.sandbox_mode == "kernel"


def test_no_ipc_provision_no_envelope_yields_none() -> None:
    """Plain session with no apparmor opt-in (provisioning method
    returns None) and no envelope-supplied mode: Session record's
    ``sandbox_mode`` is None."""
    fake_mgr = _FakeSessionManager()
    fake_mgr.ipc_provision_return = None
    helper = _bind_helper(fake_mgr)

    envelope = BootstrapEnvelope(
        session_id="sess-8",
        workspace_path=None,
        name="plain",
    )
    _, session = helper(envelope)
    assert session is not None
    assert session.sandbox_mode is None


def test_ipc_provision_fires_before_pre_init_hooks() -> None:
    """§3.13 lifecycle ordering: the relocated IPC apparmor logic
    runs BEFORE ``_run_pre_initialize_hooks`` (which still fires
    for the WS-side hook + third-party hooks).  Tests that the
    runner is up before any pre-init hook observes the server."""
    fake_mgr = _FakeSessionManager()
    helper = _bind_helper(fake_mgr)

    envelope = BootstrapEnvelope(
        session_id="sess-order",
        workspace_path="/tmp/ws",
        name="order",
        client_id="client-X",
    )
    _, session = helper(envelope)
    assert session is not None
    # Lifecycle log on the SessionManager records both events in
    # the correct order.
    assert fake_mgr.lifecycle_log == [
        "ipc_apparmor_provision",
        "pre_init_hooks",
    ]


# ----------------------------------------------------------------------
# Timestamp propagation
# ----------------------------------------------------------------------


def test_envelope_timestamp_lands_on_session_created_at() -> None:
    fake_mgr = _FakeSessionManager()
    helper = _bind_helper(fake_mgr)
    ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    envelope = BootstrapEnvelope(
        session_id="sess-9",
        workspace_path=None,
        name="time-test",
        timestamp=ts,
    )
    _, session = helper(envelope)
    assert session is not None
    assert session.created_at == ts.isoformat()


def test_missing_timestamp_falls_back_to_now() -> None:
    fake_mgr = _FakeSessionManager()
    helper = _bind_helper(fake_mgr)
    envelope = BootstrapEnvelope(
        session_id="sess-10",
        workspace_path=None,
        name="no-ts",
    )
    _, session = helper(envelope)
    assert session is not None
    # Any ISO-format timestamp is acceptable; just check non-empty.
    assert session.created_at
    # Should parse as ISO.
    datetime.fromisoformat(session.created_at)


# ----------------------------------------------------------------------
# Other Session record fields
# ----------------------------------------------------------------------


def test_provisioned_and_created_by_propagate() -> None:
    fake_mgr = _FakeSessionManager()
    helper = _bind_helper(fake_mgr)
    envelope = BootstrapEnvelope(
        session_id="sess-11",
        workspace_path="/tmp/ws",
        name="provisioned-test",
        provisioned=True,
        created_by="operator-X",
    )
    _, session = helper(envelope)
    assert session is not None
    assert session.provisioned is True
    assert session.created_by == "operator-X"


def test_description_propagates() -> None:
    fake_mgr = _FakeSessionManager()
    helper = _bind_helper(fake_mgr)
    envelope = BootstrapEnvelope(
        session_id="sess-12",
        workspace_path="/tmp/ws",
        name="desc-test",
        description="bootstrapped by test",
    )
    _, session = helper(envelope)
    assert session is not None
    assert session.description == "bootstrapped by test"
