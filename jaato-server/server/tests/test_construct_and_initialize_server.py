"""Tests for ``SessionManager._construct_and_initialize_server`` and
its disk-restore consumer (Phase 3 §3.12 disk-restore migration).

The sub-helper splits out from ``_bootstrap_session`` so the
disk-restore path (``_load_session_impl``) can share the
JaatoServer-construction-and-init logic without inheriting the
create-session Session-record shape.  These tests pin the contract:

- The sub-helper returns ``(server, sandbox_mode)`` on success and
  ``(None, None)`` on init failure.
- ``envelope.sandbox_mode`` wins over the IPC method's return
  value (disk-restore's pre-known mode is authoritative).
- Lifecycle ordering: IPC apparmor provision → pre-init hooks →
  initialize.
- ``_bootstrap_session`` delegates to it.

The disk-restore route through the helper is tested by exercising
``_load_session_impl`` against a stub session-plugin that returns
a saved state — verifies the BootstrapEnvelope is built with
``client_id=None`` and ``sandbox_mode`` from the saved state.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest

from shared.session_envelope import BootstrapEnvelope


# ----------------------------------------------------------------------
# Shared fixtures (mirror test_bootstrap_helper.py shape)
# ----------------------------------------------------------------------


class _FakeJaatoServer:
    """Stand-in for :class:`JaatoServer`.  Mirrors the test-helper
    fake from test_bootstrap_helper.py."""

    instances: List["_FakeJaatoServer"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)
        self._initialize_outcome = True
        self.config_root: Optional[str] = None
        self.registry = None
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

    # Server 0.6.131+ (PR-148): the bootstrap helper creates the plugin
    # registry inside ``_in_workspace()`` BEFORE composing the apparmor
    # profile, so the composer can query each plugin's apparmor rules.
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
    """Test double exposing the same stubs as in
    test_bootstrap_helper.py."""

    def __init__(self) -> None:
        self.pre_init_hook_calls: List[Tuple[Any, str, Optional[str], Optional[str]]] = []
        self.lifecycle_log: List[str] = []
        self.ipc_provision_return: Optional[str] = None
        self.ipc_provision_calls: List[Tuple[Any, str, Optional[str], Optional[str]]] = []

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
        # Mirrors the production signature EXPLICITLY (not **kwargs) so a
        # future divergence fails loudly here instead of silently accepting
        # an argument the real method would reject.
        apparmor_fragments_override: Optional[List[str]] = None,
        cascade_driver_id: Optional[str] = None,
    ) -> Optional[str]:
        self.ipc_provision_calls.append(
            (server, session_id, workspace_path, client_id),
        )
        if isinstance(server, _FakeJaatoServer):
            server.lifecycle_log.append("ipc_apparmor_provision")
        self.lifecycle_log.append("ipc_apparmor_provision")
        return self.ipc_provision_return


@pytest.fixture(autouse=True)
def _patch_jaato_server():
    _FakeJaatoServer.instances.clear()
    with patch("server.session_manager.JaatoServer", _FakeJaatoServer):
        yield


def _bind_subhelper(fake_mgr: _FakeSessionManager):
    """Bind the real ``_construct_and_initialize_server`` to *fake_mgr*."""
    from server.session_manager import SessionManager
    return SessionManager._construct_and_initialize_server.__get__(
        fake_mgr, SessionManager,
    )


# ----------------------------------------------------------------------
# Successful path: returns (server, sandbox_mode)
# ----------------------------------------------------------------------


def test_subhelper_returns_server_and_sandbox_mode_tuple() -> None:
    fake_mgr = _FakeSessionManager()
    fake_mgr.ipc_provision_return = None
    helper = _bind_subhelper(fake_mgr)
    envelope = BootstrapEnvelope(
        session_id="s-1",
        workspace_path="/tmp/ws",
        name="x",
    )
    server, sandbox_mode = helper(envelope)
    assert server is not None
    assert isinstance(server, _FakeJaatoServer)
    assert sandbox_mode is None


def test_subhelper_envelope_sandbox_mode_wins() -> None:
    """Disk-restore's pre-known mode wins over the IPC method
    return value (which still gets called but its result is
    overridden)."""
    fake_mgr = _FakeSessionManager()
    fake_mgr.ipc_provision_return = "soft"  # would normally win
    helper = _bind_subhelper(fake_mgr)
    envelope = BootstrapEnvelope(
        session_id="s-2",
        workspace_path="/tmp/ws",
        name="restore",
        sandbox_mode="apparmor",  # authoritative
    )
    _, sandbox_mode = helper(envelope)
    assert sandbox_mode == "apparmor"


def test_subhelper_ipc_provision_result_used_when_envelope_none() -> None:
    fake_mgr = _FakeSessionManager()
    fake_mgr.ipc_provision_return = "apparmor"
    helper = _bind_subhelper(fake_mgr)
    envelope = BootstrapEnvelope(
        session_id="s-3",
        workspace_path="/tmp/ws",
        name="ipc",
        client_id="client-1",
    )
    _, sandbox_mode = helper(envelope)
    assert sandbox_mode == "apparmor"


def test_subhelper_init_failure_returns_none_pair() -> None:
    fake_mgr = _FakeSessionManager()
    helper = _bind_subhelper(fake_mgr)

    original_init = _FakeJaatoServer.__init__

    def _failing_init(self: _FakeJaatoServer, **kwargs: Any) -> None:
        original_init(self, **kwargs)
        self._initialize_outcome = False

    with patch.object(_FakeJaatoServer, "__init__", _failing_init):
        envelope = BootstrapEnvelope(
            session_id="s-fail",
            workspace_path=None,
            name="no",
        )
        server, sandbox_mode = helper(envelope)

    assert server is None
    assert sandbox_mode is None


# ----------------------------------------------------------------------
# Lifecycle ordering
# ----------------------------------------------------------------------


def test_subhelper_lifecycle_order_matches_bootstrap_session() -> None:
    """Phase 4 §B lifecycle: resolve_session_env → with_session_env
    wrap around the inline IPC apparmor provisioning (which drives
    the runner spawn whose fork must see resolved env in os.environ)
    → pre-init hooks → ``server.initialize()``."""
    fake_mgr = _FakeSessionManager()
    helper = _bind_subhelper(fake_mgr)
    envelope = BootstrapEnvelope(
        session_id="s-order",
        workspace_path="/tmp/ws",
        name="o",
        client_id="c-1",
    )
    server, _ = helper(envelope)
    assert server is not None
    assert server.lifecycle_log == [
        "resolve_session_env",
        # PR-148: registry creation + discovery run inside the session's
        # workspace BEFORE apparmor composition, so the composer can walk
        # ``profile.plugins`` via ``registry.get_plugin``.  Without the
        # registry, zero plugin-contributed rules reach the profile.
        "with_session_env:enter",
        "in_workspace:enter",
        "create_registry_and_discover",
        "in_workspace:exit",
        "with_session_env:exit",
        # then apparmor provision + spawn
        "with_session_env:enter",
        "ipc_apparmor_provision",
        "with_session_env:exit",
        "pre_init_hooks",
        "initialize",
    ]


def test_subhelper_config_root_set_before_pre_init_hooks() -> None:
    fake_mgr = _FakeSessionManager()
    helper = _bind_subhelper(fake_mgr)
    envelope = BootstrapEnvelope(
        session_id="s-cr",
        workspace_path="/tmp/ws",
        name="cr",
        config_root="/custom/.jaato",
    )
    server, _ = helper(envelope)
    assert server is not None
    assert server.config_root == "/custom/.jaato"
    # Pre-init hook saw it set.
    hook_call = fake_mgr.pre_init_hook_calls[0]
    assert hook_call[0].config_root == "/custom/.jaato"


# ----------------------------------------------------------------------
# Disk-restore route: _load_session_impl threads through the sub-helper
# ----------------------------------------------------------------------


def test_load_session_impl_uses_sub_helper_with_no_client_id(
    tmp_path,
) -> None:
    """Disk-restore must call the sub-helper with ``client_id=None``
    (no client-driven apparmor opt-in on restore) and the saved
    ``state.sandbox_mode`` populated on the envelope so the
    Session record reflects the pre-restart value."""
    from server.session_manager import SessionManager

    sm = SessionManager()

    # Stub the session plugin to return a saved state.
    class _FakeState:
        def __init__(self) -> None:
            self.workspace_path = str(tmp_path)
            self.description = "saved-name"
            self.history: List[Any] = []
            self.user_inputs: List[str] = []
            self.workspace_files: List[str] = []
            self.metadata: Dict[str, Any] = {}
            self.created_at = datetime.now(timezone.utc)
            self.updated_at = datetime.now(timezone.utc)
            self.budget_state = None
            self.turn_accounting: List[Any] = []
            self.interrupted_turn = None
            self.sandbox_mode = "apparmor"  # pre-restart value
            # Mirrors the real SessionState field (plugins/session/base.py:92);
            # the disk-restore path reads it to resolve the re-spawned runner's
            # config_root.
            self.config_root: Optional[str] = None
            # Mirrors SessionState.profile_spec (plugins/session/base.py:70) —
            # the inline-profile restore path reads it.
            self.profile_spec: Optional[Dict[str, Any]] = None
            # Mirrors SessionState.profile_name (base.py:52); "<inline>"
            # marks a session born from an inline profile rather than a file.
            self.profile_name: Optional[str] = None

    state = _FakeState()
    sm._session_plugin.load = lambda sid, storage_dir=None: state  # type: ignore[assignment]

    # Capture the envelope passed into the sub-helper.  Force it
    # to return ``(None, None)`` so ``_load_session_impl`` exits
    # early — the test only verifies the envelope shape, not the
    # full restore-pipeline (history, monitor, plugin states etc.
    # are exercised by the broader load-session integration tests).
    captured_envelopes: List[BootstrapEnvelope] = []

    def _stub(self: Any, envelope: BootstrapEnvelope):
        captured_envelopes.append(envelope)
        return None, None  # signals init failure → caller returns early

    with patch(
        "server.session_manager.SessionManager._construct_and_initialize_server",
        _stub,
    ):
        result = sm._load_session_impl(
            session_id="restored-1",
            client_id=None,
            workspace_path=str(tmp_path),
        )

    # Sub-helper was called once with a disk-restore-shaped envelope.
    assert len(captured_envelopes) == 1
    env = captured_envelopes[0]
    assert env.session_id == "restored-1"
    assert env.client_id is None  # disk-restore has no client opt-in
    assert env.sandbox_mode == "apparmor"  # carried from saved state
    # restore_state field carries the saved state for §3.12 follow-on
    # consumers (defer-and-flush etc.).
    assert env.restore_state is not None
    assert env.restore_state.get("loaded_state") is state
