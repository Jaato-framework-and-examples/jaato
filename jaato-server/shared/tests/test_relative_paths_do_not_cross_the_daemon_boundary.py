"""Regression: a relative path must be REFUSED at the daemon boundary.

Root cause closed here (issue #742, 2026-08-30): the daemon accepted a
relative ``workspace_path`` from a client and absolutised it against its
OWN cwd.  When the two processes had been started from different
directories — which nothing prevented, and which a daemon restart could
change — the session's workspace silently split in half.  Both log lines
were present in the same run::

    workspace=.jaato-eval-workspaces/issue-fix_sweep@openrouter_gpt5mini_0
    workspace=/home/…/jaato/.jaato-eval-workspaces/issue-fix_sweep@…_0

The first is what was sent; the second is what the daemon resolved it to.
The harness wrote each arm's fixture (``.env``, acceptance script) into
one directory; the runner created the git worktree and ``.base_commit``
in the other.  So the agent got a workspace holding its repository but
not its fixture, the grader graded a workspace holding the fixture but no
repository, the arm made 25 provider calls and committed nothing, and the
grader reported it blocked for a missing ``.base_commit`` that existed
the whole time in the other half.  No error was raised on either side.

WHAT THESE TESTS ASSERT, AND WHY IT IS REJECTION.  A test that checked
"the relative path resolved to the right directory" would have passed
before the fix, because a client and daemon sharing a cwd agree by
accident — which is exactly why this survived several green runs.  Every
test here therefore asserts the REFUSAL: the boundary raises or emits an
error and applies nothing.  The absolute-path controls exist only to show
the guards discriminate.

The boundary has four enforcement points, one per way a path reaches it:

1. ``IPCClient.__init__`` — the sending half.  Refuses to put a relative
   path on the wire at all.
2. ``CommandRouter`` ``set_workspace`` — the connect-time handshake
   command.
3. ``SessionManager._apply_client_config`` — the ``ClientConfigRequest``
   handshake, which carries ``working_dir`` plus the three sibling paths
   (``config_root``, ``env_file``, the trace logs) that cross by the same
   mechanism.
4. ``BootstrapEnvelope`` / ``SessionInitEnvelope`` — the envelopes.  A
   session bootstrapped with a relative path must FAIL, not resolve;
   this is the backstop for any path that reaches session construction
   without passing 1-3.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pytest

from jaato_sdk.path_boundary import (
    RelativePathAcrossBoundaryError,
    describe_relative_path,
    require_absolute_path,
)
from shared.session_envelope import BootstrapEnvelope, SessionInitEnvelope
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

#: The defect, put back at each of the four enforcement points.  Each
#: reversion restores the pre-fix behaviour — accept the value and let
#: the receiver resolve it — and names the one test that must notice.
REVERSIONS = [
    Reversion(
        target="jaato-server/shared/session_envelope.py",
        find="""        _origin = "the session bootstrap envelope"
        require_absolute_path(
            self.workspace_path, field="workspace_path", origin=_origin)""",
        replace="""        _origin = "the session bootstrap envelope"
        _ = self.workspace_path""",
        test="test_bootstrap_envelope_refuses_a_relative_workspace",
        because="a session bootstrapping with a relative workspace path, "
                "which the runner then resolves against the daemon's cwd",
    ),
    Reversion(
        target="jaato-server/server/session_manager.py",
        find="""        if self._reject_relative_client_paths(client_id, event):
            return""",
        replace="""        if False:
            return""",
        test="test_client_config_with_a_relative_working_dir_is_refused",
        because="the daemon storing a client's relative working_dir and "
                "absolutising it against its own cwd",
    ),
    Reversion(
        target="jaato-server/server/command_router.py",
        find="""        bad = describe_relative_path(
            "workspace", workspace_path,
            origin="the daemon boundary (set_workspace)",
        )""",
        replace="""        bad = None""",
        test="test_set_workspace_with_a_relative_path_is_refused",
        because="set_workspace registering a relative workspace for the "
                "client, which every later session inherits",
    ),
    Reversion(
        target="jaato-sdk/jaato_sdk/client/ipc.py",
        find="""        require_absolute_path(
            workspace_path, field="workspace_path",
            origin="the daemon boundary",
        )""",
        replace="""        pass""",
        test="test_the_sdk_refuses_to_put_a_relative_workspace_on_the_wire",
        because="an SDK client sending a relative workspace path, whose "
                "meaning it alone knows and did not transmit",
    ),
]


# ----------------------------------------------------------------------
# The contract itself
# ----------------------------------------------------------------------


def test_a_relative_path_is_described_and_an_absolute_one_is_not() -> None:
    """``describe_relative_path`` is the single discriminator.

    Absent is not a violation: a path nobody supplied is a different
    contract from a path supplied wrongly.
    """
    message = describe_relative_path(
        "workspace_path", ".jaato-eval-workspaces/arm0")
    assert message is not None
    # The message must name the offending VALUE.  An error that says only
    # "relative path rejected" leaves the caller grepping two filesystems
    # for which of its four paths was the bad one.
    assert ".jaato-eval-workspaces/arm0" in message
    assert "workspace_path" in message

    assert describe_relative_path("workspace_path", "/abs/ws") is None
    assert describe_relative_path("workspace_path", "") is None
    assert describe_relative_path("workspace_path", None) is None


def test_the_guard_refuses_rather_than_resolving() -> None:
    """The value is never absolutised on the way through.

    Resolving is what produced the defect: it supplies the sender's
    missing half from the wrong process.  ``require_absolute_path``
    returns what it was given, or raises.
    """
    assert require_absolute_path("/abs/ws", field="workspace_path") == "/abs/ws"
    with pytest.raises(RelativePathAcrossBoundaryError) as exc:
        require_absolute_path("rel/ws", field="workspace_path")
    assert exc.value.field == "workspace_path"
    assert exc.value.value == "rel/ws"
    # A ValueError subclass, so existing envelope validation handlers
    # (which catch ValueError) keep working.
    assert isinstance(exc.value, ValueError)


# ----------------------------------------------------------------------
# 4. The envelopes — "a session bootstrapped with a relative path fails"
# ----------------------------------------------------------------------


def test_bootstrap_envelope_refuses_a_relative_workspace() -> None:
    with pytest.raises(RelativePathAcrossBoundaryError) as exc:
        BootstrapEnvelope(
            session_id="s1",
            workspace_path=".jaato-eval-workspaces/arm0",
            name="arm0",
        )
    assert ".jaato-eval-workspaces/arm0" in str(exc.value)


def test_bootstrap_envelope_refuses_the_sibling_paths_too() -> None:
    """``config_root`` and ``env_file`` cross by the same mechanism.

    Auditing only ``workspace_path`` would leave a session reading its
    profiles and its provider credentials from the daemon's cwd while
    working in the client's — a quieter version of the same split.
    """
    for field, value in (("config_root", "task/.jaato"),
                         ("env_file", ".env")):
        with pytest.raises(RelativePathAcrossBoundaryError) as exc:
            BootstrapEnvelope(
                session_id="s1", workspace_path="/abs/ws", name="n",
                **{field: value},
            )
        assert exc.value.field == field


def test_bootstrap_envelope_accepts_absolute_paths() -> None:
    envelope = BootstrapEnvelope(
        session_id="s1", workspace_path="/abs/ws", name="n",
        config_root="/abs/task/.jaato", env_file="/abs/ws/.env",
    )
    assert envelope.workspace_path == "/abs/ws"


def test_session_init_envelope_refuses_a_relative_workspace() -> None:
    with pytest.raises(RelativePathAcrossBoundaryError):
        SessionInitEnvelope(
            session_id="s1",
            workspace_path="relative/ws",
            profile_name=None,
            provider_name="anthropic",
            model_name="claude-sonnet-4-5",
        )


def test_session_init_envelope_refuses_a_relative_path_off_the_wire() -> None:
    """The runner-side deserializer enforces it too.

    ``from_dict`` is where a runner receives the daemon's envelope; a
    daemon built before this guard could still send one.
    """
    wire = SessionInitEnvelope(
        session_id="s1", workspace_path="/abs/ws", profile_name=None,
        provider_name="anthropic", model_name="m",
    ).to_dict()
    wire["workspace_path"] = "relative/ws"
    with pytest.raises(RelativePathAcrossBoundaryError):
        SessionInitEnvelope.from_dict(wire)


def test_session_init_envelope_survives_a_round_trip_when_absolute() -> None:
    envelope = SessionInitEnvelope(
        session_id="s1", workspace_path="/abs/ws", profile_name=None,
        provider_name="anthropic", model_name="m", config_root="/abs/cfg",
    )
    assert (SessionInitEnvelope.from_dict(envelope.to_dict()).workspace_path
            == "/abs/ws")


# ----------------------------------------------------------------------
# 3. The ClientConfigRequest handshake
# ----------------------------------------------------------------------


class _FakeManager:
    """Just enough ``SessionManager`` for ``_apply_client_config``.

    Holds the two collaborators the method touches on the reject path
    (``_client_config``, ``_emit_to_client``) plus the session lookup it
    reaches on the accept path.
    """

    def __init__(self) -> None:
        self._client_config: Dict[str, Dict[str, Any]] = {}
        self._client_to_session: Dict[str, str] = {}
        self._sessions: Dict[str, Any] = {}
        self.emitted: List[Any] = []

    def _emit_to_client(self, client_id: str, event: Any) -> None:
        self.emitted.append(event)


def _apply_client_config(manager: _FakeManager, event: Any) -> None:
    from server.session_manager import SessionManager

    manager._reject_relative_client_paths = (
        SessionManager._reject_relative_client_paths.__get__(
            manager, SessionManager))
    manager._CLIENT_CONFIG_PATH_FIELDS = (
        SessionManager._CLIENT_CONFIG_PATH_FIELDS)
    # Bound rather than stubbed: the live-session push is the other half
    # of "apply", and a stub would let a reversion that skipped validation
    # still look clean here.
    manager._apply_client_config_to_live_session = (
        SessionManager._apply_client_config_to_live_session.__get__(
            manager, SessionManager))
    SessionManager._apply_client_config.__get__(manager, SessionManager)(
        "client-1", event)


def _client_config_request(**kwargs: Any) -> Any:
    from jaato_sdk.events import ClientConfigRequest
    return ClientConfigRequest(**kwargs)


def test_client_config_with_a_relative_working_dir_is_refused() -> None:
    manager = _FakeManager()
    _apply_client_config(manager, _client_config_request(
        working_dir=".jaato-eval-workspaces/arm0"))

    assert len(manager.emitted) == 1
    error = manager.emitted[0]
    assert error.error_type == "RelativePathAcrossBoundary"
    assert ".jaato-eval-workspaces/arm0" in error.error
    # And nothing was stored: a refusal that still applied the value
    # would be the original defect with an extra log line.
    assert manager._client_config == {}


def test_a_refused_client_config_applies_none_of_its_other_fields() -> None:
    """One bad path rejects the whole handshake.

    Applying the good half would leave the session with, say, a valid
    workspace and no ``config_root`` — a different silent-wrong-directory
    bug rather than a fixed one.
    """
    manager = _FakeManager()
    _apply_client_config(manager, _client_config_request(
        working_dir="/abs/ws",
        config_root="task/.jaato",
        env_file="/abs/ws/.env",
    ))

    assert manager._client_config == {}
    assert manager.emitted and "task/.jaato" in manager.emitted[0].error


def test_every_path_bearing_field_of_the_handshake_is_checked() -> None:
    """The audit the issue asked for, pinned.

    ``config_root``, ``env_file`` and the two trace-log paths reach the
    daemon's filesystem exactly as ``working_dir`` does.  A guard that
    covered only the workspace would leave the mechanism in place.
    """
    for field in ("working_dir", "config_root", "env_file",
                  "trace_log_path", "provider_trace_log"):
        manager = _FakeManager()
        _apply_client_config(manager, _client_config_request(
            **{field: "some/relative/path"}))
        assert manager.emitted, f"{field} crossed the boundary unchecked"
        assert field in manager.emitted[0].error
        assert manager._client_config == {}


def test_an_absolute_client_config_is_applied() -> None:
    manager = _FakeManager()
    _apply_client_config(manager, _client_config_request(
        working_dir="/abs/ws", config_root="/abs/task/.jaato",
        env_file="/abs/ws/.env"))

    assert manager.emitted == []
    assert manager._client_config["client-1"]["working_dir"] == "/abs/ws"
    assert manager._client_config["client-1"]["config_root"] == "/abs/task/.jaato"


# ----------------------------------------------------------------------
# 2. The set_workspace handshake command
# ----------------------------------------------------------------------


class _FakeEventSink:
    """Records what the router would have registered / sent."""

    def __init__(self) -> None:
        self.workspaces: Dict[str, str] = {}
        self.events: List[Any] = []

    def set_client_workspace(self, client_id: str, workspace_path: str) -> None:
        self.workspaces[client_id] = workspace_path

    def get_client_workspace(self, client_id: str) -> Optional[str]:
        return self.workspaces.get(client_id)

    def send_event(self, client_id: str, event: Any) -> None:
        self.events.append(event)


def _router(sink: _FakeEventSink) -> Any:
    from unittest.mock import MagicMock

    from server.command_router import CommandRouter
    return CommandRouter(MagicMock(), sink, {})


def _set_workspace(router: Any, path: str) -> None:
    from jaato_sdk.events import CommandRequest
    router._dispatch(
        "client-1", "", CommandRequest(command="set_workspace", args=[path]))


def test_set_workspace_with_a_relative_path_is_refused() -> None:
    sink = _FakeEventSink()
    _set_workspace(_router(sink), ".jaato-eval-workspaces/arm0")

    assert sink.workspaces == {}, (
        "a relative workspace was registered for the client; every session "
        "it creates inherits it and resolves it against the daemon's cwd")
    assert sink.events and sink.events[0].error_type == "RelativePathAcrossBoundary"
    assert ".jaato-eval-workspaces/arm0" in sink.events[0].error


def test_set_workspace_with_an_absolute_path_is_registered() -> None:
    sink = _FakeEventSink()
    _set_workspace(_router(sink), "/abs/ws")

    assert sink.workspaces == {"client-1": "/abs/ws"}
    assert sink.events == []


# ----------------------------------------------------------------------
# 1. The sending half
# ----------------------------------------------------------------------


def test_the_sdk_refuses_to_put_a_relative_workspace_on_the_wire() -> None:
    """The client fails in its OWN process, where the cwd is knowable.

    This is the half that makes the daemon's refusal actionable: the
    traceback lands in the caller, next to the code that chose the path.
    """
    from jaato_sdk.client.ipc import IPCClient
    from jaato_sdk.events import ClientType

    with pytest.raises(RelativePathAcrossBoundaryError) as exc:
        IPCClient(client_type=ClientType.API,
                  workspace_path=".jaato-eval-workspaces/arm0")
    assert ".jaato-eval-workspaces/arm0" in str(exc.value)

    with pytest.raises(RelativePathAcrossBoundaryError):
        IPCClient(client_type=ClientType.API,
                  workspace_path="/abs/ws", config_root="task/.jaato")


def test_the_sdk_accepts_an_absolute_workspace() -> None:
    from jaato_sdk.client.ipc import IPCClient
    from jaato_sdk.events import ClientType

    client = IPCClient(client_type=ClientType.API,
                       workspace_path=os.sep + os.path.join("abs", "ws"))
    assert client.workspace_path.endswith("ws")
