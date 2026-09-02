"""``session.get_rendered_system_instruction`` — the prompt, as rendered.

Issue #787.  The daemon persists this so a revive can RESTORE the system
instruction rather than rebuild it; rebuilding re-runs the persona's
``{{!py:...}}`` prefetch scripts, which is what made a session with a
mandatory prefetch impossible to wake.

The distinction the handler must preserve is between the SNAPSHOT (what
``configure()`` produced) and the LIVE attribute (which keeps growing as
plugins inject deferred instructions and references are pinned).  Restoring
the live value would re-add content the revived session re-produces for
itself — once per revive, cumulatively — so the snapshot is what travels.
"""

from __future__ import annotations

import socket
from typing import Any, Optional

from server.runner.rpc import RunnerRPC
from server.runner.session import RunnerSessionHost
from shared.session_envelope import SessionInitEnvelope


def _make_lone_runner() -> RunnerRPC:
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    b.close()

    def _no_executor(name: str, args: Any):
        return False, {"error": "no executor"}

    return RunnerRPC(a, _no_executor)


def _envelope() -> SessionInitEnvelope:
    return SessionInitEnvelope(
        session_id="sess-rendered-prompt",
        workspace_path="/tmp/ws",
        profile_name="worker",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _FakeSession:
    """A session whose live prompt has drifted past its rendered one."""

    def __init__(self, rendered: Optional[str] = "AS RENDERED",
                 raises: Optional[Exception] = None):
        self._rendered = rendered
        self._raises = raises

    def get_rendered_system_instruction(self) -> Optional[str]:
        if self._raises is not None:
            raise self._raises
        return self._rendered

    def get_system_instruction(self) -> Optional[str]:
        return "AS RENDERED\n\n<deferred plugin instructions injected later>"


class _LegacySession:
    """A session object predating the accessor (mixed-build stubs)."""


def _install(rpc: RunnerRPC, session: Any) -> None:
    rpc._session_host = RunnerSessionHost(
        envelope=_envelope(), runtime=None, session=session,
    )


def test_it_returns_the_snapshot_not_the_live_prompt():
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession())

    ok, result = rpc._handle_session_get_rendered_system_instruction()

    assert ok is True
    assert result == {"rendered_system_instruction": "AS RENDERED"}, (
        "the handler returned the live prompt; restoring that on revive "
        "re-adds the runtime injections the revived session re-produces "
        "for itself, doubling them once per revive"
    )


def test_none_is_an_answer_not_an_error():
    """A session that has not been configured has nothing rendered.

    The daemon persists nothing in that case and the revive falls back to
    re-rendering — the pre-#787 behaviour — so this must not surface as a
    failure that a caller has to special-case.
    """
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession(rendered=None))

    ok, result = rpc._handle_session_get_rendered_system_instruction()

    assert ok is True
    assert result == {"rendered_system_instruction": None}


def test_a_session_without_the_accessor_reports_nothing_rendered():
    rpc = _make_lone_runner()
    _install(rpc, _LegacySession())

    ok, result = rpc._handle_session_get_rendered_system_instruction()

    assert ok is True
    assert result == {"rendered_system_instruction": None}


def test_a_raising_accessor_is_reported_as_a_read_failure():
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession(raises=RuntimeError("boom")))

    ok, result = rpc._handle_session_get_rendered_system_instruction()

    assert ok is False
    assert result["stage"] == "read"
    assert "boom" in result["error"]


def test_a_non_string_render_is_refused():
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession(rendered={"not": "a string"}))

    ok, result = rpc._handle_session_get_rendered_system_instruction()

    assert ok is False
    assert result["stage"] == "read"


def test_no_session_host_is_a_typed_error_not_a_crash():
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_get_rendered_system_instruction()
    assert ok is False
    assert "error" in result
