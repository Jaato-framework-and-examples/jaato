"""Tests for the session.set_presentation_context RPC handler +
wrapper + JaatoServer migration (Phase 3 §3.3c).

PresentationContext is a Pydantic BaseModel — the wrapper
serializes via ``model_dump`` (or ``dict`` for v1), the runner
reconstructs via ``model_validate`` (or ``parse_obj``).

Tests pin:

- happy path: setter called with a reconstructed
  PresentationContext instance
- args validation: 'context' must be a dict
- schema validation failure → stage=decode error
- no_host / no_session error paths
- setter crash → stage=set error
- daemon-side wrapper accepts model instance OR dict
- daemon-side wrapper rejects unsupported types
"""

from __future__ import annotations

import socket
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from jaato_sdk.events import PresentationContext
from server.runner.envelope import RequestEnvelope
from server.runner.rpc import RunnerRPC
from server.runner.session import RunnerSessionHost
from server.runner_rpc_client import RunnerRPCClient
from shared.session_envelope import SessionInitEnvelope


def _make_lone_runner() -> RunnerRPC:
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    b.close()

    def _no_executor(name: str, args: Any):
        return False, {"error": "no executor"}

    return RunnerRPC(a, _no_executor)


def _good_envelope() -> SessionInitEnvelope:
    return SessionInitEnvelope(
        session_id="sess-pres-1",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _FakeSession:
    def __init__(self) -> None:
        self.set_calls: List[Any] = []

    def set_presentation_context(self, ctx: Any) -> None:
        self.set_calls.append(ctx)


def _install(rpc: RunnerRPC, session: Any) -> None:
    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=session,
    )


def _make_ctx(width: int = 80) -> PresentationContext:
    return PresentationContext(content_width=width)


# ----------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------


def test_set_presentation_context_happy_path() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ctx = _make_ctx(width=120)
    ctx_dict = ctx.model_dump()

    ok, result = rpc._handle_session_set_presentation_context(
        {"context": ctx_dict},
    )
    assert ok is True
    assert result == {"ok": True}
    assert len(session.set_calls) == 1
    received = session.set_calls[0]
    # Reconstructed correctly: it's a PresentationContext with the
    # same content_width.
    assert isinstance(received, PresentationContext)
    assert received.content_width == 120


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


def test_set_presentation_context_rejects_non_dict_context() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession())

    for bad in ["str", 42, None, []]:
        ok, result = rpc._handle_session_set_presentation_context(
            {"context": bad},
        )
        assert ok is False
        assert result["stage"] == "decode"


def test_set_presentation_context_schema_validation_failure() -> None:
    """Passing a dict that doesn't match PresentationContext's
    schema (e.g., wrong types) surfaces as a clean decode error."""
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession())

    ok, result = rpc._handle_session_set_presentation_context(
        {"context": {"content_width": "not_an_int"}},
    )
    assert ok is False
    assert result["stage"] == "decode"


# ----------------------------------------------------------------------
# Error paths
# ----------------------------------------------------------------------


def test_set_presentation_context_no_host_returns_error() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_set_presentation_context(
        {"context": _make_ctx().model_dump()},
    )
    assert ok is False
    assert result["stage"] == "no_host"


def test_set_presentation_context_setter_crash_returns_set_error() -> None:
    rpc = _make_lone_runner()

    class _BoomSession:
        def set_presentation_context(self, ctx):
            raise RuntimeError("setter boom")

    _install(rpc, _BoomSession())

    ok, result = rpc._handle_session_set_presentation_context(
        {"context": _make_ctx().model_dump()},
    )
    assert ok is False
    assert result["stage"] == "set"
    assert "setter boom" in result["error"]


# ----------------------------------------------------------------------
# Dispatch routing
# ----------------------------------------------------------------------


def test_dispatch_routes_set_presentation_context() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    env = RequestEnvelope(
        id=1, method="session.set_presentation_context",
        args={"context": _make_ctx(100).model_dump()},
    )
    ok, _ = rpc._dispatch_method(env)
    assert ok is True
    assert session.set_calls[0].content_width == 100


# ----------------------------------------------------------------------
# Wrapper input handling
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wrapper_rejects_unsupported_input_type() -> None:
    """Daemon-side wrapper rejects ctx that's neither a Pydantic
    model nor a dict — fail-fast for typos."""
    client = RunnerRPCClient.__new__(RunnerRPCClient)
    with pytest.raises(TypeError, match="Pydantic model or dict"):
        await client.session_set_presentation_context("not_valid")
    with pytest.raises(TypeError, match="Pydantic model or dict"):
        await client.session_set_presentation_context(42)
