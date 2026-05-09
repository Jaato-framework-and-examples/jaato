"""Tests for the session.get_history + session.get_context_usage
RPC handlers (Phase 3 §3.3c precursors).

Both are read-only probes the daemon will dispatch to once the
seat-flip migrates ``client.get_history()`` /
``client.get_context_usage()`` through the runner-RPC surface.

Tests pin:

- get_history serializes each Message via to_dict() and returns
  ``{"history": [<dict>, ...]}``.
- get_history accepts ``raw=True`` to return the un-transformed
  view via get_history_raw().
- per-message serialization failures don't drop the whole history
  (placeholder substituted; count preserved).
- get_context_usage returns the session's usage dict wrapped in
  ``{"usage": <dict>}``.
- both handlers return clean ``no_host`` / ``no_session`` errors
  when not bootstrapped.
- get_context_usage rejects non-dict returns (defensive against
  custom session subclasses).
"""

from __future__ import annotations

import socket
from typing import Any, Dict, List, Optional

import pytest

from server.runner.envelope import RequestEnvelope
from server.runner.rpc import RunnerRPC
from server.runner.session import RunnerSessionHost
from shared.session_envelope import SessionInitEnvelope


def _make_lone_runner() -> RunnerRPC:
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    b.close()

    def _no_executor(name: str, args: Any):
        return False, {"error": "no executor"}

    return RunnerRPC(a, _no_executor)


def _good_envelope() -> SessionInitEnvelope:
    return SessionInitEnvelope(
        session_id="sess-h-1",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _FakeMessage:
    """Stand-in for shared.plugins.model_provider.types.Message
    that exposes to_dict() the way the real message dataclass does."""

    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}


class _FakeSession:
    def __init__(self) -> None:
        self.transformed_history: List[Any] = []
        self.raw_history: List[Any] = []
        self.usage: Dict[str, Any] = {
            "model": "test-model",
            "context_limit": 100000,
            "total_tokens": 1234,
            "prompt_tokens": 1234,
            "output_tokens": 0,
            "turns": 5,
            "percent_used": 1.234,
            "tokens_remaining": 98766,
        }
        self.usage_raises: Optional[Exception] = None
        self.history_raises: Optional[Exception] = None

    def get_history(self) -> List[Any]:
        if self.history_raises:
            raise self.history_raises
        return self.transformed_history

    def get_history_raw(self) -> List[Any]:
        if self.history_raises:
            raise self.history_raises
        return self.raw_history

    def get_context_usage(self) -> Dict[str, Any]:
        if self.usage_raises:
            raise self.usage_raises
        return self.usage


def _install(rpc: RunnerRPC, session: Any) -> None:
    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=session,
    )


# ----------------------------------------------------------------------
# get_history — happy path
# ----------------------------------------------------------------------


def test_get_history_returns_serialized_messages() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    session.transformed_history = [
        _FakeMessage("user", "hello"),
        _FakeMessage("assistant", "hi"),
    ]
    _install(rpc, session)

    ok, result = rpc._handle_session_get_history({})
    assert ok is True
    assert result == {
        "history": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
    }


def test_get_history_raw_uses_raw_view() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    session.transformed_history = [_FakeMessage("user", "transformed")]
    session.raw_history = [_FakeMessage("user", "raw")]
    _install(rpc, session)

    ok, result = rpc._handle_session_get_history({"raw": True})
    assert ok is True
    assert result["history"][0]["content"] == "raw"


def test_get_history_empty_returns_empty_list() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession())

    ok, result = rpc._handle_session_get_history({})
    assert ok is True
    assert result == {"history": []}


def test_get_history_per_message_failure_substitutes_placeholder() -> None:
    """A single message's to_dict() crash must not drop the
    whole history — the count stays accurate and the placeholder
    flags the bug so daemon-side logging can pick it up."""
    rpc = _make_lone_runner()

    class _BadMessage:
        def to_dict(self):
            raise RuntimeError("serialization boom")

    session = _FakeSession()
    session.transformed_history = [
        _FakeMessage("user", "good"),
        _BadMessage(),
        _FakeMessage("user", "also good"),
    ]
    _install(rpc, session)

    ok, result = rpc._handle_session_get_history({})
    assert ok is True
    assert len(result["history"]) == 3
    assert result["history"][0]["content"] == "good"
    assert result["history"][1] == {"role": "system", "content": "<unserialisable>"}
    assert result["history"][2]["content"] == "also good"


# ----------------------------------------------------------------------
# get_history — error paths
# ----------------------------------------------------------------------


def test_get_history_no_host_returns_error() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_get_history({})
    assert ok is False
    assert result["stage"] == "no_host"


def test_get_history_no_session_returns_error() -> None:
    rpc = _make_lone_runner()
    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=None,
    )
    ok, result = rpc._handle_session_get_history({})
    assert ok is False
    assert result["stage"] == "no_session"


def test_get_history_session_raises_returns_read_error() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    session.history_raises = RuntimeError("history boom")
    _install(rpc, session)

    ok, result = rpc._handle_session_get_history({})
    assert ok is False
    assert result["stage"] == "read"
    assert "history boom" in result["error"]


# ----------------------------------------------------------------------
# get_context_usage
# ----------------------------------------------------------------------


def test_get_context_usage_returns_session_dict() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ok, result = rpc._handle_session_get_context_usage()
    assert ok is True
    assert result == {"usage": session.usage}


def test_get_context_usage_returns_copy_not_reference() -> None:
    """The returned dict must be a copy — daemon-side mutation
    shouldn't propagate back into the session's internal state."""
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ok, result = rpc._handle_session_get_context_usage()
    assert ok is True
    result["usage"]["total_tokens"] = 999999
    assert session.usage["total_tokens"] == 1234  # unchanged


def test_get_context_usage_no_host_returns_error() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_get_context_usage()
    assert ok is False
    assert result["stage"] == "no_host"


def test_get_context_usage_session_raises_returns_read_error() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    session.usage_raises = ValueError("usage boom")
    _install(rpc, session)

    ok, result = rpc._handle_session_get_context_usage()
    assert ok is False
    assert result["stage"] == "read"
    assert "usage boom" in result["error"]


def test_get_context_usage_non_dict_return_returns_read_error() -> None:
    """Defensive: a custom session subclass returning a non-dict
    surfaces as a clean error rather than letting the wire
    encoder choke later."""
    rpc = _make_lone_runner()

    class _BadSession:
        def get_context_usage(self):
            return ["not", "a", "dict"]

    _install(rpc, _BadSession())

    ok, result = rpc._handle_session_get_context_usage()
    assert ok is False
    assert result["stage"] == "read"
    assert "expected dict" in result["error"]


# ----------------------------------------------------------------------
# Dispatch routing
# ----------------------------------------------------------------------


def test_dispatch_routes_get_history() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    session.transformed_history = [_FakeMessage("user", "x")]
    _install(rpc, session)

    env = RequestEnvelope(id=1, method="session.get_history", args={})
    ok, result = rpc._dispatch_method(env)
    assert ok is True
    assert result["history"][0]["content"] == "x"


def test_dispatch_routes_get_context_usage() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession())

    env = RequestEnvelope(
        id=2, method="session.get_context_usage", args={},
    )
    ok, result = rpc._dispatch_method(env)
    assert ok is True
    assert "usage" in result


# ----------------------------------------------------------------------
# Real Message serialization (regression — handler must not require
# a to_dict method on every message; dataclasses.asdict + enum
# coercion is the post-§3.3c-precursor fallback path)
# ----------------------------------------------------------------------


def test_get_history_serializes_real_message_dataclasses() -> None:
    """Production Message instances (from
    jaato_sdk.plugins.model_provider.types) don't define
    to_dict() — they're plain dataclasses with Role-enum role
    fields.  The handler must serialize them via
    ``dataclasses.asdict`` + enum coercion so the JSON encoder
    doesn't choke on the wire."""
    from jaato_sdk.plugins.model_provider.types import Message
    import json

    from jaato_sdk.plugins.model_provider.types import Role
    rpc = _make_lone_runner()
    session = _FakeSession()
    session.transformed_history = [
        Message.from_text(Role.USER, "hello"),
        Message.from_text(Role.MODEL, "hi back"),
    ]
    _install(rpc, session)

    ok, result = rpc._handle_session_get_history({})
    assert ok is True
    history = result["history"]
    assert len(history) == 2
    # Each entry must round-trip cleanly through json.dumps
    # (no Enum / dataclass leaks).
    for entry in history:
        encoded = json.dumps(entry)
        decoded = json.loads(encoded)
        assert decoded["role"] in ("user", "model")
        assert isinstance(decoded["parts"], list)
        # parts contain text Part with the original content.
        assert decoded["parts"][0]["text"] in ("hello", "hi back")


def test_serialize_helper_coerces_enums_recursively() -> None:
    """The _coerce_for_json helper handles nested enums / dicts /
    lists / tuples — covers the full conversation Part/Role tree."""
    import enum
    from server.runner.rpc import _coerce_for_json

    class _Color(enum.Enum):
        RED = "red"
        BLUE = "blue"

    payload = {
        "color": _Color.RED,
        "items": [_Color.BLUE, {"nested": _Color.RED}],
        "tup": (_Color.RED, _Color.BLUE),
    }
    out = _coerce_for_json(payload)
    assert out == {
        "color": "red",
        "items": ["blue", {"nested": "red"}],
        "tup": ["red", "blue"],
    }
