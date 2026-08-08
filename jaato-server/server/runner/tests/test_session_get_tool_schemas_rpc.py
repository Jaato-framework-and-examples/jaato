"""Tests for the session.get_tool_schemas RPC handler +
daemon-side wrapper (Phase 3 §7c step 6.6.4.5c.5).

Final handler in Path D's 5-handler decomposition.  Reads the
runner-side session's resolved tool schemas (preloaded plugins +
on-demand activations).  Replaces 2 daemon-side reaches:

- ``core.py:1407`` (tool-ID registry build) — reads ``.name`` +
  ``.category``
- ``core.py:3759`` (``signal_completion_in_surface`` filter at the
  completion-nudge guard) — reads ``.name`` via ``getattr``

Wire shape per the §7c step 6.6.4.5c.5 audit decision: **dict-
shape-only** (Path A).  Pre-implementation grep verified all 7
``ToolSchema`` fields + nested ``EditableContent`` fields are
JSON-encodable; ``traits: FrozenSet[str]`` round-trips as
``traits: List[str]`` on the wire and back to ``FrozenSet`` on
receipt.

Critical migration note: this RPC is needed because
``JaatoRuntime.get_tool_schemas()`` returns the registry's full
set, NOT the session-resolved subset that ``JaatoSession`` returns.
Refinement 2's daemon-side ``_runtime`` cache was rejected during
5b's in-flight audit for this exact reason — over-including
filtered-out tools would break the ``signal_completion_in_surface``
check (interactive sessions filter out signal_completion).

Tests pin (~14-17 per the audit's expected count):

- happy paths: empty list / single tool / multi-tool
- field round-trip: name / description / parameters /
  category / discoverability / editable / traits
- FrozenSet ↔ list conversion preserves trait membership
- EditableContent nested dataclass round-trip
- Reconstruction equality: ToolSchema attr access works unchanged
- Empty input_schema / empty traits / no editable edge cases
- no_host / no_session / missing_method / call error paths
- Dispatch routing
- E2E via async + threadsafe wrappers
- Integration pin 1: name-iteration pattern (core.py:1407)
- Integration pin 2: signal_completion_in_surface pattern (core.py:3759)
- Structural pin on real JaatoSession.get_tool_schemas
"""

from __future__ import annotations

import asyncio
import socket
import threading
from typing import Any, Dict, List, Optional

import pytest

from jaato_sdk.plugins.model_provider.types import (
    EditableContent,
    ToolSchema,
)
from server.runner.envelope import RequestEnvelope
from server.runner.rpc import RunnerRPC
from server.runner.session import RunnerSessionHost
from server.runner_rpc_client import RunnerRPCClient, RunnerCallError
from shared.session_envelope import SessionInitEnvelope


def _make_lone_runner() -> RunnerRPC:
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    b.close()

    def _no_executor(name: str, args: Any):
        return False, {"error": "no executor"}

    return RunnerRPC(a, _no_executor)


def _good_envelope() -> SessionInitEnvelope:
    return SessionInitEnvelope(
        session_id="sess-ts-1",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _FakeSession:
    def __init__(
        self,
        schemas: Optional[List[ToolSchema]] = None,
        raise_on_call: Optional[Exception] = None,
    ) -> None:
        self._schemas = schemas or []
        self._raise = raise_on_call
        self.calls = 0

    def get_tool_schemas(self) -> List[ToolSchema]:
        self.calls += 1
        if self._raise is not None:
            raise self._raise
        return list(self._schemas)


class _SessionWithoutMethod:
    pass


def _install(rpc: RunnerRPC, session: Any) -> None:
    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=session,
    )


# ----------------------------------------------------------------------
# Direct handler tests
# ----------------------------------------------------------------------


def test_get_tool_schemas_empty_list() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession(schemas=[]))

    ok, result = rpc._handle_session_get_tool_schemas()
    assert ok is True
    assert result == {"schemas": []}


def test_get_tool_schemas_single_minimal_tool() -> None:
    """Tool with only required fields (name + description) +
    framework defaults."""
    rpc = _make_lone_runner()
    schema = ToolSchema(name="foo", description="A foo tool")
    _install(rpc, _FakeSession(schemas=[schema]))

    ok, result = rpc._handle_session_get_tool_schemas()
    assert ok is True
    assert result["schemas"] == [{
        "name": "foo",
        "description": "A foo tool",
        "parameters": {},
        "category": None,
        "discoverability": "discoverable",
        "editable": None,
        "traits": [],
    }]


def test_get_tool_schemas_all_fields_populated() -> None:
    """Full-field tool: parameters / category / discoverability /
    editable / traits all set."""
    rpc = _make_lone_runner()
    schema = ToolSchema(
        name="write_file",
        description="Write file",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
        },
        category="filesystem",
        discoverability="core",
        editable=EditableContent(
            parameters=["path"], format="yaml",
            template="# Edit this file path",
        ),
        traits=frozenset({"file_writer", "permission_required"}),
    )
    _install(rpc, _FakeSession(schemas=[schema]))

    ok, result = rpc._handle_session_get_tool_schemas()
    assert ok is True
    serialized = result["schemas"][0]
    assert serialized["name"] == "write_file"
    assert serialized["category"] == "filesystem"
    assert serialized["discoverability"] == "core"
    assert serialized["parameters"] == {
        "type": "object",
        "properties": {"path": {"type": "string"}},
    }
    assert serialized["editable"] == {
        "parameters": ["path"],
        "format": "yaml",
        "template": "# Edit this file path",
    }
    # FrozenSet → list on the wire — order isn't guaranteed.
    assert set(serialized["traits"]) == {"file_writer", "permission_required"}


def test_get_tool_schemas_frozenset_traits_converted_to_list() -> None:
    """traits FrozenSet[str] must convert to list (JSON has no set)."""
    rpc = _make_lone_runner()
    schema = ToolSchema(
        name="x", description="y",
        traits=frozenset({"a", "b", "c"}),
    )
    _install(rpc, _FakeSession(schemas=[schema]))

    ok, result = rpc._handle_session_get_tool_schemas()
    assert ok is True
    traits = result["schemas"][0]["traits"]
    assert isinstance(traits, list)
    assert set(traits) == {"a", "b", "c"}


def test_get_tool_schemas_empty_traits_default() -> None:
    """Default empty FrozenSet round-trips as empty list."""
    rpc = _make_lone_runner()
    schema = ToolSchema(name="x", description="y")
    _install(rpc, _FakeSession(schemas=[schema]))

    ok, result = rpc._handle_session_get_tool_schemas()
    assert ok is True
    assert result["schemas"][0]["traits"] == []


def test_get_tool_schemas_no_editable_default() -> None:
    """Default None editable round-trips as None."""
    rpc = _make_lone_runner()
    schema = ToolSchema(name="x", description="y")
    _install(rpc, _FakeSession(schemas=[schema]))

    ok, result = rpc._handle_session_get_tool_schemas()
    assert ok is True
    assert result["schemas"][0]["editable"] is None


def test_get_tool_schemas_multi_tool() -> None:
    """List with several tools — all serialize independently."""
    rpc = _make_lone_runner()
    schemas = [
        ToolSchema(name="a", description="A", category="cat1"),
        ToolSchema(name="b", description="B", category="cat2"),
        ToolSchema(name="c", description="C"),
    ]
    _install(rpc, _FakeSession(schemas=schemas))

    ok, result = rpc._handle_session_get_tool_schemas()
    assert ok is True
    assert len(result["schemas"]) == 3
    assert [s["name"] for s in result["schemas"]] == ["a", "b", "c"]


def test_get_tool_schemas_no_host_returns_error() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_get_tool_schemas()
    assert ok is False
    assert result["stage"] == "no_host"


def test_get_tool_schemas_missing_method_returns_error() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _SessionWithoutMethod())

    ok, result = rpc._handle_session_get_tool_schemas()
    assert ok is False
    assert result["stage"] == "missing_method"


def test_get_tool_schemas_method_raises_returns_call_error() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession(raise_on_call=RuntimeError("registry boom")))

    ok, result = rpc._handle_session_get_tool_schemas()
    assert ok is False
    assert result["stage"] == "call"
    assert "registry boom" in result["error"]


def test_dispatch_routes_session_get_tool_schemas() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession(schemas=[
        ToolSchema(name="x", description="y"),
    ]))

    env = RequestEnvelope(
        id=1, method="session.get_tool_schemas", args={},
    )
    ok, result = rpc._dispatch_method(env)
    assert ok is True
    assert result["schemas"][0]["name"] == "x"


# ----------------------------------------------------------------------
# E2E: full reconstruction round-trip
# ----------------------------------------------------------------------


class _Pair:
    def __init__(self, runner_rpc, runner_thread, daemon_client) -> None:
        self.runner_rpc = runner_rpc
        self.runner_thread = runner_thread
        self.daemon_client = daemon_client


async def _make_pair() -> _Pair:
    daemon_sock, runner_sock = socket.socketpair(
        socket.AF_UNIX, socket.SOCK_STREAM,
    )

    def _no_executor(name, args):
        return False, {"error": "no executor"}

    runner_rpc = RunnerRPC(runner_sock, _no_executor)
    thread = threading.Thread(
        target=runner_rpc.serve, name="rpc-ts-test", daemon=True,
    )
    thread.start()

    daemon_client = RunnerRPCClient(daemon_sock, runner_pid=0)
    await daemon_client.start()
    return _Pair(runner_rpc, thread, daemon_client)


async def _teardown(pair: _Pair) -> None:
    pair.runner_rpc.shutdown()
    pair.runner_thread.join(timeout=2)
    pair.daemon_client._closed = True
    if pair.daemon_client._writer is not None:
        try:
            pair.daemon_client._writer.close()
            await pair.daemon_client._writer.wait_closed()
        except Exception:
            pass
    if pair.daemon_client._read_task is not None:
        pair.daemon_client._read_task.cancel()
        try:
            await pair.daemon_client._read_task
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_e2e_full_field_round_trip_preserves_equality() -> None:
    """All-field tool round-trips to a reconstructed ToolSchema
    that's equal to the original (per the audit's equality pin)."""
    pair = await _make_pair()
    try:
        original = ToolSchema(
            name="write_file",
            description="Write a file",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            category="filesystem",
            discoverability="core",
            editable=EditableContent(
                parameters=["path", "content"],
                format="yaml",
                template="# Edit",
            ),
            traits=frozenset({"file_writer", "permission_required"}),
        )
        _install(pair.runner_rpc, _FakeSession(schemas=[original]))

        result = await pair.daemon_client.session_get_tool_schemas()
        assert len(result) == 1
        reconstructed = result[0]
        assert isinstance(reconstructed, ToolSchema)
        # Dataclass equality covers all fields.
        assert reconstructed == original
        # Type pins.
        assert isinstance(reconstructed.traits, frozenset)
        assert isinstance(reconstructed.editable, EditableContent)
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_iteration_pattern_works_unchanged() -> None:
    """Integration pin: the daemon-side core.py:1407 site iterates
    schemas reading ``.name`` and ``.category`` — wrapper
    reconstruction must preserve that attr access."""
    pair = await _make_pair()
    try:
        schemas = [
            ToolSchema(name="read_file", description="r",
                       category="filesystem"),
            ToolSchema(name="search", description="s",
                       category="search"),
            ToolSchema(name="run_tests", description="t"),  # no category
        ]
        _install(pair.runner_rpc, _FakeSession(schemas=schemas))

        result = await pair.daemon_client.session_get_tool_schemas()
        # Mirror the daemon-side core.py:1407 iteration pattern.
        names = [s.name for s in result]
        categories = [s.category for s in result if s.category]
        assert names == ["read_file", "search", "run_tests"]
        assert categories == ["filesystem", "search"]
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_signal_completion_filter_works_unchanged() -> None:
    """Integration pin: the daemon-side core.py:3759 site does
    ``any(getattr(t, 'name', None) == 'signal_completion' for t in ...)``
    — wrapper reconstruction must support getattr-based access."""
    pair = await _make_pair()
    try:
        with_signal = [
            ToolSchema(name="foo", description="x"),
            ToolSchema(name="signal_completion", description="end"),
        ]
        without_signal = [
            ToolSchema(name="foo", description="x"),
            ToolSchema(name="bar", description="y"),
        ]

        # Case 1: signal_completion in surface → filter detects it.
        _install(pair.runner_rpc, _FakeSession(schemas=with_signal))
        result1 = await pair.daemon_client.session_get_tool_schemas()
        assert any(
            getattr(t, "name", None) == "signal_completion" for t in result1
        ) is True

        # Case 2: signal_completion NOT in surface → filter returns False.
        _install(pair.runner_rpc, _FakeSession(schemas=without_signal))
        result2 = await pair.daemon_client.session_get_tool_schemas()
        assert any(
            getattr(t, "name", None) == "signal_completion" for t in result2
        ) is False
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_threadsafe_from_thread() -> None:
    pair = await _make_pair()
    try:
        schemas = [
            ToolSchema(name="x", description="y",
                       traits=frozenset({"t1"})),
        ]
        _install(pair.runner_rpc, _FakeSession(schemas=schemas))

        result_holder: Dict[str, Any] = {}

        def _worker() -> None:
            try:
                result_holder["value"] = (
                    pair.daemon_client
                    .session_get_tool_schemas_threadsafe()
                )
            except Exception as exc:  # noqa: BLE001
                result_holder["error"] = exc

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        for _ in range(50):
            if "value" in result_holder or "error" in result_holder:
                break
            await asyncio.sleep(0.05)
        t.join(timeout=2)

        assert "error" not in result_holder, result_holder.get("error")
        result = result_holder["value"]
        assert len(result) == 1
        assert isinstance(result[0], ToolSchema)
        assert result[0].traits == frozenset({"t1"})
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_missing_method_raises() -> None:
    pair = await _make_pair()
    try:
        _install(pair.runner_rpc, _SessionWithoutMethod())
        with pytest.raises(RunnerCallError) as exc_info:
            await pair.daemon_client.session_get_tool_schemas()
        assert "get_tool_schemas" in str(exc_info.value)
    finally:
        await _teardown(pair)


# ----------------------------------------------------------------------
# Real JaatoSession: structural pin
# ----------------------------------------------------------------------


def test_jaato_session_has_public_get_tool_schemas_method() -> None:
    from shared.jaato_session import JaatoSession
    assert callable(getattr(JaatoSession, "get_tool_schemas", None))
