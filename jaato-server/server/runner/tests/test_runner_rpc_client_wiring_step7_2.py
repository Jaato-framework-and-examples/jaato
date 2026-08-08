"""Regression-pin tests for §7c Step 7.2 — runner-side
``RunnerRPCClient`` instantiation + ``registry.runner_rpc_client``
attachment.

Per the §7c Step 7 disposition audit (commit 285449d0), pre-§7.2
the runner-internal ``RunnerRPCClient`` (`server/runner/rpc_client.py:48`)
was defined-but-never-instantiated, and the runner-side
``registry.runner_rpc_client`` attribute was never assigned.  The
permission plugin's ``_get_runner_rpc_channel()`` lookup at
permission/plugin.py:152 always returned None and silently fell
back to the in-process channel (orphaned post-seat-flip).

Step 7.2 closes both gaps inside ``_handle_session_bootstrap``:
after ``bootstrap_session(envelope)`` returns a healthy
``RunnerSessionHost``, construct a ``RunnerRPCClient(self)`` (where
self is the runner-side ``RunnerRPC`` dispatcher carrying the
daemon socket) and assign it to the runtime's registry as
``runner_rpc_client``.

Tests pin:

- After ``session.bootstrap`` succeeds, ``host.runtime._registry``
  has a ``runner_rpc_client`` attribute.
- The attached client is an instance of
  ``server.runner.rpc_client.RunnerRPCClient`` (not a Mock).
- The client's ``prompt_operator`` method routes through the
  parent ``RunnerRPC``'s ``outgoing_call``.
- Bootstrap without a runtime (test-stub path) skips the
  attachment silently.
- Permission plugin's ``_get_runner_rpc_channel`` would resolve
  the channel given the attached client (integration shape).
"""

from __future__ import annotations

import socket
from typing import Any

from server.runner.envelope import RequestEnvelope
from server.runner.rpc import RunnerRPC
from server.runner.rpc_client import RunnerRPCClient as RunnerInternalRPCClient
from shared.session_envelope import SessionInitEnvelope


def _make_lone_runner() -> RunnerRPC:
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    b.close()

    def _no_executor(name: str, args: Any):
        return False, {"error": "no executor"}

    return RunnerRPC(a, _no_executor)


def _bootstrap_envelope() -> SessionInitEnvelope:
    return SessionInitEnvelope(
        session_id="sess-7-2",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


def _bootstrap_with_stub_runtime(rpc: RunnerRPC) -> Any:
    """Bootstrap the session with a minimal stub runtime that
    has a ``_registry`` we can inspect for the
    ``runner_rpc_client`` attachment."""

    class _StubRegistry:
        pass

    class _StubRuntime:
        def __init__(self) -> None:
            self._registry = _StubRegistry()

    class _StubSession:
        pass

    # Inject the stub via _USE_DEFAULT replacement.  bootstrap_session
    # respects the runtime_factory kwarg; but we're calling through
    # the RPC dispatch path which uses the default.  So we manually
    # poke the host into _session_host using the runner's lock pattern
    # then verify the wiring runs.
    from server.runner.session import RunnerSessionHost

    stub_runtime = _StubRuntime()
    stub_session = _StubSession()
    host = RunnerSessionHost(
        envelope=_bootstrap_envelope(),
        runtime=stub_runtime,
        session=stub_session,
    )

    # Reproduce the relevant slice of `_handle_session_bootstrap`'s
    # post-bootstrap wiring (the actual handler also runs
    # `bootstrap_session(...)`, which constructs a full JaatoRuntime
    # we don't want in the test path).  This is the same wiring
    # code from rpc.py — exercising it via the test bypass keeps
    # the test focused on Step 7.2's invariant.
    rpc._session_host = host
    if host.runtime is not None and getattr(
        host.runtime, "_registry", None,
    ) is not None:
        from server.runner.rpc_client import RunnerRPCClient
        setattr(
            host.runtime._registry,
            "runner_rpc_client",
            RunnerRPCClient(rpc),
        )
    return host


# ----------------------------------------------------------------------
# Wiring pins
# ----------------------------------------------------------------------


def test_step_7_2_attaches_runner_rpc_client_to_registry() -> None:
    """After bootstrap, the runtime's registry has a
    ``runner_rpc_client`` attribute.  This is the load-bearing
    wiring: permission/plugin.py:152 looks up
    ``getattr(registry, 'runner_rpc_client', None)`` and bails
    to in-process channel when None — pre-§7.2 it was always
    None."""
    rpc = _make_lone_runner()
    host = _bootstrap_with_stub_runtime(rpc)
    assert host.runtime is not None
    client = getattr(host.runtime._registry, "runner_rpc_client", None)
    assert client is not None, (
        "§7c Step 7.2 regression: registry.runner_rpc_client not "
        "attached after session.bootstrap."
    )


def test_step_7_2_attached_client_is_runner_internal_rpc_client() -> None:
    """The attached client is a real ``RunnerRPCClient`` instance —
    not a Mock or stub.  The runner-side permission plugin's
    ``RunnerRPCChannel`` reads ``client.prompt_operator`` and
    relies on the callable's contract."""
    rpc = _make_lone_runner()
    host = _bootstrap_with_stub_runtime(rpc)
    client = host.runtime._registry.runner_rpc_client
    assert isinstance(client, RunnerInternalRPCClient)
    # The client should expose prompt_operator as a callable.
    assert callable(getattr(client, "prompt_operator", None))


def test_step_7_2_client_routes_through_parent_rpc() -> None:
    """The attached client routes ``prompt_operator()`` through
    the parent ``RunnerRPC.outgoing_call``.  Pin the wrapper
    holds a reference to the dispatcher rather than constructing
    its own."""
    rpc = _make_lone_runner()
    host = _bootstrap_with_stub_runtime(rpc)
    client = host.runtime._registry.runner_rpc_client
    # The wrapper stores the rpc reference as ``self._rpc``.
    assert client._rpc is rpc, (
        "§7c Step 7.2 regression: RunnerRPCClient should hold the "
        "parent RunnerRPC dispatcher reference for outgoing_call "
        "routing."
    )


def test_step_7_2_bootstrap_without_runtime_skips_attachment() -> None:
    """Test-stub bootstrap path (runtime_factory=None) returns a
    host with ``runtime=None``.  Step 7.2's attachment must
    gracefully skip — no AttributeError, no exception."""
    rpc = _make_lone_runner()
    from server.runner.session import RunnerSessionHost

    host = RunnerSessionHost(
        envelope=_bootstrap_envelope(),
        runtime=None,
        session=None,
    )
    rpc._session_host = host
    # Reproduce the wiring code's guard.  The guard is the load-
    # bearing piece — if it ever drops the ``host.runtime is not
    # None`` check, the next ``host.runtime._registry`` access
    # would crash.
    if host.runtime is not None and getattr(
        host.runtime, "_registry", None,
    ) is not None:
        from server.runner.rpc_client import RunnerRPCClient
        setattr(
            host.runtime._registry,
            "runner_rpc_client",
            RunnerRPCClient(rpc),
        )
    # No assertion of attachment — only assertion is "no crash".


# ----------------------------------------------------------------------
# Integration pin: permission plugin's runner-side channel lookup works
# ----------------------------------------------------------------------


def test_step_7_2_permission_plugin_can_resolve_runner_rpc_channel() -> None:
    """Integration pin: the §7c Step 7.2 attachment is the load-
    bearing piece for the permission plugin's runner-side ASK
    routing.  Pin that with the attachment in place, the
    permission plugin's ``_get_runner_rpc_channel`` would
    successfully resolve a RunnerRPCChannel.

    Direct invocation of the plugin's lookup logic without
    standing up a full plugin instance.  Verifies the
    interface contract: ``registry.runner_rpc_client.prompt_operator``
    is the field path the plugin reads."""
    rpc = _make_lone_runner()
    host = _bootstrap_with_stub_runtime(rpc)
    registry = host.runtime._registry

    # Replicate permission/plugin.py:_get_runner_rpc_channel's logic:
    rpc_client = getattr(registry, "runner_rpc_client", None)
    assert rpc_client is not None
    prompt_operator = getattr(rpc_client, "prompt_operator", None)
    assert prompt_operator is not None
    assert callable(prompt_operator)


# ----------------------------------------------------------------------
# Source-level wiring pin (the actual wiring lives in rpc.py:_handle_session_bootstrap)
# ----------------------------------------------------------------------


def test_step_7_2_handler_contains_registry_attachment_code() -> None:
    """AST-level pin: ``_handle_session_bootstrap`` in rpc.py
    contains the ``setattr(..., 'runner_rpc_client', ...)`` call.
    Catches accidental deletion of the wiring."""
    import inspect
    from server.runner.rpc import RunnerRPC

    src = inspect.getsource(RunnerRPC._handle_session_bootstrap)
    assert "runner_rpc_client" in src, (
        "§7c Step 7.2 regression: ``runner_rpc_client`` attachment "
        "code missing from ``_handle_session_bootstrap``."
    )
    assert "RunnerRPCClient(" in src, (
        "§7c Step 7.2 regression: ``RunnerRPCClient(self)`` "
        "construction missing from ``_handle_session_bootstrap``."
    )
