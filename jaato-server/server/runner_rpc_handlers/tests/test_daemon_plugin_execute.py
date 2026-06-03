"""Tests for :mod:`server.runner_rpc_handlers.daemon_plugin_execute`.

Mirror of ``test_spawn_isolated_runner`` for the cross-tier reverse-
dispatch handler.  Covers:

- Args validation (missing keys, wrong types) raises ``ValueError`` at
  the dispatcher boundary.
- Plugin / tool lookup failure raises ``ValueError`` with useful
  diagnostic.
- Successful dispatch runs the executor via run_in_executor (worker
  thread) so blocking bodies don't stall the asyncio loop.
- Shutdown is idempotent; post-shutdown calls raise ``RuntimeError``.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from server.runner_rpc_handlers.daemon_plugin_execute import (
    DaemonPluginExecuteHandler,
    register,
)


# ----------------------------------------------------------------------
# Test fakes
# ----------------------------------------------------------------------


class _FakePlugin:
    """Stand-in plugin with a registered executor."""

    def __init__(self, name: str = "fake_plugin") -> None:
        self.name = name
        self.executor_thread_id: int = -1
        self.executor_args: Dict[str, Any] = {}

    def _execute_demo(self, args: Dict[str, Any]) -> Dict[str, Any]:
        # Record the thread we ran on — used to assert the handler
        # ran us off the asyncio loop.
        self.executor_thread_id = threading.get_ident()
        self.executor_args = dict(args)
        return {"answer": "ok", "echo": args}

    def get_executors(self) -> Dict[str, Any]:
        return {"demo_tool": self._execute_demo}


class _FakeRegistry:
    """Stand-in registry holding a plugin by name."""

    def __init__(self, plugins: Dict[str, Any]) -> None:
        self._plugins = plugins

    def get_plugin(self, name: str) -> Any:
        return self._plugins.get(name)


class _FakeServer:
    def __init__(self, registry: Any) -> None:
        self.registry = registry


def _make_handler(plugins: Dict[str, Any] = None) -> DaemonPluginExecuteHandler:
    server = _FakeServer(
        registry=_FakeRegistry(plugins or {"fake_plugin": _FakePlugin()}),
    )
    return DaemonPluginExecuteHandler(server=server)


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


def test_construct_requires_non_none_server() -> None:
    with pytest.raises(ValueError):
        DaemonPluginExecuteHandler(server=None)


# ----------------------------------------------------------------------
# Args validation
# ----------------------------------------------------------------------


@pytest.mark.parametrize("missing_key", ["plugin_name", "tool_name", "args"])
def test_missing_required_key_raises(missing_key: str) -> None:
    handler = _make_handler()
    args = {
        "plugin_name": "fake_plugin",
        "tool_name": "demo_tool",
        "args": {},
    }
    del args[missing_key]
    with pytest.raises(ValueError, match=missing_key):
        asyncio.run(handler.handle(args))


@pytest.mark.parametrize("bad_value", [None, "", 42, [], {}])
def test_invalid_plugin_name_raises(bad_value: Any) -> None:
    handler = _make_handler()
    with pytest.raises(ValueError, match="plugin_name"):
        asyncio.run(handler.handle({
            "plugin_name": bad_value,
            "tool_name": "demo_tool",
            "args": {},
        }))


@pytest.mark.parametrize("bad_value", [None, "", 42, [], {}])
def test_invalid_tool_name_raises(bad_value: Any) -> None:
    handler = _make_handler()
    with pytest.raises(ValueError, match="tool_name"):
        asyncio.run(handler.handle({
            "plugin_name": "fake_plugin",
            "tool_name": bad_value,
            "args": {},
        }))


@pytest.mark.parametrize("bad_value", [None, "", 42, ["a", "b"]])
def test_invalid_args_type_raises(bad_value: Any) -> None:
    handler = _make_handler()
    with pytest.raises(ValueError, match="args"):
        asyncio.run(handler.handle({
            "plugin_name": "fake_plugin",
            "tool_name": "demo_tool",
            "args": bad_value,
        }))


# ----------------------------------------------------------------------
# Lookup failures
# ----------------------------------------------------------------------


def test_missing_registry_raises() -> None:
    server = _FakeServer(registry=None)
    handler = DaemonPluginExecuteHandler(server=server)
    with pytest.raises(ValueError, match="no registry"):
        asyncio.run(handler.handle({
            "plugin_name": "fake_plugin",
            "tool_name": "demo_tool",
            "args": {},
        }))


def test_unknown_plugin_raises() -> None:
    handler = _make_handler()
    with pytest.raises(ValueError, match="not found"):
        asyncio.run(handler.handle({
            "plugin_name": "ghost_plugin",
            "tool_name": "demo_tool",
            "args": {},
        }))


def test_unknown_tool_raises() -> None:
    handler = _make_handler()
    with pytest.raises(ValueError, match="not found in plugin"):
        asyncio.run(handler.handle({
            "plugin_name": "fake_plugin",
            "tool_name": "ghost_tool",
            "args": {},
        }))


# ----------------------------------------------------------------------
# Success path
# ----------------------------------------------------------------------


def test_dispatches_executor_with_args() -> None:
    plugin = _FakePlugin()
    handler = _make_handler({"fake_plugin": plugin})

    result = asyncio.run(handler.handle({
        "plugin_name": "fake_plugin",
        "tool_name": "demo_tool",
        "args": {"q": "hello"},
    }))

    assert result == {"answer": "ok", "echo": {"q": "hello"}}
    assert plugin.executor_args == {"q": "hello"}


def test_executor_runs_off_event_loop_thread() -> None:
    """The handler uses ``run_in_executor`` so blocking bodies don't
    stall the asyncio loop — verify the executor ran on a thread
    different from the loop's calling thread."""
    plugin = _FakePlugin()
    handler = _make_handler({"fake_plugin": plugin})

    async def _drive() -> None:
        loop_thread = threading.get_ident()
        await handler.handle({
            "plugin_name": "fake_plugin",
            "tool_name": "demo_tool",
            "args": {},
        })
        assert plugin.executor_thread_id != loop_thread, (
            f"Executor ran on loop thread {loop_thread} — "
            f"would stall every other RPC on the channel."
        )

    asyncio.run(_drive())


def test_executor_exception_propagates_to_dispatcher() -> None:
    """When the in-process executor raises, the exception propagates
    out — the dispatcher (one level up) catches and translates to a
    typed error envelope."""

    class _Plugin:
        name = "boom"

        def get_executors(self):
            def _bomb(_args):
                raise RuntimeError("kaboom")
            return {"bad_tool": _bomb}

    handler = _make_handler({"boom": _Plugin()})

    with pytest.raises(RuntimeError, match="kaboom"):
        asyncio.run(handler.handle({
            "plugin_name": "boom",
            "tool_name": "bad_tool",
            "args": {},
        }))


# ----------------------------------------------------------------------
# Shutdown
# ----------------------------------------------------------------------


def test_shutdown_marks_closed() -> None:
    handler = _make_handler()
    handler.shutdown()
    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(handler.handle({
            "plugin_name": "fake_plugin",
            "tool_name": "demo_tool",
            "args": {},
        }))


def test_shutdown_is_idempotent() -> None:
    handler = _make_handler()
    handler.shutdown()
    handler.shutdown()  # no-op, no raise


# ----------------------------------------------------------------------
# Register helper
# ----------------------------------------------------------------------


def test_register_binds_handler_to_rpc_server() -> None:
    """``register`` wires the handler's ``handle`` method to the
    ``daemon.plugin_execute`` method on the rpc server."""
    rpc_server = MagicMock()
    handler = _make_handler()
    register(rpc_server, handler)

    rpc_server.register.assert_called_once_with(
        "daemon.plugin_execute", handler.handle,
    )
