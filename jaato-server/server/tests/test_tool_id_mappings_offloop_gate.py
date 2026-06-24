"""_build_tool_id_mappings gates the session_get_tool_schemas_threadsafe RPC off-loop.

Re-attach re-entrancy regression: that RPC is a run_coroutine_threadsafe +
future.result(); invoked on the event-loop thread it self-deadlocks (the loop
can't pump the coro it's blocked on) -> 15s timeout — the same class as the
original register-RPC stall.  So it must run ONLY off-loop; on-loop emits
(emit_current_state / initialize / _register_client_tools) map daemon-tier names
only, and the off-loop re-emits (runner-ready + client-tool _push) add runner-tier.
"""
import asyncio

from server.core import JaatoServer


def _server_with_rpc():
    srv = JaatoServer.__new__(JaatoServer)        # skip heavy __init__
    calls = {"rpc": 0}

    class _RPC:
        def session_get_tool_schemas_threadsafe(self):
            calls["rpc"] += 1
            return []

    srv._runner_rpc = _RPC()
    srv.registry = None   # skip the registry walk; isolate the RPC gate
    return srv, calls


def test_rpc_skipped_on_loop_thread():
    srv, calls = _server_with_rpc()

    async def _on_loop():
        srv._build_tool_id_mappings()

    asyncio.run(_on_loop())                       # runs on a live event loop
    assert calls["rpc"] == 0   # on the loop -> RPC gated (would self-deadlock)


def test_rpc_called_off_loop():
    srv, calls = _server_with_rpc()
    srv._build_tool_id_mappings()                 # no running loop
    assert calls["rpc"] == 1   # off-loop -> RPC invoked for runner-tier names
