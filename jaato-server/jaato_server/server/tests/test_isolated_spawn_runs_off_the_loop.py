"""The isolated spawn runs off the daemon loop.

``SpawnIsolatedRunnerHandler.handle`` runs on the daemon loop, and the
spawn it calls waits on that loop (``rpc.start()``, then the bootstrap
RPC). Called in place it waited on itself until the 10 s timeout, so
every isolated subagent on the runner path failed at
``stage=forwarding`` with a bare ``TimeoutError``, on any host. Seen on
the SELinux phase 3 kernel run, once the parent-id fix let the spawn get
that far.
"""

import asyncio
from types import SimpleNamespace

from jaato_server.server.runner_rpc_handlers.spawn_isolated_runner import (
    SpawnIsolatedRunnerHandler,
)
from jaato_server.shared.tests.reversion import Reversion

REVERSIONS = [
    Reversion(
        target="jaato-server/jaato_server/server/runner_rpc_handlers/spawn_isolated_runner.py",
        find="""            return await asyncio.to_thread(
                self._session_manager._spawn_isolated_runner,
""",
        replace="""            return self._session_manager._spawn_isolated_runner(
""",
        test="test_a_spawn_that_waits_on_the_loop_completes",
        because="the spawn would wait on the loop it runs on and time out",
    ),
]


class _WaitsOnTheLoop:
    """Stands in for SessionManager: its spawn waits on the daemon loop,
    as ``_do_spawn_isolated_runner`` does with ``rpc.start()``."""

    def __init__(self, loop):
        self._loop = loop

    def _spawn_isolated_runner(self, **kwargs):
        async def started():
            return "started"
        fut = asyncio.run_coroutine_threadsafe(started(), self._loop)
        return {"ok": True, "waited": fut.result(timeout=2.0)}


def _args():
    return {"parent_session_id": "sess-A", "subagent_id": "a1",
            "profile_payload": {"name": "p", "description": "d", "model": "m",
                                "provider": "anthropic", "plugins": ["cli"],
                                "plugin_configs": {}},
            "task": "t", "workspace_path": "/w",
            "agent_params": {"isolated": True}}


def test_a_spawn_that_waits_on_the_loop_completes():
    async def run():
        handler = SpawnIsolatedRunnerHandler("sess-A")
        handler.set_spawn_dependencies(_WaitsOnTheLoop(asyncio.get_running_loop()))
        return await asyncio.wait_for(handler.handle(_args()), timeout=5.0)

    assert asyncio.run(run()) == {"ok": True, "waited": "started"}
