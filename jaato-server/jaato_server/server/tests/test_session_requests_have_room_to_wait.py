"""Session requests run on an executor sized for blocking work.

Every IPC request but ``ClientConfigRequest`` (and the WS server's session
requests) runs on the loop's default executor, which Python sizes at
``cpu_count + 4``.  A ``session.new`` holds its thread for the whole
creation, mostly waiting on its runner, so 16 parallel creates on a 4-core
host queued up to 5 s behind 8 threads, and every other request queued
behind them.  See :mod:`jaato_server.server.daemon_executor`.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

from jaato_server.server.daemon_executor import (
    DAEMON_EXECUTOR_WORKERS, install_daemon_executor,
)
from jaato_server.shared.tests.reversion import Reversion

_MAIN = "jaato-server/jaato_server/server/__main__.py"

REVERSIONS = [
    Reversion(
        target=_MAIN,
        find="        install_daemon_executor(asyncio.get_running_loop())",
        replace="        pass",
        test="test_the_daemon_installs_it_when_its_loop_starts",
        because=("the daemon left on Python's default executor: a burst of "
                 "session.new queues behind cpu+4 threads"),
    ),
]


def test_the_executor_is_sized_for_blocking_work():
    async def run():
        loop = asyncio.get_running_loop()
        executor = install_daemon_executor(loop)
        assert executor._max_workers == DAEMON_EXECUTOR_WORKERS
        # The loop's default executor is the one installed: work handed
        # to run_in_executor(None, ...) runs on its threads.
        import threading
        name = await loop.run_in_executor(
            None, lambda: threading.current_thread().name)
        assert name.startswith("jaato-daemon-req")
    asyncio.run(run())


def test_installing_twice_keeps_the_first_executor():
    async def run():
        loop = asyncio.get_running_loop()
        first = install_daemon_executor(loop)
        assert install_daemon_executor(loop) is first
    asyncio.run(run())


def test_the_daemon_installs_it_when_its_loop_starts():
    """``JaatoDaemon.start`` calls the installer (call site, not behaviour)."""
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "__main__.py").read_text())
    starts = [n for n in ast.walk(tree)
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "start"]
    calls = {
        c.func.id for s in starts for c in ast.walk(s)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
    }
    assert "install_daemon_executor" in calls
