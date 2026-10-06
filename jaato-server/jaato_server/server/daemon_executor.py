"""The thread pool the daemon runs session requests on.

Every IPC request but ``ClientConfigRequest``, and the WS server's
session requests, run on the event loop's DEFAULT executor
(``loop.run_in_executor(None, ...)``) so a slow handler does not block
the loop.  Python sizes that executor at ``min(32, cpu_count + 4)``:
8 threads on a 4-core host.

That size suits CPU-bound work and is wrong for this one.  A
``session.new`` holds its thread for the whole creation, most of it
blocked waiting on its runner (spawn, ``session.bootstrap``, the
provider connect).  Measured with ``scripts/bench_cascade_fanout.py``
on a 4-core host, 16 parallel stages of one cascade:

| default executor | handler wait (max) | last stage open |
|---|---|---|
| 8 threads   | 5.2 s | 14.3 s |
| 64 threads  | 0.0 s |  7.8 s |

The queue also held every OTHER request on the daemon, so a burst of
creates delayed messages to sessions that were already open.

:data:`DAEMON_EXECUTOR_WORKERS` is a ceiling, not a cost: a
``ThreadPoolExecutor`` starts threads as work arrives, so an idle daemon
holds as many threads as it ever needed at once.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

#: Most threads the daemon's default executor may run at once.
DAEMON_EXECUTOR_WORKERS = 64

_MARK = "_jaato_daemon_executor"


def install_daemon_executor(loop: asyncio.AbstractEventLoop) -> ThreadPoolExecutor:
    """Make *loop*'s default executor one sized for blocking session work.

    Called where the daemon (and the standalone WS server) starts its
    loop, before any request is dispatched.  Idempotent per loop: the WS
    server's ``start`` also runs inside the daemon's loop, and replacing
    the executor a second time would leave requests already running on
    the first one with no owner.  Returns the loop's executor.
    """
    existing = getattr(loop, _MARK, None)
    if existing is not None:
        return existing
    executor = ThreadPoolExecutor(
        max_workers=DAEMON_EXECUTOR_WORKERS,
        thread_name_prefix="jaato-daemon-req",
    )
    loop.set_default_executor(executor)
    setattr(loop, _MARK, executor)
    return executor
