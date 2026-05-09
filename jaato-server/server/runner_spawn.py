"""Per-session runner subprocess spawn helper (Phase 3 §3.12).

Extracted from ``server/__main__.py:_spawn_session_runner`` (Phase 2
task 2.3) so both the IPC apparmor pre-init hook and the WS-server
apparmor pre-init hook can spawn runners through the same code
path.

Lifecycle:

1. Caller (an apparmor pre-init hook) finishes
   ``apparmor.provision_profile`` for the session.
2. Caller invokes :func:`spawn_session_runner`.
3. Function spawns the runner via :class:`RunnerSpawner`, opens a
   :class:`RunnerRPCClient` against the parent end of the
   socketpair, starts the read-loop on the daemon's asyncio loop,
   and attaches the RPC handle onto the JaatoServer via
   ``server.set_runner_rpc(rpc, spawned)``.
4. After this returns, plugins discovered during
   ``server.initialize()`` see ``registry.runner_rpc`` set at
   configure time.

Failures raise; the caller catches and downgrades the session to
``sandbox_mode = "soft"`` per the §4.6 fallback contract.

Phase 3 §3.3c part 2's optional ``JAATO_RUNNER_HOSTS_SESSION``
bootstrap-envelope dispatch stays in ``server/__main__.py``'s
wrapper — that's part of the seat-flip bring-up window, not the
shared spawn primitive.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional


logger = logging.getLogger(__name__)


def spawn_session_runner(
    *,
    server: Any,  # JaatoServer (forward-typed; importing the real
                  # type creates a cycle through server/core.py).
    session_id: str,
    workspace_path: str,
    profile_name: str,
    daemon_loop: asyncio.AbstractEventLoop,
) -> None:
    """Spawn the per-session runner subprocess and wire its RPC handle
    onto the JaatoServer.

    Args:
        server: The session's :class:`JaatoServer` instance.
        session_id: Session identifier (passed via env to the runner).
        workspace_path: Session workspace; used both as the runner's
            cwd and as the prefix for the per-session log file path
            (plan §5.1).
        profile_name: AppArmor profile name (already loaded in the
            kernel).
        daemon_loop: The daemon's main asyncio loop — needed to run
            ``RunnerRPCClient.start()`` since it's async.

    Raises:
        RuntimeError: when *daemon_loop* is None or the runner-RPC
            start times out.  Caller catches and downgrades to
            ``sandbox_mode = "soft"``.
        Exception: any spawn / RPC failure.  Caller catches and
            downgrades.
    """
    from server.runner_spawner import RunnerSpawner
    from server.runner_rpc_client import RunnerRPCClient

    if daemon_loop is None:
        raise RuntimeError(
            "spawn_session_runner: daemon loop unavailable; cannot "
            "start RunnerRPCClient"
        )

    spawner = RunnerSpawner()

    log_path: Optional[str] = None
    if workspace_path:
        log_dir = os.path.join(workspace_path, ".jaato", "logs")
        log_path = os.path.join(log_dir, f"runner-{session_id}.log")

    spawned = spawner.spawn(
        profile_name=profile_name,
        session_id=session_id,
        workspace_path=workspace_path,
        log_path=log_path,
    )

    rpc = RunnerRPCClient(
        spawned.parent_socket,
        runner_pid=spawned.pid,
        loop=daemon_loop,
    )

    fut = asyncio.run_coroutine_threadsafe(rpc.start(), daemon_loop)
    fut.result(timeout=10.0)

    server.set_runner_rpc(rpc, spawned)
    logger.info(
        "runner spawned for session %s: pid=%d profile=%s log=%s",
        session_id, spawned.pid, profile_name, log_path or "(inherited)",
    )
