"""Pool PR 5a-fix: ``session.bootstrap`` dispatches on the main thread.

PR #94 (PR 5a) added ``_maybe_self_confine`` to ``bootstrap_session``
step 1c, calling ``aa_change_profile`` from inside the bootstrap
flow.  Under the previous dispatch shape, that flow ran in a
worker thread (``RunnerRPC.serve`` does ``self._pool.submit(...)``).

Linux apparmor's ``aa_change_profile`` is per-calling-thread: only
the worker thread that called it would have been confined.  The
main thread + subsequently-spawned workers would inherit the main
thread's unconfined cred via pthread_create.  v67 cascade smoke
caught this empirically (worker confined, verification via
``/proc/self/attr/current`` read the main thread's profile and
raised ``ConfinementMismatchError``).

This module pins the fix: ``RunnerRPC.serve`` special-cases
``session.bootstrap`` requests and handles them synchronously on
the main thread (NOT via ``self._pool.submit``).  Subsequent RPCs
(tool.execute, session.send_message, etc.) continue to dispatch
via the worker pool — those workers spawn AFTER bootstrap and
inherit the main thread's (now-confined) cred.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any, Dict, List

from server.runner.rpc import RunnerRPC
from shared.framing import read_frame_sync, write_frame_sync


def _capture_thread_id_runner() -> tuple[RunnerRPC, socket.socket, List[int]]:
    """Build a RunnerRPC whose handler records the thread ID it ran on.

    Returns:
        (runner_rpc, daemon_side_socket, captured_thread_ids).
        Each handled request appends ``threading.get_native_id()`` to
        the list, so the test can verify whether bootstrap ran on the
        main thread vs a worker.
    """
    daemon_sock, runner_sock = socket.socketpair(
        socket.AF_UNIX, socket.SOCK_STREAM,
    )
    captured: List[int] = []

    def _record_thread_executor(name: str, args: Dict[str, Any]):
        captured.append(threading.get_native_id())
        return True, {"name": name, "thread_id": threading.get_native_id()}

    runner_rpc = RunnerRPC(runner_sock, _record_thread_executor)
    return runner_rpc, daemon_sock, captured


def _send_request(
    daemon_sock: socket.socket,
    method: str,
    id: int = 1,
    args: Dict[str, Any] | None = None,
) -> None:
    """Write a request frame to the daemon-side socket."""
    payload = json.dumps({
        "id": id,
        "kind": "request",
        "method": method,
        "args": args if args is not None else {},
    })
    write_frame_sync(daemon_sock, payload)


def _read_response(daemon_sock: socket.socket, timeout: float = 3.0) -> Dict[str, Any]:
    """Read one response frame from the daemon-side socket."""
    daemon_sock.settimeout(timeout)
    try:
        raw = read_frame_sync(daemon_sock)
    finally:
        daemon_sock.settimeout(None)
    assert raw is not None, "runner closed without responding"
    return json.loads(raw)


def test_session_bootstrap_runs_on_main_thread() -> None:
    """``session.bootstrap`` is dispatched synchronously on the
    serve-loop thread (main thread), not submitted to the worker
    pool.

    The verification: the executor's recorded thread_id matches the
    serve thread's native id.  If bootstrap dispatched via
    ``self._pool.submit``, the recorded id would belong to a worker
    thread.
    """
    runner_rpc, daemon_sock, captured = _capture_thread_id_runner()

    serve_thread_id_holder: List[int] = []

    def _serve_with_tid_capture() -> None:
        serve_thread_id_holder.append(threading.get_native_id())
        runner_rpc.serve()

    serve_thread = threading.Thread(
        target=_serve_with_tid_capture,
        name="rpc-serve-test",
        daemon=True,
    )
    serve_thread.start()

    # Wait for the serve loop to record its own tid.
    for _ in range(50):
        if serve_thread_id_holder:
            break
        time.sleep(0.02)
    assert serve_thread_id_holder, "serve thread didn't start in time"
    serve_tid = serve_thread_id_holder[0]

    try:
        # Send a session.bootstrap request — the runner-side handler
        # for session.bootstrap is special and doesn't invoke the
        # executor we wired (it goes through _handle_session_bootstrap
        # directly, validates the envelope, and returns an error
        # because the envelope is empty).  We don't care about the
        # outcome; we only care THAT the dispatch happened on the
        # main thread.
        #
        # Capture the thread id by intercepting _handle_session_bootstrap.
        original_handler = runner_rpc._handle_session_bootstrap
        invocation_tids: List[int] = []

        def _wrapped_handler(args: Dict[str, Any]):
            invocation_tids.append(threading.get_native_id())
            return original_handler(args)

        runner_rpc._handle_session_bootstrap = _wrapped_handler  # type: ignore

        _send_request(daemon_sock, "session.bootstrap", id=1)
        # Read the response so the serve loop has finished processing.
        _read_response(daemon_sock)

        assert len(invocation_tids) == 1, (
            f"expected session.bootstrap to be handled exactly once, "
            f"observed {len(invocation_tids)} handler invocations"
        )
        assert invocation_tids[0] == serve_tid, (
            f"session.bootstrap ran on thread {invocation_tids[0]} but "
            f"serve loop is on thread {serve_tid} — should be the same "
            f"for the main-thread dispatch path"
        )
    finally:
        daemon_sock.close()
        runner_rpc.shutdown()
        serve_thread.join(timeout=2)


def test_non_bootstrap_request_runs_on_worker_thread() -> None:
    """Sanity check the OTHER side of the dispatch: a non-bootstrap
    request (e.g. ``tool.execute``) goes through ``self._pool.submit``,
    so its handler runs on a worker thread distinct from the serve
    thread."""
    runner_rpc, daemon_sock, captured = _capture_thread_id_runner()

    serve_thread_id_holder: List[int] = []

    def _serve_with_tid_capture() -> None:
        serve_thread_id_holder.append(threading.get_native_id())
        runner_rpc.serve()

    serve_thread = threading.Thread(
        target=_serve_with_tid_capture,
        name="rpc-serve-test",
        daemon=True,
    )
    serve_thread.start()

    for _ in range(50):
        if serve_thread_id_holder:
            break
        time.sleep(0.02)
    assert serve_thread_id_holder, "serve thread didn't start in time"
    serve_tid = serve_thread_id_holder[0]

    try:
        _send_request(
            daemon_sock,
            "tool.execute",
            id=1,
            args={"name": "some_tool", "args": {}},
        )
        _read_response(daemon_sock)

        assert len(captured) == 1, (
            f"expected exactly one executor invocation; got {len(captured)}"
        )
        assert captured[0] != serve_tid, (
            f"tool.execute handler ran on thread {captured[0]} which "
            f"matches the serve thread {serve_tid} — dispatch should have "
            f"submitted to the worker pool, spawning a distinct thread"
        )
    finally:
        daemon_sock.close()
        runner_rpc.shutdown()
        serve_thread.join(timeout=2)


def test_first_session_bootstrap_runs_before_any_worker_thread_spawns() -> None:
    """The key invariant for AppArmor self-confine: at the moment
    ``session.bootstrap`` is dispatched, NO worker threads have
    spawned yet.  That way ``aa_change_profile`` on the main thread
    means subsequently-spawned workers inherit the confined cred
    via pthread_create — same as cold-spawn's ``__main__.py`` step
    2 ordering.

    Verification: ``self._pool._threads`` (the ThreadPoolExecutor's
    internal worker set) is empty when the handler runs.
    """
    runner_rpc, daemon_sock, captured = _capture_thread_id_runner()

    pool_thread_count_at_dispatch: List[int] = []

    original_handler = runner_rpc._handle_session_bootstrap

    def _wrapped_handler(args: Dict[str, Any]):
        # _pool._threads is a set in ThreadPoolExecutor.
        pool_thread_count_at_dispatch.append(len(runner_rpc._pool._threads))
        return original_handler(args)

    runner_rpc._handle_session_bootstrap = _wrapped_handler  # type: ignore

    serve_thread = threading.Thread(
        target=runner_rpc.serve, name="rpc-serve-test", daemon=True,
    )
    serve_thread.start()

    try:
        _send_request(daemon_sock, "session.bootstrap", id=1)
        _read_response(daemon_sock)

        assert pool_thread_count_at_dispatch == [0], (
            f"At the moment session.bootstrap was handled, the worker "
            f"pool already had {pool_thread_count_at_dispatch[0]} thread(s). "
            f"This means aa_change_profile from inside bootstrap would "
            f"NOT propagate to those pre-existing workers — they'd remain "
            f"unconfined.  Bootstrap must be the first dispatch, BEFORE "
            f"any pool.submit() spawns a worker."
        )
    finally:
        daemon_sock.close()
        runner_rpc.shutdown()
        serve_thread.join(timeout=2)
