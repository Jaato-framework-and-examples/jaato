"""Observability for the three silent early exits of the WS AppArmor
pre-init hook (#1296).

``_apparmor_pre_init_hook`` (``websocket.py``, wired by
``JaatoWSServer.set_command_router``) is meant to confine EVERY
WS-provisioned session unconditionally.  Until #1296 it had four early
returns and only the last one (branch (d), "no daemon loop captured") ever
logged anything — the other three returned in total silence:

    (a) ``if not workspace_path: return``                    — silent
    (b) ``except OSError: return``  (realpath failed)         — silent
    (c) the workspace-not-under-root gate                     — silent
    (d) ``if daemon_loop is None: ... logger.warning(...)``   — ALREADY LOGS

#1293 (fixed by PR #1295, the branch this file's commit sits on) turned out
to be explainable in part because this hook silently declined for a
WS-provisioned session that should have been confined — and nothing in any
daemon log said WHICH of (a)/(b)/(c) it took, or why.  This file is
observability only: it does NOT change which branch fires when, or what
happens after it fires (every ``return`` is still exactly where it was).

**The volume constraint is the whole point of the design.**  This hook is
registered on the SessionManager's shared, transport-agnostic
``_pre_initialize_hooks`` list (see ``_run_pre_initialize_hooks``), so it
fires on every session bootstrap on the daemon — IPC and CLI sessions
included, not only WS ones.  Branch (c) is the ROUTINE exit for every
non-WS session, which on most deployments is most sessions; branch (a) is
just as routine on a deployment with IPC/embedded/API sessions that never
select a workspace at all, since the SAME hook answers for those too.
Making either of those branches log above DEBUG would spam the daemon log
on every ordinary session creation, which is precisely what this issue
warns against.  Branch (b) is different in kind: it fires only once a REAL
``workspace_path`` was handed to the hook and ``os.path.realpath`` (of
either that path or the WS server's own configured root) raised
``OSError`` — an anomaly, not a normal session shape — so it is logged at
WARNING, naming both paths so a broken symlink or a vanished root is
diagnosable from the log alone (the leading hypothesis in #1293's own
incident).  Branch (d) is untouched: it already logged correctly at
WARNING before this change, and this file does not touch its wording or
level.

**No kernel here.**  As with the #1293 file this branch already carries,
this container has no AppArmor LSM.  These tests drive the hook's real
closure (constructed the same way ``set_command_router`` constructs it)
against fabricated WS-server / server state, and assert on the resulting
log records and on which of ``resolve_plugin_apparmor_rules`` /
``spawn_session_runner`` / ``dispatch_bootstrap_envelope`` were (not)
reached — never on kernel behaviour.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional
from unittest.mock import patch

from jaato_server.shared.tests.reversion import Reversion

_WS = "jaato-server/jaato_server/server/websocket.py"
_LOGGER_NAME = "jaato_server.server.websocket"


# ----------------------------------------------------------------------
# Reversions — read by
# ``shared/tests/test_every_guard_detects_its_own_reversion.py``.  Every
# ``test`` names a test declared IN THIS MODULE (#1065).
# ----------------------------------------------------------------------
REVERSIONS = [
    # (a) — the no-workspace_path exit stops naming the session at DEBUG.
    Reversion(
        target=_WS,
        find=(
            "            if not workspace_path:\n"
            "                # #1296: this hook fires for EVERY session "
            "bootstrap on\n"
            "                # this daemon, not only WS-provisioned ones "
            "(it is\n"
            "                # registered on the shared, transport-agnostic\n"
            "                # ``_pre_initialize_hooks`` list), so a "
            "session created\n"
            "                # with no workspace_path at all — an "
            "IPC/embedded/API\n"
            "                # session that never selected one — reaches "
            "here\n"
            "                # routinely.  DEBUG, not WARNING: making "
            "this branch loud\n"
            "                # would reproduce exactly the log-spam this "
            "issue exists\n"
            "                # to avoid, one level up from branch (c) "
            "below.\n"
            "                logger.debug(\n"
            '                    "AppArmor pre-init: no workspace_path '
            'for session %s "\n'
            '                    "— nothing to confine, skipping",\n'
            "                    session_id,\n"
            "                )\n"
            "                return"
        ),
        replace=(
            "            if not workspace_path:\n"
            "                return"
        ),
        test="test_no_workspace_path_logs_session_id_at_debug",
        because=(
            "the no-workspace_path exit returns silently again, giving an "
            "operator no way to tell it took this branch rather than one "
            "of the other two silent exits"
        ),
    ),
    # (b) — the realpath-OSError exit stops naming both paths at WARNING.
    Reversion(
        target=_WS,
        find=(
            "            except OSError as exc:\n"
            "                # #1296: unlike the two branches around it, "
            "this one is\n"
            "                # NOT the routine \"this session isn't mine\" "
            "exit — it\n"
            "                # only fires once a real workspace_path was "
            "handed in and\n"
            "                # the kernel refused to resolve it (or the "
            "WS server's\n"
            "                # own configured root cannot be resolved), "
            "which is\n"
            "                # anomalous rather than a normal session "
            "shape.  WARNING,\n"
            "                # naming both paths so a broken symlink or a "
            "vanished\n"
            "                # root is diagnosable from the log alone.\n"
            "                logger.warning(\n"
            '                    "AppArmor pre-init: realpath failed for '
            'session %s "\n'
            '                    "(workspace_path=%r, ws_workspace_root=%r): '
            '%s: %s "\n'
            '                    "— skipping AppArmor confinement for this '
            'session",\n'
            "                    session_id, workspace_path, "
            "ws_server._workspace_root,\n"
            "                    type(exc).__name__, exc,\n"
            "                )\n"
            "                return"
        ),
        replace=(
            "            except OSError:\n"
            "                return"
        ),
        test="test_realpath_failure_logs_both_paths_at_warning",
        because=(
            "a realpath OSError (a broken symlink, a vanished workspace "
            "root) returns silently again, with no evidence in any log an "
            "operator can read about which path failed to resolve or why"
        ),
    ),
    # (c) — the not-under-WS-root exit stops naming both paths at DEBUG.
    Reversion(
        target=_WS,
        find=(
            "            ):\n"
            "                # #1296: the ROUTINE exit — every non-WS "
            "session\n"
            "                # (IPC / user-CWD) takes this branch, since "
            "the hook is\n"
            "                # registered for every session bootstrap "
            "regardless of\n"
            "                # transport.  DEBUG only, so it stays "
            "greppable on\n"
            "                # demand without adding one line per "
            "ordinary session\n"
            "                # creation at the daemon's default log "
            "level.\n"
            "                logger.debug(\n"
            '                    "AppArmor pre-init: session %s workspace '
            '%s is not "\n'
            '                    "under the WS server\'s workspace_root %s '
            '— IPC or "\n'
            '                    "user-CWD session, not WS-provisioned",\n'
            "                    session_id, sess_workspace, "
            "ws_workspace_root,\n"
            "                )\n"
            "                return  # IPC or user-CWD session — not "
            "WS-provisioned"
        ),
        replace=(
            "            ):\n"
            "                return  # IPC or user-CWD session — not "
            "WS-provisioned"
        ),
        test="test_not_under_ws_root_logs_both_paths_at_debug",
        because=(
            "the not-under-WS-root exit — the routine case for every "
            "IPC/user-CWD session — returns silently again"
        ),
    ),
]


# ----------------------------------------------------------------------
# Harness — a minimal, in-memory replica of what
# ``JaatoWSServer.set_command_router`` needs to construct and register
# the real ``_apparmor_pre_init_hook`` closure, mirroring the pattern in
# ``test_a_sessions_own_creation_confines_it_1293.py`` (kept local rather
# than imported, so this file's reversions and this file's tests are
# self-contained in one module, per the #1065 rule that a ``test`` names a
# test declared IN THIS module).
# ----------------------------------------------------------------------


class _FakeAppArmorForHook:
    """Never reached by any of (a)/(b)/(c) — present only so the hook's
    attribute lookups on ``ws_server._apparmor`` do not raise before the
    branch under test returns."""

    def is_available(self) -> bool:
        return True


class _FakeSMForHook:
    def __init__(self) -> None:
        self.pre_init_hooks: List[Any] = []

    def add_pre_initialize_hook(self, hook: Any) -> None:
        self.pre_init_hooks.append(hook)

    def add_session_hook(self, hook: Any) -> None:
        pass


class _FakeRouterForHook:
    def __init__(self) -> None:
        self._session_manager = _FakeSMForHook()


class _FakeEventSinkAdapterForHook:
    """The real shape (#1299): the daemon loop lives HERE, never as a
    bare ``_event_loop`` on the server itself."""

    def __init__(self, loop: Any) -> None:
        self._event_loop = loop


def _make_ws_server_for_hook(workspace_root: str):
    from jaato_server.server.websocket import JaatoWSServer

    ws = JaatoWSServer.__new__(JaatoWSServer)
    ws._apparmor = _FakeAppArmorForHook()
    ws._cgroups = None
    ws._workspace_root = workspace_root
    # #1299: the loop lives on _event_sink_adapter, not as a bare
    # attribute on the server — see _FakeEventSinkAdapterForHook above.
    ws._event_sink_adapter = _FakeEventSinkAdapterForHook("<loop>")
    return ws


def _register_pre_init_hook(ws: Any) -> Callable[..., None]:
    from jaato_server.server.websocket import JaatoWSServer

    router = _FakeRouterForHook()
    JaatoWSServer.set_command_router(ws, router)
    assert router._session_manager.pre_init_hooks, "hook not registered"
    return router._session_manager.pre_init_hooks[0]


def _fake_server() -> Any:
    server = type("_S", (), {})()
    server.config_root = None  # type: ignore[attr-defined]
    server.note_runner_bootstrap_outcome = (  # type: ignore[attr-defined]
        lambda *a, **k: None
    )
    return server


def _refuse_if_reached(*_a: Any, **_k: Any) -> None:
    """Stand-in for every call past the branch under test.  If control
    flow reaches here, the early ``return`` did NOT fire — the assertion
    this drives is that the branch's control flow (its early exit) is
    unchanged, not only that it logs."""
    raise AssertionError(
        "reached past the early-exit branch under test — the hook's "
        "control flow changed"
    )


def _patched_downstream():
    """Patches every call site a correctly-early-returning branch must
    never reach: profile-rule resolution, runner spawn, bootstrap
    dispatch."""
    return (
        patch(
            "jaato_server.server.apparmor.resolve_plugin_apparmor_rules",
            _refuse_if_reached,
        ),
        patch(
            "jaato_server.server.runner_spawn.spawn_session_runner",
            _refuse_if_reached,
        ),
        patch(
            "jaato_server.server.runner_spawn.dispatch_bootstrap_envelope",
            _refuse_if_reached,
        ),
    )


# ----------------------------------------------------------------------
# (a) — no workspace_path at all.
# ----------------------------------------------------------------------


def test_no_workspace_path_logs_session_id_at_debug(tmp_path, caplog) -> None:
    """Branch (a) names the session at DEBUG and returns without touching
    anything downstream — never louder, since this is the routine shape
    for every session (of any transport) created with no workspace."""
    p1, p2, p3 = _patched_downstream()
    with p1, p2, p3:
        ws = _make_ws_server_for_hook(str(tmp_path))
        hook = _register_pre_init_hook(ws)

        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            result = hook(_fake_server(), "sess-no-ws", None, client_id=None)

    assert result is None, "the hook must return (not raise) on this branch"
    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("sess-no-ws" in r.getMessage() for r in debug_records), (
        "the no-workspace_path branch must log the session id at DEBUG"
    )
    assert not any(r.levelno > logging.DEBUG for r in caplog.records), (
        "the routine no-workspace_path exit must never log above DEBUG"
    )


# ----------------------------------------------------------------------
# (b) — realpath() raised OSError.
# ----------------------------------------------------------------------


def test_realpath_failure_logs_both_paths_at_warning(
    tmp_path, caplog, monkeypatch,
) -> None:
    """Branch (b) is the one genuine anomaly of the three: a real
    ``workspace_path`` was supplied and ``os.path.realpath`` could not
    resolve it (or the WS server's own root).  It logs at WARNING, naming
    both paths so a broken symlink or a vanished root is diagnosable from
    the log alone — the leading hypothesis in #1293's own incident."""

    real_realpath = __import__("os").path.realpath

    def _raising_realpath(path, *a: Any, **k: Any):
        if path == "workspace-that-cannot-resolve":
            raise OSError("simulated resolution failure")
        return real_realpath(path, *a, **k)

    monkeypatch.setattr("os.path.realpath", _raising_realpath)

    p1, p2, p3 = _patched_downstream()
    with p1, p2, p3:
        ws = _make_ws_server_for_hook(str(tmp_path))
        hook = _register_pre_init_hook(ws)

        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            result = hook(
                _fake_server(), "sess-bad-path",
                "workspace-that-cannot-resolve", client_id=None,
            )

    assert result is None, "the hook must return (not raise) on this branch"
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning_records, "the realpath-OSError branch must log at WARNING"
    message = warning_records[0].getMessage()
    assert "sess-bad-path" in message
    assert "workspace-that-cannot-resolve" in message
    assert str(tmp_path) in message, (
        "the WARNING must name the WS server's own workspace_root too, "
        "not only the session's workspace_path"
    )


# ----------------------------------------------------------------------
# (c) — workspace_path resolves, but not under the WS server's root.
# ----------------------------------------------------------------------


def test_not_under_ws_root_logs_both_paths_at_debug(
    tmp_path, caplog,
) -> None:
    """Branch (c) is the routine exit for every IPC/user-CWD session:
    logged at DEBUG (the volume constraint this issue is about), and
    still names both paths so a trailing-slash or symlink mismatch is
    diagnosable on demand rather than invisible."""
    ws_root = tmp_path / "ws-root"
    ws_root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    p1, p2, p3 = _patched_downstream()
    with p1, p2, p3:
        ws = _make_ws_server_for_hook(str(ws_root))
        hook = _register_pre_init_hook(ws)

        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            result = hook(
                _fake_server(), "sess-elsewhere", str(elsewhere),
                client_id=None,
            )

    assert result is None, "the hook must return (not raise) on this branch"
    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert debug_records, "the not-under-WS-root branch must log at DEBUG"
    message = debug_records[0].getMessage()
    assert "sess-elsewhere" in message
    assert str(elsewhere.resolve()) in message
    assert str(ws_root.resolve()) in message
    assert not any(r.levelno > logging.DEBUG for r in caplog.records), (
        "the routine not-under-WS-root exit must never log above DEBUG"
    )


# ----------------------------------------------------------------------
# Branch (d) is untouched — confirmed, not merely assumed, because this
# file's whole premise is that the OTHER three branches used to be silent
# the same way (d) is not.
# ----------------------------------------------------------------------


def test_no_daemon_loop_still_logs_at_warning_unchanged(
    tmp_path, caplog,
) -> None:
    """Branch (d) already logged correctly before #1296 and must keep
    doing so, unedited — the one branch this issue says to leave alone."""
    ws = _make_ws_server_for_hook(str(tmp_path))
    ws._event_sink_adapter._event_loop = None  # the (d) condition
    hook = _register_pre_init_hook(ws)

    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        result = hook(
            _fake_server(), "sess-no-loop", str(tmp_path), client_id=None,
        )

    assert result is None
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning_records, "branch (d) must still log at WARNING"
    assert "sess-no-loop" in warning_records[0].getMessage()
