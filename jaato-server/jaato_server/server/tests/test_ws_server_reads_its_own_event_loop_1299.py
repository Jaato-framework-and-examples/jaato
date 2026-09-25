"""``JaatoWSServer`` has no ``_event_loop`` of its own — two call sites read
it directly anyway (#1299), and both must instead read
``self._event_sink_adapter._event_loop``, the object that actually holds it
(``WSEventSinkAdapter.bind_loop()``, called from ``start()``).

Live evidence (#1299): every WS session's ``_apparmor_pre_init_hook`` crashed
with ``AttributeError: 'JaatoWSServer' object has no attribute
'_event_loop'``, caught per-hook by ``_run_pre_initialize_hooks`` and logged
as a WARNING rather than aborting session creation — so AppArmor provisioning
and the runner spawn (both further down the same hook, past this line) never
ran, and a WS session ended up serving work fully in-process, unconfined,
while a SEPARATE post-init hook (``_apparmor_session_hook``) still loaded a
real AppArmor profile into the kernel for a runner that never existed. This
was invisible until yesterday (#1295) closed the double-spawn bug (#1293)
that had been providing an (unconfined) fallback runner via a different
code path — once that duplicate spawn correctly stopped running for WS
sessions, this pre-existing defect became the only thing standing between a
WS session and a runner, and it always crashed.

**Why the #1293 test file's own WS-hook fixture could not have caught
this.** ``test_a_sessions_own_creation_confines_it_1293.py``'s
``_make_ws_server_for_hook`` builds its fake server with
``ws._event_loop = "<loop>"`` — set DIRECTLY on the ``JaatoWSServer``
double, an attribute the real class never has. That fixture shape silently
matches the bug rather than the real object, which is exactly how it went
undetected: a test fixture that invents an attribute production code is
wrong to read proves nothing about whether production code reads it
correctly. This file's fixtures never set a bare ``_event_loop`` on the
server double; the loop lives only on a fake ``_event_sink_adapter``, the
shape the real ``JaatoWSServer`` actually has.
"""

from __future__ import annotations

from typing import Any, List, Optional
from unittest.mock import patch

import pytest

from jaato_server.shared.tests.reversion import Reversion

_WS = "jaato-server/jaato_server/server/websocket.py"


# ----------------------------------------------------------------------
# Reversions — read by
# ``shared/tests/test_every_guard_detects_its_own_reversion.py``.  Every
# ``test`` names a test declared IN THIS MODULE (#1065).
# ----------------------------------------------------------------------
REVERSIONS = [
    Reversion(
        target=_WS,
        find="            daemon_loop = _ws_daemon_loop(ws_server)",
        replace="            daemon_loop = ws_server._event_loop",
        test="test_pre_init_hook_reads_the_adapters_loop_not_a_nonexistent_one",
        because=(
            "the AppArmor pre-init hook reads ws_server._event_loop directly "
            "again, which does not exist on JaatoWSServer and raises "
            "AttributeError on every WS session, silently skipping "
            "confinement and the runner spawn"
        ),
    ),
    Reversion(
        target=_WS,
        find="        loop = _ws_daemon_loop(self)",
        replace="        loop = self._event_loop",
        test="test_app_secret_resolver_degrades_when_no_adapter_exists_yet",
        because=(
            "_resolve_app_secret_over_bind_channel reads self._event_loop "
            "directly again, raising AttributeError instead of the "
            "documented AppSecretAnswer(status='unreachable', ...) degrade"
        ),
    ),
    Reversion(
        target=_WS,
        find=(
            "    adapter = ws_server._event_sink_adapter\n"
            "    return adapter._event_loop if adapter is not None else None"
        ),
        replace="    return ws_server._event_loop",
        test="test_pre_init_hook_reads_the_adapters_loop_not_a_nonexistent_one",
        because=(
            "_ws_daemon_loop itself reads ws_server._event_loop directly, "
            "which does not exist on JaatoWSServer — both call sites would "
            "inherit the crash through the very helper meant to prevent it"
        ),
    ),
]


class _FakeEventSinkAdapter:
    """The real shape: the loop lives HERE, not on the server."""

    def __init__(self, loop: Optional[Any]) -> None:
        self._event_loop = loop


class _FakeAppArmorForHook:
    def is_available(self) -> bool:
        return True

    def provision_profile(self, *a: Any, **k: Any) -> bool:
        return True

    def get_profile_name(self, session_id: str) -> str:
        return f"jaato-ws-{session_id}"

    def confinement_id_for_boundary(self, *a: Any, **k: Any) -> str:
        return "boundary-id"


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


def _make_ws_server(workspace_root: str, *, adapter_loop: Optional[Any]):
    """A ``JaatoWSServer`` double whose ONLY source of a loop is a fake
    ``_event_sink_adapter`` — never a bare ``_event_loop`` on the server
    itself, since the real class has none.  ``adapter_loop=None`` with a
    real (adapter-bearing) instance models "start() has not bound one
    yet"; pass ``no_adapter=True`` via ``_event_sink_adapter = None`` at
    the call site to model "start() has not even constructed one".
    """
    from jaato_server.server.websocket import JaatoWSServer

    ws = JaatoWSServer.__new__(JaatoWSServer)
    ws._apparmor = _FakeAppArmorForHook()
    ws._cgroups = None
    ws._workspace_root = workspace_root
    ws._event_sink_adapter = _FakeEventSinkAdapter(adapter_loop)
    return ws


def _register_pre_init_hook(ws: Any) -> Any:
    from jaato_server.server.websocket import JaatoWSServer

    router = _FakeRouterForHook()
    JaatoWSServer.set_command_router(ws, router)
    assert router._session_manager.pre_init_hooks, "hook not registered"
    return router._session_manager.pre_init_hooks[0]


def test_pre_init_hook_reads_the_adapters_loop_not_a_nonexistent_one(
    tmp_path, caplog,
) -> None:
    """A real loop object on ``_event_sink_adapter`` reaches the hook's
    ``daemon_loop`` — proving the read goes through the adapter, not a
    nonexistent attribute on the server itself.  Reaching AppArmor
    provisioning (never the "no daemon loop captured" WARNING) is the
    observable proof, since ``daemon_loop`` itself is a local.
    """
    ws_root = tmp_path / "ws_root"
    ws_root.mkdir()
    sess_dir = ws_root / "session_dir"
    sess_dir.mkdir()

    ws = _make_ws_server(str(ws_root), adapter_loop=object())
    hook = _register_pre_init_hook(ws)

    with patch(
        "jaato_server.server.apparmor.resolve_plugin_apparmor_rules",
        lambda **k: None,
    ), patch(
        "jaato_server.server.runner_spawn.spawn_session_runner",
        lambda **k: None,
    ), patch(
        "jaato_server.server.runner_spawn.dispatch_bootstrap_envelope",
        lambda **k: None,
    ):
        server = type("_S", (), {})()
        server.config_root = None  # type: ignore[attr-defined]
        server._profile = None  # type: ignore[attr-defined]
        server.note_runner_bootstrap_outcome = lambda *a, **k: None  # type: ignore[attr-defined]

        with caplog.at_level("WARNING"):
            hook(server, "s-loop-ok", str(sess_dir), client_id=None)

    assert not any(
        "no daemon loop captured" in r.message for r in caplog.records
    ), "the hook fell back to the no-loop branch despite a real loop being available"


def test_pre_init_hook_degrades_when_no_adapter_exists_yet(
    tmp_path, caplog,
) -> None:
    """``_event_sink_adapter`` itself is ``None`` (``start()`` has not run
    yet) — the hook must take the documented "no daemon loop captured"
    WARNING-and-return path, never crash with an unrelated AttributeError
    on ``None._event_loop``."""
    ws_root = tmp_path / "ws_root"
    ws_root.mkdir()
    sess_dir = ws_root / "session_dir"
    sess_dir.mkdir()

    ws = _make_ws_server(str(ws_root), adapter_loop=None)
    ws._event_sink_adapter = None
    hook = _register_pre_init_hook(ws)

    server = type("_S", (), {})()
    with caplog.at_level("WARNING"):
        hook(server, "s-no-adapter", str(sess_dir), client_id=None)  # must not raise

    assert any(
        "no daemon loop captured" in r.message for r in caplog.records
    ), "a missing adapter did not produce the documented degrade warning"


def test_app_secret_resolver_degrades_when_no_adapter_exists_yet() -> None:
    """With no ``_event_sink_adapter`` at all, the method must return the
    documented ``AppSecretAnswer(status="unreachable", ...)`` rather than
    raise ``AttributeError`` — the exact contract violation #1299 reports.
    """
    from jaato_server.server.app_secret import AppSecretAnswer
    from jaato_server.server.websocket import JaatoWSServer

    ws = JaatoWSServer.__new__(JaatoWSServer)
    ws._event_sink_adapter = None
    ws._app_connection_for = lambda app_id: "client-1"  # type: ignore[method-assign]

    answer = JaatoWSServer._resolve_app_secret_over_bind_channel(
        ws, "app-1", "user-1", "ws-1", "SECRET", 0.01,
    )
    assert isinstance(answer, AppSecretAnswer)
    assert answer.status == "unreachable"
