"""A session's own creation resolves an AppArmor profile and a
``config_root`` the same way a revive of it would (#1293).

Live daemon evidence (#1293): a session's FIRST bootstrap
(``session.new`` on a WS client) resolved no AppArmor profile
(``profile=(none) ... confined=False``) and a ``file_edit`` plugin
initialize failure (``no config_root set``); every LATER revive of the
same session resolved both correctly.  Traced against ``main`` (which
already carries #1260/#1278/#1281), this turned out to be two
independently-confirmed defects, not one:

1. **``config_root`` (the fully-confirmed, deterministic half).**
   ``_create_session_impl`` (CREATE) stopped at the client's own
   ``ClientConfigRequest.config_root`` — absent that, ``config_root``
   stayed ``None``.  ``_load_session_impl`` (REVIVE), via
   ``_resolve_restore_config_root``, has a THIRD fallback tier —
   ``<workspace_path>/.jaato`` — that CREATE never had.  No WS client in
   this tree ever sends ``ClientConfigRequest.config_root`` (the TS SDK's
   ``openSession`` only forwards an explicit ``configRoot``; it never
   derives one from ``workspacePath`` the way the Python ``IPCClient``
   does), so this reproduces on EVERY WS-created session, matching the
   reported ``file_edit`` failure exactly.  Fixed by giving CREATE the
   same fallback, via the SAME resolver revive already used — one
   definition, so the two paths cannot drift about what an omitted
   ``config_root`` means.

2. **The WS-overlap-precedence skip never fires (the code-confirmed,
   independently real half).**  ``SessionManager.set_apparmor_dependencies``
   is what stores ``self._ws_server_ref`` — the reference
   ``_workspace_under_ws_root`` reads to decide whether
   ``_provision_ipc_apparmor_and_spawn_runner`` (the IPC/SessionManager
   spawn path) should skip a WS-provisioned session and leave it to the
   WS pre-init hook.  ``JaatoDaemon.start()`` called it BEFORE
   ``self._ws_server = JaatoWSServer(...)`` was constructed, so
   ``ws_server=self._ws_server`` was always ``None`` — permanently, for
   the daemon's whole life.  ``_workspace_under_ws_root`` therefore always
   answered ``False`` and the skip never fired: every WS-provisioned
   session's bootstrap ran BOTH the unconditional IPC spawn path (which
   spawns UNCONFINED, since a WS client never sets the IPC-only
   ``ClientConfigRequest.apparmor`` opt-in) AND the WS pre-init hook
   (which confines unconditionally whenever the host supports it) —
   a double spawn per bootstrap, exactly the shape #1282 flagged and left
   open. Fixed by moving the wiring call to AFTER both transports are
   constructed, restoring the WS pre-init hook as the SOLE confinement +
   spawn decision-maker for WS-provisioned sessions (#1253's fail-closed
   posture then applies with no competing spawn to race against).

A third, smaller gap in the same family: the WS pre-init hook's own
``resolve_plugin_apparmor_rules(...)`` call hardcoded ``config_root=None``
rather than reading ``server.config_root`` (already resolved by the time
pre-init hooks run) — so even a correctly-confined WS session composed its
AppArmor profile without whatever config_root-relative grant a plugin
contributes.  Fixed to read ``server.config_root``, matching what the IPC
path already threads.

Whether the reported session's OWN incident additionally exercised a race
between the two (now-eliminated) spawn paths could not be pinned down with
certainty from the code alone — see the accompanying PR description.  What
IS certain, and what this file pins: with (2) fixed there is no unconfined
spawn on the WS path to race against in the first place, and with (1) fixed
CREATE and REVIVE agree, deterministically, about what an omitted
``config_root`` means.

**No kernel here.**  This container carries no AppArmor LSM (#1023/#1033/
#1100 all record it).  These tests exercise the FRAMEWORK's decisions —
which value ``_ws_server_ref`` ends up holding, whether the WS-overlap skip
fires, what ``config_root`` a fresh session resolves — from fabricated
manager/server state and source-level (AST) assertions, never the kernel's
behaviour.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest

from jaato_server.shared.tests.reversion import Reversion

_MAIN = "jaato-server/jaato_server/server/__main__.py"
_SM = "jaato-server/jaato_server/server/session_manager.py"
_WS = "jaato-server/jaato_server/server/websocket.py"


# ----------------------------------------------------------------------
# Reversions — read by
# ``shared/tests/test_every_guard_detects_its_own_reversion.py``.  Every
# ``test`` names a test declared IN THIS MODULE (#1065).
# ----------------------------------------------------------------------
REVERSIONS = [
    # (2) — the WS-server reference reaches SessionManager only AFTER the
    # WS server exists.  Dropping the (moved) call restores the original
    # bug: ``self._ws_server_ref`` never gets the real object, so the
    # WS-overlap-precedence skip never fires and every WS-provisioned
    # session's bootstrap runs both spawn paths.
    Reversion(
        target=_MAIN,
        find=(
            '            logger.info(f"WebSocket server will listen on '
            '{scheme}://{host}:{port}")\n'
            "\n"
            "        # #1293: wire the IPC AppArmor + runner-spawn "
            "dependencies onto\n"
            "        # the session manager AFTER both transports (IPC, WS) "
            "have been\n"
            "        # constructed, so ``self._ws_server`` is the real "
            "object when\n"
            "        # ``self.web_socket`` is configured — see the comment "
            "above where\n"
            "        # this call used to sit, before ``self._ws_server`` "
            "existed.\n"
            "        self._wire_ipc_apparmor_dependencies()"
        ),
        replace=(
            '            logger.info(f"WebSocket server will listen on '
            '{scheme}://{host}:{port}")'
        ),
        test="test_wire_ipc_apparmor_dependencies_runs_after_ws_server_construction",
        because=(
            "self._wire_ipc_apparmor_dependencies() is never called after "
            "self._ws_server is constructed, so SessionManager._ws_server_ref "
            "stays None for the daemon's whole life and the WS-overlap "
            "precedence skip never fires"
        ),
    ),
    # (1) — CREATE's config_root fallback.  Reverting to the old two-source
    # chain (explicit override, then the client's own config_root) drops the
    # third tier revive already has, so a WS client that never sends
    # ClientConfigRequest.config_root creates a session with config_root=None.
    Reversion(
        target=_SM,
        find=(
            "        if config_root is None:\n"
            "            config_root = self._resolve_restore_config_root(\n"
            "                None, client_config.get('config_root'), "
            "workspace_path,\n"
            "            )"
        ),
        replace=(
            "        if config_root is None:\n"
            "            config_root = client_config.get('config_root')"
        ),
        test="test_create_session_config_root_defaults_to_workspace_jaato",
        because=(
            "a fresh session created by a client that never sends "
            "config_root (every WS client in this tree) has config_root=None"
        ),
    ),
    # (3) — the WS pre-init hook's own config_root, threaded into
    # resolve_plugin_apparmor_rules.  Reverting to the hardcoded None means
    # a confined WS session's AppArmor profile is composed with no
    # config_root, silently dropping any config_root-relative plugin grant.
    Reversion(
        target=_WS,
        find=(
            "                    # #1293: was hardcoded ``None``.  "
            "``server.config_root``\n"
            "                    # is already set by "
            "``_construct_and_initialize_server``\n"
            "                    # (BEFORE pre-init hooks run) from the "
            "same\n"
            "                    # ``envelope.config_root`` the IPC path\n"
            "                    # "
            "(``_provision_ipc_apparmor_and_spawn_runner``) threads\n"
            "                    # through — reading it here keeps this "
            "hook's plugin\n"
            "                    # rule set (and thus the rendered "
            "profile) consistent\n"
            "                    # with what the IPC path would have "
            "computed for the\n"
            "                    # same session, instead of silently "
            "omitting any\n"
            "                    # config_root-relative grant a plugin "
            "contributes.\n"
            "                    config_root=getattr(server, "
            '"config_root", None),\n'
            "                    managed_workspace_root=ws_workspace_root,\n"
            "                )\n"
            "                # #1033: the profile is named after the "
            "BOUNDARY, so a"
        ),
        replace=(
            "                    config_root=None,\n"
            "                    managed_workspace_root=ws_workspace_root,\n"
            "                )\n"
            "                # #1033: the profile is named after the "
            "BOUNDARY, so a"
        ),
        test="test_ws_pre_init_hook_forwards_the_servers_config_root",
        because=(
            "the WS pre-init hook composes the AppArmor profile with "
            "config_root=None hardcoded, instead of the session's own "
            "resolved config_root"
        ),
    ),
]


# ----------------------------------------------------------------------
# (2) — the WS-server reference reaches SessionManager only after the WS
# server exists.  Source-level (AST) rather than behavioural: driving
# ``JaatoDaemon.start()`` for real needs a running asyncio loop, real
# sockets and a real daemon lifecycle — what the ORDERING guarantees is a
# property of the SOURCE, and #1080's own precedent ("a guard on prose is a
# guard on nothing") is why this walks the AST rather than grepping text.
# ----------------------------------------------------------------------


def _daemon_start_source() -> str:
    repo_root = Path(__file__).resolve()
    # jaato-server/jaato_server/server/tests/<this file> -> repo root is
    # four parents up.
    for _ in range(6):
        repo_root = repo_root.parent
        candidate = repo_root / _MAIN
        if candidate.is_file():
            return candidate.read_text()
    raise AssertionError(f"could not locate {_MAIN} from {__file__}")


def _line_of_ws_server_assignment(tree: ast.AST) -> Optional[int]:
    """The line of ``self._ws_server = JaatoWSServer(...)``."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        func_name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if func_name != "JaatoWSServer":
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "_ws_server"
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                return node.lineno
    return None


def _line_of_wire_apparmor_call(tree: ast.AST) -> Optional[int]:
    """The line of ``self._wire_ipc_apparmor_dependencies()``."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr != "_wire_ipc_apparmor_dependencies":
            continue
        if not (isinstance(func.value, ast.Name) and func.value.id == "self"):
            continue
        return node.lineno
    return None


def test_wire_ipc_apparmor_dependencies_runs_after_ws_server_construction() -> None:
    """``self._wire_ipc_apparmor_dependencies()`` must run strictly after
    ``self._ws_server = JaatoWSServer(...)``.

    Before #1293 it ran BEFORE — ``self._ws_server`` was still ``None`` —
    so ``SessionManager.set_apparmor_dependencies(ws_server=None, ...)`` was
    the only call the daemon ever made, on every daemon that has ever run
    this source.  ``_ws_server_ref`` never held the real WS server.
    """
    tree = ast.parse(_daemon_start_source())
    ws_line = _line_of_ws_server_assignment(tree)
    call_line = _line_of_wire_apparmor_call(tree)
    assert ws_line is not None, (
        "could not find `self._ws_server = JaatoWSServer(...)` in "
        f"{_MAIN} — has it moved or been renamed?"
    )
    assert call_line is not None, (
        "could not find a call to `self._wire_ipc_apparmor_dependencies()` "
        f"in {_MAIN} — the wiring call was removed entirely"
    )
    assert call_line > ws_line, (
        "self._wire_ipc_apparmor_dependencies() (line "
        f"{call_line}) must run AFTER self._ws_server = JaatoWSServer(...) "
        f"(line {ws_line}), or SessionManager._ws_server_ref is wired with "
        "None and the WS-overlap-precedence skip never fires (#1293)"
    )


# ----------------------------------------------------------------------
# (1) — config_root defaults on CREATE the same way it does on REVIVE.
# Drives the real SessionManager().create_session(...) with a lightweight
# fake JaatoServer (mirrors test_bootstrap_helper.py's pattern) so the
# assertion is on the actual code path a WS session.new takes, not on the
# resolver in isolation.
# ----------------------------------------------------------------------


class _FakeServerForCreate:
    """A JaatoServer stand-in wide enough to survive
    ``_create_session_impl`` end to end without real plugins, providers
    or disk-backed session persistence beyond the workspace directory the
    test already owns.
    """

    instances: List["_FakeServerForCreate"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)
        self.config_root: Optional[str] = None
        self.registry = None
        self.auth_pending = False
        self.model_provider = "echo"
        self.model_name = "echo-1"
        self._runner_rpc = None
        self._profile = None
        self._auth_complete_cb: Optional[Callable[[], None]] = None
        _FakeServerForCreate.instances.append(self)

    def initialize(self) -> bool:
        return True

    def set_event_callback(self, cb: Any) -> None:
        self._event_cb = cb

    def set_app_secret_resolver(self, resolver: Any) -> None:
        self._app_secret_resolver = resolver

    def _resolve_session_env(self) -> None:
        pass

    def _with_session_env(self):
        from contextlib import contextmanager

        @contextmanager
        def _noop():
            yield

        return _noop()

    def _in_workspace(self):
        from contextlib import contextmanager

        @contextmanager
        def _noop():
            yield

        return _noop()

    def create_registry_and_discover(self) -> None:
        pass

    def set_auth_complete_callback(self, cb: Callable[[], None]) -> None:
        self._auth_complete_cb = cb

    def _find_plugin_for_command(self, name: str) -> None:
        return None


@pytest.fixture()
def _patch_fake_server():
    _FakeServerForCreate.instances.clear()
    with patch(
        "jaato_server.server.session_manager.JaatoServer",
        _FakeServerForCreate,
    ):
        yield


def _make_bare_session_manager():
    """A real ``SessionManager()``, with the two apparmor/spawn seams
    stubbed so no runner is ever spawned and no AppArmor manager is ever
    touched — this test is about ``config_root`` only.  ``client_id`` has
    no ``ClientConfigRequest.config_root`` on record, mirroring every WS
    client in this tree (the TS SDK never sends one, confirmed against
    ``jaato-web-coder-ui/src`` and ``jaato-sdk-ts/src/convenience.ts``).
    """
    from jaato_server.server.session_manager import SessionManager

    sm = SessionManager()
    # No apparmor opt-in machinery involved in this test: keep it inert
    # rather than exercising (and having to fake) the real provisioning /
    # spawn machinery, which #1293's OTHER two reversions already cover.
    sm._provision_ipc_apparmor_and_spawn_runner = (  # type: ignore[assignment]
        lambda *a, **k: None
    )
    return sm


def test_create_session_config_root_defaults_to_workspace_jaato(
    tmp_path, _patch_fake_server,
) -> None:
    """A fresh session created by a client that never sent
    ``ClientConfigRequest.config_root`` (every WS client here) resolves
    ``config_root`` to ``<workspace_path>/.jaato`` — the SAME default a
    revive of that session already fell back to via
    ``_resolve_restore_config_root``.

    Before #1293's fix this was the reported symptom's second half: a
    ``file_edit`` plugin initialize failure ("no config_root set") on the
    session's OWN creation, silently fixed only once it was later revived.
    """
    sm = _make_bare_session_manager()
    workspace = str(tmp_path)
    client_id = "client-1"
    # The client sent a ClientConfigRequest with no config_root — the
    # shape ``_client_config[client_id]`` takes for every WS client in
    # this tree (jaato-sdk-ts's convenience.ts only forwards an explicit
    # ``configRoot``; the web client sends no client-config fields beyond
    # ``presentation`` at all).
    sm._client_config[client_id] = {}

    session_id = sm.create_session(
        client_id=client_id,
        workspace_path=workspace,
    )
    assert session_id, "session creation must succeed against the fake server"

    assert len(_FakeServerForCreate.instances) == 1
    server = _FakeServerForCreate.instances[0]
    assert server.config_root == str(Path(workspace) / ".jaato"), (
        "a session created with no client-supplied config_root must "
        "default to <workspace_path>/.jaato — the same fallback "
        "_resolve_restore_config_root already gives a REVIVE"
    )


def test_create_session_explicit_client_config_root_still_wins(
    tmp_path, _patch_fake_server,
) -> None:
    """The client's own ``ClientConfigRequest.config_root`` (when sent)
    still wins over the new <workspace>/.jaato default — the fallback
    added here must not shadow an explicit, non-default value."""
    sm = _make_bare_session_manager()
    workspace = str(tmp_path)
    client_id = "client-1"
    explicit_root = str(tmp_path / "elsewhere" / ".jaato-custom")
    sm._client_config[client_id] = {"config_root": explicit_root}

    session_id = sm.create_session(
        client_id=client_id,
        workspace_path=workspace,
    )
    assert session_id

    server = _FakeServerForCreate.instances[0]
    assert server.config_root == explicit_root


# ----------------------------------------------------------------------
# (3) — the WS pre-init hook forwards the session's own config_root into
# resolve_plugin_apparmor_rules, instead of the hardcoded None.
# ----------------------------------------------------------------------


class _FakeAppArmorForHook:
    def __init__(self) -> None:
        self.available = True

    def is_available(self) -> bool:
        return self.available

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


def _make_ws_server_for_hook(workspace_root: str):
    from jaato_server.server.websocket import JaatoWSServer

    ws = JaatoWSServer.__new__(JaatoWSServer)
    ws._apparmor = _FakeAppArmorForHook()
    ws._cgroups = None
    ws._workspace_root = workspace_root
    ws._event_loop = "<loop>"
    return ws


def _register_pre_init_hook(ws: Any) -> Any:
    from jaato_server.server.websocket import JaatoWSServer

    router = _FakeRouterForHook()
    JaatoWSServer.set_command_router(ws, router)
    assert router._session_manager.pre_init_hooks, "hook not registered"
    return router._session_manager.pre_init_hooks[0]


def test_ws_pre_init_hook_forwards_the_servers_config_root(
    tmp_path, monkeypatch,
) -> None:
    """The WS pre-init hook's ``resolve_plugin_apparmor_rules(...)`` call
    is given the session's OWN ``server.config_root`` (already resolved by
    ``_construct_and_initialize_server`` before pre-init hooks run), not a
    hardcoded ``None`` — matching what the IPC path
    (``_provision_ipc_apparmor_and_spawn_runner``) already threads."""
    ws_root = tmp_path / "ws_root"
    ws_root.mkdir()
    sess_dir = ws_root / "session_dir"
    sess_dir.mkdir()
    resolved_config_root = str(sess_dir / ".jaato")

    calls: List[Dict[str, Any]] = []

    def _recording_resolver(**kwargs: Any) -> None:
        calls.append(kwargs)
        return None

    monkeypatch.setattr(
        "jaato_server.server.apparmor.resolve_plugin_apparmor_rules",
        _recording_resolver,
    )
    with patch(
        "jaato_server.server.runner_spawn.spawn_session_runner",
        lambda **k: None,
    ), patch(
        "jaato_server.server.runner_spawn.dispatch_bootstrap_envelope",
        lambda **k: None,
    ):
        ws = _make_ws_server_for_hook(str(ws_root))
        hook = _register_pre_init_hook(ws)

        server = type("_S", (), {})()
        server.config_root = resolved_config_root  # type: ignore[attr-defined]
        server.note_runner_bootstrap_outcome = lambda *a, **k: None  # type: ignore[attr-defined]

        hook(server, "s-cfgroot", str(sess_dir), client_id=None)

    assert len(calls) == 1, "resolve_plugin_apparmor_rules must be called exactly once"
    assert calls[0]["config_root"] == resolved_config_root, (
        "the WS pre-init hook must forward server.config_root into "
        "resolve_plugin_apparmor_rules, not a hardcoded None"
    )
