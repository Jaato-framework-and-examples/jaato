"""The managed workspace root reaches the SessionManager spawn path (#1280).

On a live premium daemon a WUI session's agent ran ``echo "$HOME"`` through
``cli_based_tool`` and got ``/root``, the daemon's own home, although the
workspace (``~/.jaato/workspaces/<name>``) is daemon-managed and #1225 says it
gets ``HOME=<ws>/.home``.  #1274's ``<ws>/.jaato/tool-venv`` was off too.

Both defaults are decided per workspace by ``_is_daemon_managed(workspace,
managed_workspace_root)``, which is False whenever the root is ``None``.  The
WS pre-init hook passes the root at every call.  The spawn path a premium WUI
``session.new`` takes, ``SessionManager._spawn_session_runner_unconditional``
(the same path #1253 / #1278 fixed for ``confinement_required``), passed it
nowhere:

| call on that path | what the missing root turned off |
|---|---|
| ``spawn_session_runner`` | the daemon creating ``<ws>/.home`` |
| ``dispatch_bootstrap_envelope`` | ``workspace_home`` / ``workspace_venv`` in the envelope |
| ``resolve_plugin_apparmor_rules`` | the #1274 exec grants for the venv / home ``bin`` |

And the session manager had no way to learn the root:
``set_apparmor_dependencies(ws_server=...)`` is called before the daemon
constructs the WS server, so that reference is ``None`` on every WS daemon.
``JaatoWSServer.set_command_router`` (the seam that registers the pre-init
hook) now hands the root over.

The defect is a missing keyword, so the tests drive the REAL
``_spawn_session_runner_unconditional`` and the real helpers it calls,
stubbing only what would fork a process.  A test that called
``inject_workspace_home`` directly would pass on the broken tree.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from jaato_server.server.session_manager import SessionManager
from jaato_server.shared.plugins.workspace_home import DEFAULT_WORKSPACE_HOME
from jaato_server.shared.plugins.workspace_venv import DEFAULT_WORKSPACE_VENV
from jaato_server.shared.tests.reversion import Reversion

_SM = "jaato-server/jaato_server/server/session_manager.py"
_WS = "jaato-server/jaato_server/server/websocket.py"


# ----------------------------------------------------------------------
# Reversions, read by
# ``shared/tests/test_every_guard_detects_its_own_reversion.py``.  Every
# ``test`` names a test declared IN THIS MODULE (#1065): the meta-guard
# resolves it as ``<this module>::<test>``.
# ----------------------------------------------------------------------
REVERSIONS = [
    Reversion(
        target=_SM,
        find=(
            "                cascade_driver_id=cascade_driver_id,\n"
            "                managed_workspace_root=managed_workspace_root,\n"
            "            )"
        ),
        replace=(
            "                cascade_driver_id=cascade_driver_id,\n"
            "            )"
        ),
        test="test_unconditional_spawn_creates_the_workspace_home",
        because=(
            "spawn_session_runner no longer gets the managed root on this "
            "path, so the daemon never creates <ws>/.home for a WUI session"
        ),
    ),
    Reversion(
        target=_SM,
        find=(
            "                    # workspace HOME / venv defaults from this.\n"
            "                    managed_workspace_root=managed_workspace_root,\n"
        ),
        replace=(
            "                    # workspace HOME / venv defaults from this.\n"
        ),
        test="test_envelope_carries_workspace_home_and_venv",
        because=(
            "the bootstrap envelope is built without the managed root, so "
            "workspace_home / workspace_venv are silently off (HOME=/root)"
        ),
    ),
    Reversion(
        target=_SM,
        find=(
            "            managed_workspace_root=self._managed_workspace_root_for_spawn(),\n"
            "        )"
        ),
        replace="        )",
        test="test_provision_path_resolves_plugin_rules_with_the_managed_root",
        because=(
            "the AppArmor plugin rules on this path lose the #1274 exec "
            "grants for the workspace venv and home bin directories"
        ),
    ),
    Reversion(
        target=_WS,
        find='        _hand_managed_root_to(sm, getattr(self, "_workspace_root", None))',
        replace="        pass  # #1280 reversion",
        test="test_ws_server_hands_its_root_to_the_session_manager",
        because=(
            "the session manager is never told the WS provisioning root, so "
            "its spawn path has nothing to pass"
        ),
    ),
]


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------


class _StopBeforeFork(Exception):
    """Raised in place of a cold-spawn fork: everything before it is real."""


class _FakeRPC:
    """Captures the envelope the real ``dispatch_bootstrap_envelope`` sends."""

    def __init__(self) -> None:
        self.envelopes: List[Any] = []

    def bootstrap_session_threadsafe(self, envelope: Any, timeout: float) -> Dict[str, Any]:
        self.envelopes.append(envelope)
        return {"ok": True}


class _FakeServer:
    """Enough of a JaatoServer for the spawn path and ``build_session_envelope``.

    Profile-less on purpose: every workspace the web client creates is a bare
    ``.env`` with no profile, which is the reported case.
    """

    def __init__(self) -> None:
        self._profile = None
        self._session_env = {
            "MODEL_NAME": "some-model", "JAATO_PROVIDER": "anthropic",
        }
        self._agent_params: Dict[str, str] = {}
        self._main_agent_id = "main"
        self.config_root: Optional[str] = None
        self.runner_rpc = _FakeRPC()
        self._spawn_isolated_runner_handler = None
        self._cascade_budget_pool = None
        self._draws_on_parent_budget = True
        self.bootstrap_outcomes: List[Optional[str]] = []

    def note_runner_bootstrap_outcome(self, reason: Optional[str]) -> None:
        self.bootstrap_outcomes.append(reason)


def _bare_sm(managed_root: Optional[str]) -> SessionManager:
    """A SessionManager with only what the unconditional spawn path reads.

    ``set_managed_workspace_root`` and ``_managed_workspace_root_for_spawn``
    are the real methods; everything else the path touches is a no-op.
    """
    sm = SessionManager.__new__(SessionManager)
    sm._daemon_loop = object()  # type: ignore[attr-defined]
    sm._pool_manager_ref = None  # type: ignore[attr-defined]
    sm.get_cascade_budget = lambda *a, **k: None  # type: ignore[assignment]
    sm._reconcile_cascade_pool = lambda *a, **k: None  # type: ignore[assignment]
    sm._record_runner_identity = lambda *a, **k: None  # type: ignore[assignment]
    sm._teardown_prior_apparmor_profile_after_transition = (  # type: ignore[assignment]
        lambda **k: None
    )
    sm._resolve_cgroups_manager = lambda: None  # type: ignore[assignment]
    sm._notify_apparmor = lambda *a, **k: None  # type: ignore[assignment]
    sm.set_managed_workspace_root(managed_root)
    return sm


def _spawn(sm: SessionManager, server: _FakeServer, workspace: Path) -> bool:
    return sm._spawn_session_runner_unconditional(
        server=server,
        session_id="sess-1280",
        workspace_path=str(workspace),
        client_id="client-1",
        profile_name="",
    )


@pytest.fixture()
def _stop_before_fork(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the real ``spawn_session_runner`` up to the cold-spawn fork."""
    def _raise(*_a: Any, **_k: Any) -> None:
        raise _StopBeforeFork()

    monkeypatch.setattr(
        "jaato_server.server.runner_spawn._cold_spawn_runner", _raise)
    monkeypatch.setattr(
        "jaato_server.server.runner_spawn._ensure_session_tmpdir",
        lambda *a, **k: None,
    )


@pytest.fixture()
def _no_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the process spawn; the real bootstrap dispatch still runs."""
    monkeypatch.setattr(
        "jaato_server.server.runner_spawn.spawn_session_runner",
        lambda **kwargs: None,
    )


def _layout(tmp_path: Path) -> Dict[str, Path]:
    root = tmp_path / "workspaces"
    managed = root / "this-is-a-confinement-test"
    outside = tmp_path / "users-own-checkout"
    managed.mkdir(parents=True)
    outside.mkdir()
    return {"root": root, "managed": managed, "outside": outside}


# ----------------------------------------------------------------------
# The workspace HOME directory is created (spawn_session_runner)
# ----------------------------------------------------------------------


def test_unconditional_spawn_creates_the_workspace_home(
    tmp_path: Path, _stop_before_fork: None,
) -> None:
    """A workspace under the managed root gets ``<ws>/.home`` from the daemon
    before the runner starts, or the redirected HOME points nowhere."""
    paths = _layout(tmp_path)
    sm = _bare_sm(str(paths["root"]))

    ok = _spawn(sm, _FakeServer(), paths["managed"])

    assert ok is False, "the fork was stubbed to fail; nothing else should"
    assert (paths["managed"] / DEFAULT_WORKSPACE_HOME).is_dir(), (
        "the unconditional spawn path did not create the workspace HOME for "
        "a daemon-managed workspace"
    )


def test_unconditional_spawn_leaves_an_unmanaged_workspace_alone(
    tmp_path: Path, _stop_before_fork: None,
) -> None:
    """An IPC / user-CWD workspace outside the root grows no ``.home``."""
    paths = _layout(tmp_path)
    sm = _bare_sm(str(paths["root"]))

    _spawn(sm, _FakeServer(), paths["outside"])

    assert not (paths["outside"] / DEFAULT_WORKSPACE_HOME).exists()


# ----------------------------------------------------------------------
# The envelope carries the defaults (dispatch_bootstrap_envelope)
# ----------------------------------------------------------------------


def test_envelope_carries_workspace_home_and_venv(
    tmp_path: Path, _no_process: None,
) -> None:
    """The real ``build_session_envelope`` folds both managed defaults into the
    envelope's ``cli`` config for a workspace under the managed root."""
    paths = _layout(tmp_path)
    sm = _bare_sm(str(paths["root"]))
    server = _FakeServer()

    assert _spawn(sm, server, paths["managed"]) is True

    assert len(server.runner_rpc.envelopes) == 1
    cli = server.runner_rpc.envelopes[0].plugin_configs.get("cli") or {}
    assert cli.get("workspace_home") == DEFAULT_WORKSPACE_HOME, (
        "HOME stays the runner's (/root) for a WUI session: the envelope "
        "carries no workspace_home"
    )
    assert cli.get("workspace_venv") == DEFAULT_WORKSPACE_VENV


def test_envelope_for_an_unmanaged_workspace_is_unchanged(
    tmp_path: Path, _no_process: None,
) -> None:
    """Outside the root, nothing is injected: the envelope is what it was
    before #1280."""
    paths = _layout(tmp_path)
    sm = _bare_sm(str(paths["root"]))
    server = _FakeServer()

    _spawn(sm, server, paths["outside"])

    configs = server.runner_rpc.envelopes[0].plugin_configs
    cli = configs.get("cli") or {}
    assert "workspace_home" not in cli
    assert "workspace_venv" not in cli


def test_no_root_known_injects_nothing(
    tmp_path: Path, _no_process: None,
) -> None:
    """An IPC-only daemon (no WS server, so no root) behaves as before."""
    paths = _layout(tmp_path)
    sm = _bare_sm(None)
    server = _FakeServer()

    _spawn(sm, server, paths["managed"])

    cli = server.runner_rpc.envelopes[0].plugin_configs.get("cli") or {}
    assert "workspace_home" not in cli


# ----------------------------------------------------------------------
# The AppArmor plugin rules get the root (resolve_plugin_apparmor_rules)
# ----------------------------------------------------------------------


def test_provision_path_resolves_plugin_rules_with_the_managed_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rules that grant exec on the venv / home ``bin`` are resolved with
    the same root the envelope uses."""
    paths = _layout(tmp_path)
    seen: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        "jaato_server.server.apparmor.resolve_plugin_apparmor_rules",
        lambda **kwargs: seen.append(kwargs),
    )
    sm = SessionManager.__new__(SessionManager)
    sm._client_config = {}  # type: ignore[attr-defined]
    sm._pending_client_tools = {}  # type: ignore[attr-defined]
    sm._workspace_under_ws_root = lambda _p: False  # type: ignore[assignment]
    sm._spawn_session_runner_unconditional = (  # type: ignore[assignment]
        lambda **kwargs: True
    )
    sm.set_managed_workspace_root(str(paths["root"]))

    sm._provision_ipc_apparmor_and_spawn_runner(
        _FakeServer(), "sess-1280", str(paths["managed"]), "client-1",
    )

    assert len(seen) == 1
    assert seen[0].get("managed_workspace_root") == str(paths["root"])


# ----------------------------------------------------------------------
# The session manager learns the root from the WS server
# ----------------------------------------------------------------------


def test_ws_server_hands_its_root_to_the_session_manager(tmp_path: Path) -> None:
    """``set_command_router`` gives the router's session manager the WS
    server's ``workspace_root``: it is the only source the session manager
    has, since ``set_apparmor_dependencies`` runs before the WS server
    exists."""
    from jaato_server.server.websocket import JaatoWSServer

    root = str(tmp_path / "workspaces")
    sm = SessionManager.__new__(SessionManager)
    sm._pre_initialize_hooks = []  # type: ignore[attr-defined]
    sm._session_hooks = []  # type: ignore[attr-defined]

    class _Router:
        _session_manager = sm

    class _WS:
        _workspace_root = root

    JaatoWSServer.set_command_router(_WS(), _Router())  # type: ignore[arg-type]

    assert sm._managed_workspace_root_for_spawn() == root


def test_a_session_manager_without_the_setter_is_tolerated() -> None:
    """A test double or out-of-tree session manager without
    ``set_managed_workspace_root`` is left alone, not crashed."""
    from jaato_server.server.websocket import _hand_managed_root_to

    _hand_managed_root_to(object(), "/somewhere")


# ----------------------------------------------------------------------
# Call-site guard: every spawn call on this path passes the root
# ----------------------------------------------------------------------


def _method(tree: ast.Module, cls: str, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"{cls}.{name} not found")


def _calls(fn: ast.FunctionDef, callee: str) -> List[ast.Call]:
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            called = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
            if called == callee:
                out.append(node)
    return out


@pytest.mark.parametrize(("method", "callee"), [
    ("_spawn_session_runner_unconditional", "spawn_session_runner"),
    ("_spawn_session_runner_unconditional", "dispatch_bootstrap_envelope"),
    ("_provision_ipc_apparmor_and_spawn_runner", "resolve_plugin_apparmor_rules"),
])
def test_every_call_on_the_unconditional_path_passes_the_root(
    method: str, callee: str,
) -> None:
    """A later edit that adds a second call without the keyword is caught
    here, whether or not a behavioural test happens to reach it.  The source
    is read from beside this file (not ``inspect.getsource``), so the
    reversion meta-guard's sandbox copy is what gets parsed."""
    source = (Path(__file__).resolve().parents[1] / "session_manager.py").read_text()
    fn = _method(ast.parse(source), "SessionManager", method)
    calls = _calls(fn, callee)
    assert calls, f"{method} no longer calls {callee}"
    for call in calls:
        keywords = {kw.arg for kw in call.keywords}
        assert "managed_workspace_root" in keywords, (
            f"{method} calls {callee} without managed_workspace_root "
            f"(line {call.lineno}): the #1225 / #1274 defaults go silently "
            "off for WUI sessions"
        )
