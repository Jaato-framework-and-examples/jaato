"""The daemon carries an SELinux boundary from provisioning to the runner (2b).

The decision to pass the handle explicitly (not a stash on the server)
means each of these is a value a test can hand in and read back:

* the daemon selects SELinux once, refuses to start when it must, and hands
  the backend to both transports;
* IPC and WS provision with it, call the session confined, and record
  ``selinux`` / ``soft``;
* an SELinux session never takes a pool slot, which cannot enter the domain;
* the cold-spawn child sets the exec context last, after the privilege
  drop, and tells the runner which context it entered;
* the envelope carries the descriptor and the boundary id, and the runner
  keys its tmpdir and private ``/tmp`` on it;
* a cold-spawned runner confirms its domain instead of ``aa_change_profile``.

No kernel: what is pinned is what the daemon asks for.
"""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from jaato_server.server import runner_spawn, runner_spawner, websocket
from jaato_server.server.confinement import ConfinementHandle
from jaato_server.server.runner import session as runner_session
from jaato_server.server.session_manager import SessionManager
from jaato_server.shared.session_envelope import SessionInitEnvelope
from jaato_server.shared.tests.reversion import Reversion

_SPAWN = "jaato-server/jaato_server/server/runner_spawn.py"
_SPAWNER = "jaato-server/jaato_server/server/runner_spawner.py"
_SM = "jaato-server/jaato_server/server/session_manager.py"
_WS = "jaato-server/jaato_server/server/websocket.py"
_RUNNER = "jaato-server/jaato_server/server/runner/session.py"
_RUNNER_MAIN = "jaato-server/jaato_server/server/runner/__main__.py"

REVERSIONS = [
    Reversion(
        target=_SPAWN,
        find="            and cgroup_attach is None and not is_selinux(confinement))\n",
        replace="            and cgroup_attach is None)\n",
        test="test_an_selinux_session_never_takes_a_pool_slot",
        because="a pool slot is threaded and cannot enter an SELinux domain; "
                "the runner would refuse every such session",
    ),
    Reversion(
        target=_SPAWNER,
        find="        _set_exec_context_in_child(exec_context)\n",
        replace="",
        test="test_the_child_sets_the_exec_context_after_the_drop_and_before_exec",
        because="the runner would exec in the daemon's domain, unconfined",
    ),
    Reversion(
        target=_SPAWNER,
        find="            env[\"JAATO_RUNNER_SELINUX_LABEL\"] = selinux_label\n",
        replace="            pass\n",
        test="test_the_runner_is_told_which_context_it_entered",
        because="a cold-spawned runner with no AppArmor profile would refuse "
                "to start, or try aa_change_profile under SELinux",
    ),
    Reversion(
        target=_SM,
        find="        if backend is None:\n            return (*self._provision_apparmor_for_session(**kwargs), None)\n",
        replace="        if True:\n            return (*self._provision_apparmor_for_session(**kwargs), None)\n",
        test="test_ipc_provisions_with_the_selected_selinux_backend",
        because="an SELinux daemon would provision IPC sessions through "
                "AppArmor, find it unavailable and run them unconfined",
    ),
    Reversion(
        target=_SM,
        find="        if backend is not None:\n            return bool(backend.is_available())\n        return self._apparmor_available()\n",
        replace="        return self._apparmor_available()\n",
        test="test_ipc_requires_confinement_on_an_selinux_host",
        because="a failed SELinux provision would read as an unconfined host "
                "and spawn a runner with no boundary (#1253)",
    ),
    Reversion(
        target=_WS,
        find="    if selinux is not None:\n        from jaato_server.server.runner_spawn import stashed_private_tmp\n",
        replace="    if False:\n        from jaato_server.server.runner_spawn import stashed_private_tmp\n",
        test="test_ws_records_the_selinux_mode",
        because="every WS session on an SELinux host would be recorded soft "
                "(or, with no manager, not at all)",
    ),
    Reversion(
        target=_RUNNER,
        find="        return True, conf.confinement_id or None\n",
        replace="        return True, None\n",
        test="test_the_runner_keys_its_tmpdir_on_the_selinux_boundary",
        because="TMPDIR would be /tmp/jaato-<sid>, a directory the daemon "
                "neither created nor labelled jaato_tmp_t",
    ),
    Reversion(
        target=_RUNNER_MAIN,
        find="    if selinux_label:\n        from . import lsm_confine\n",
        replace="    if False:\n        from . import lsm_confine\n",
        test="test_a_cold_spawned_runner_confirms_its_domain",
        because="the runner would call aa_change_profile with an empty name "
                "on an SELinux host",
    ),
]

_RUNNER_CTX = "unconfined_u:unconfined_r:jaato_runner_t:s0:c1,c2"
_CHILD_CTX = "unconfined_u:unconfined_r:jaato_child_t:s0:c1,c2"


def _handle(backend="selinux", complain=False):
    return ConfinementHandle(backend=backend, label=_RUNNER_CTX,
                             confinement_id="ws-0123", child_label=_CHILD_CTX,
                             complain=complain)


# ------------------------------------------------------------------ spawn


def test_an_selinux_session_never_takes_a_pool_slot(monkeypatch):
    monkeypatch.delenv("JAATO_RUNNER_POOL_ENABLED", raising=False)
    pool = object()
    assert runner_spawn._pool_may_serve(pool, None, None) is True
    assert runner_spawn._pool_may_serve(pool, None, _handle()) is False


def test_the_session_tmpdir_is_keyed_on_the_handle():
    assert runner_spawn._confinement_id_of("", _handle()) == "ws-0123"
    assert runner_spawner.RunnerSpawner._session_tmpdir("s1", "", _handle()).endswith(
        "jaato-ws-0123/s1")


def test_the_runner_is_told_which_context_it_entered():
    env = runner_spawner.RunnerSpawner()._build_env(
        profile_name="", session_id="s1", workspace_path="/w", log_path=None,
        max_output_chars=None, tool_timeout_seconds=None, disable_confine=False,
        tmpdir="/tmp/jaato-ws-0123/s1", selinux_label=_RUNNER_CTX)
    assert env["JAATO_RUNNER_SELINUX_LABEL"] == _RUNNER_CTX
    assert "JAATO_RUNNER_DISABLE_CONFINE" not in env
    assert env["TMPDIR"] == "/tmp/jaato-ws-0123/s1"


def test_the_exec_context_write_is_the_runner_label(tmp_path):
    attr = tmp_path / "exec"
    attr.write_text("")
    with patch.object(runner_spawner, "_ATTR_EXEC", str(attr)):
        runner_spawner._set_exec_context_in_child(_RUNNER_CTX)
        runner_spawner._set_exec_context_in_child(None)
    assert attr.read_text() == _RUNNER_CTX


def test_the_child_sets_the_exec_context_after_the_drop_and_before_exec():
    tree = ast.parse(Path(runner_spawner.__file__).read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_exec_runner")
    order = [
        (c.lineno, getattr(c.func, "id", None) or getattr(c.func, "attr", None))
        for c in ast.walk(fn) if isinstance(c, ast.Call)
    ]
    names = [name for _, name in sorted(order)]
    assert "_set_exec_context_in_child" in names
    i = names.index("_set_exec_context_in_child")
    assert names.index("_drop_privileges_in_child") < i < names.index("execvpe")


def test_an_selinux_spawn_needs_no_profile_name(tmp_path):
    spawner = runner_spawner.RunnerSpawner()
    with patch.object(spawner, "_assert_daemon_unconfined"), \
         patch.object(runner_spawner.RunnerSpawner, "_session_tmpdir",
                      staticmethod(lambda sid, p="", c=None: str(tmp_path / "t"))), \
         patch("os.fork", side_effect=RuntimeError("stop-before-fork")):
        with pytest.raises(RuntimeError, match="stop-before-fork"):
            spawner.spawn(profile_name="", session_id="s1", workspace_path=str(tmp_path),
                          confinement=_handle())
        with pytest.raises(ValueError):
            spawner.spawn(profile_name="", session_id="s1", workspace_path=str(tmp_path))


# ------------------------------------------------------------------- IPC


class _Backend:
    def __init__(self, handle):
        self.handle = handle
        self.boundaries = []

    def is_available(self):
        return True

    def provision(self, session_id, boundary):
        self.boundaries.append(boundary)
        return self.handle


def _manager(backend):
    sm = SessionManager.__new__(SessionManager)
    sm.set_selinux_backend(backend)
    sm._notify_apparmor = MagicMock()
    sm._managed_workspace_root_for_spawn = lambda: "/srv/ws"
    return sm


def test_ipc_provisions_with_the_selected_selinux_backend():
    backend = _Backend(_handle())
    sm = _manager(backend)
    name, mode, handle = sm._provision_session_boundary(
        session_id="s1", workspace_path="/srv/ws/a", client_id="c",
        config_root=None, env_file=None, requested_fragments=None,
        plugin_rules=None, private_tmp_dir=None)
    assert (name, mode, handle) == ("", "selinux", backend.handle)
    assert backend.boundaries[0].managed is True


def test_a_permissive_domain_is_recorded_as_such():
    sm = _manager(_Backend(_handle(complain=True)))
    _, mode, _ = sm._provision_session_boundary(
        session_id="s1", workspace_path="/home/u/repo", client_id="c")
    assert mode == "selinux-permissive"


def test_a_failed_selinux_provision_is_soft_and_said():
    sm = _manager(_Backend(None))
    name, mode, handle = sm._provision_session_boundary(
        session_id="s1", workspace_path="/w", client_id="c")
    assert (name, mode, handle) == ("", "soft", None)
    assert sm._notify_apparmor.call_args.kwargs["lsm"] == "selinux"


def test_ipc_requires_confinement_on_an_selinux_host():
    sm = _manager(_Backend(None))
    sm._apparmor_manager = None
    assert sm._kernel_confinement_available() is True


# -------------------------------------------------------------------- WS


def _ws_server(backend):
    return SimpleNamespace(_selinux_backend=backend, _apparmor=None)


def test_ws_provisions_a_managed_boundary_with_selinux():
    backend = _Backend(_handle())
    assert websocket._ws_confinement_available(_ws_server(backend)) is True
    handle = websocket._provision_ws_boundary(_ws_server(backend), "s1", "/srv/ws/a", None, None)
    assert handle is backend.handle
    assert backend.boundaries[0].managed is True


def test_ws_records_the_selinux_mode():
    sess = SimpleNamespace(workspace_path="/srv/ws/a", sandbox_mode=None)
    decided = websocket._record_ws_mode_without_apparmor(
        _ws_server(_Backend(_handle())), SimpleNamespace(), "s1", sess)
    assert decided is True and sess.sandbox_mode == "selinux"
    sess.sandbox_mode = None
    websocket._record_ws_mode_without_apparmor(
        _ws_server(_Backend(None)), SimpleNamespace(), "s1", sess)
    assert sess.sandbox_mode == "soft"


def test_ws_keeps_the_apparmor_path_when_selinux_is_not_selected():
    apparmor = MagicMock()
    apparmor.is_available.return_value = True
    ws = SimpleNamespace(_selinux_backend=None, _apparmor=apparmor)
    sess = SimpleNamespace(workspace_path="/srv/ws/a", sandbox_mode=None)
    assert websocket._record_ws_mode_without_apparmor(ws, SimpleNamespace(), "s1", sess) is False
    assert sess.sandbox_mode is None


# ---------------------------------------------------------------- daemon


def test_the_daemon_refuses_an_unknown_backend(monkeypatch):
    from jaato_server.server.__main__ import JaatoDaemon

    monkeypatch.setenv("JAATO_CONFINEMENT", "bogus")
    daemon = JaatoDaemon.__new__(JaatoDaemon)
    with pytest.raises(SystemExit, match="refusing to start"):
        daemon._select_confinement_backend()


def test_the_daemon_hands_a_selected_selinux_backend_on(monkeypatch):
    from jaato_server.server.__main__ import JaatoDaemon
    from jaato_server.server.confinement import selinux

    monkeypatch.setenv("JAATO_CONFINEMENT", "selinux")
    monkeypatch.setattr(selinux.SELinuxBackend, "is_available", lambda self: True)
    daemon = JaatoDaemon.__new__(JaatoDaemon)
    daemon._select_confinement_backend()
    assert isinstance(daemon._selinux_backend, selinux.SELinuxBackend)


# ---------------------------------------------------------------- runner


def _envelope(**kw):
    return SessionInitEnvelope(session_id="s1", workspace_path="/tmp/ws",
                               provider_name="echo", model_name="m", **kw)


def test_the_runner_keys_its_tmpdir_on_the_selinux_boundary():
    env = _envelope(profile_name="", confinement={
        "backend": "selinux", "label": _RUNNER_CTX, "child_label": _CHILD_CTX,
        "confinement_id": "ws-0123"})
    assert runner_session._runner_boundary(env) == (True, "ws-0123")
    assert runner_session._runner_boundary(_envelope(profile_name="")) == (False, None)


def test_a_cold_spawned_runner_confirms_its_domain(monkeypatch):
    from jaato_server.server.runner import __main__ as runner_main
    from jaato_server.server.runner import bootstrap, lsm_confine

    monkeypatch.setenv("JAATO_RUNNER_SELINUX_LABEL", _RUNNER_CTX)
    with patch.object(lsm_confine, "self_confine") as confirm, \
         patch.object(bootstrap, "confine_to_profile") as aa:
        runner_main._cold_spawn_confine("", False, MagicMock())
    confirm.assert_called_once_with("selinux", _RUNNER_CTX)
    aa.assert_not_called()
