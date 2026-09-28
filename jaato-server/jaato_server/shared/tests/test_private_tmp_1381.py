"""A private /tmp per workspace (#1381, step 1).

A confined session's profile granted only its session tmpdir under ``/tmp``
(#1171), so every habitual ``/tmp/x`` a model wrote cost a refused call.
A confined runner of a daemon-managed workspace now unshares a mount
namespace and binds ``<ws>/.tmp`` over ``/tmp`` and ``/var/tmp`` BEFORE it
confines, and its profile grants ``/tmp/**`` -- only then.

What this guards, each half of the fail-closed contract:

* the ``/tmp`` grant is rendered only for a boundary that asked for it, and
  it is part of the confinement id (so of the pool slot key);
* the default resolution: on for daemon-managed workspaces, opt-in for a
  user's own checkout, off with one WARNING where the host cannot;
* the runner REFUSES to start when the namespace cannot be set up, on both
  the pool-slot path (``session.bootstrap``) and the cold-spawn path (the
  forked child before ``exec``);
* an unconfined runner never enters a namespace;
* ``TMPDIR`` and the ``sandbox_utils`` allowance follow the namespace;
* housekeeping: the Files panel, the ``.gitignore`` block, the runtime
  aspect.

CI has no AppArmor kernel, so profile names are strings here.  The one test
that really unshares runs in a subprocess and is skipped where the host
refuses ``CAP_SYS_ADMIN``.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from jaato_server.server.apparmor import AppArmorManager
from jaato_server.server.confinement_id import profile_name_for
from jaato_server.server.runner import session as runner_session
from jaato_server.server.runner.session import BootstrapError
from jaato_server.server.runner_pool import PoolSlot, SlotKey
from jaato_server.server.runner_spawn import (
    private_tmp_kwargs, session_private_tmp,
)
from jaato_server.server.runner_spawner import PRIVATE_TMP_EXIT_CODE
from jaato_server.server.workspace_monitor import _WORKSPACE_HOME_IGNORE
from jaato_server.shared import private_tmp
from jaato_server.shared.plugins import sandbox_utils
from jaato_server.shared.plugins.environment import runtime
from jaato_server.shared.scaffold import gitignore
from jaato_server.shared.session_envelope import SessionInitEnvelope
from jaato_server.shared.tests.reversion import Reversion

_APPARMOR = "jaato-server/jaato_server/server/apparmor.py"
_PRIVATE = "jaato-server/jaato_server/shared/private_tmp.py"
_RS = "jaato-server/jaato_server/server/runner/session.py"
_SPAWN = "jaato-server/jaato_server/server/runner_spawn.py"
_GITIGNORE = "jaato-server/jaato_server/shared/scaffold/gitignore.py"

REVERSIONS = [
    Reversion(
        target=_APPARMOR,
        find='            private_tmp_rules=self._private_tmp_rules(private_tmp_dir, "  "),\n',
        replace='            private_tmp_rules=self._private_tmp_rules(None, "  "),\n',
        test="test_the_tmp_grant_is_rendered_in_base_tool_hat_and_child",
        because="the base body of a private-/tmp boundary loses its /tmp grant",
    ),
    Reversion(
        target=_APPARMOR,
        find=(
            "                plugin_rules=plugin_rules,\n"
            "                private_tmp_dir=private_tmp_dir,\n"
            "            )\n"
        ),
        replace=(
            "                plugin_rules=plugin_rules,\n"
            "            )\n"
        ),
        test="test_a_private_tmp_boundary_has_its_own_confinement_id",
        because=(
            "two boundaries differing only in /tmp share a profile name, so a "
            "pool slot in one namespace is handed to a session wanting the other"
        ),
    ),
    Reversion(
        target=_PRIVATE,
        find="    return _is_under(workspace_path, managed_workspace_root)\n",
        replace="    return True\n",
        test="test_a_users_own_checkout_is_opt_in",
        because="a user asking the agent to read /tmp/log.txt sees .tmp instead",
    ),
    Reversion(
        target=_RS,
        find="    except (PrivateTmpError, OSError) as exc:\n",
        replace="    except ZeroDivisionError as exc:\n",
        test="test_a_namespace_that_cannot_be_set_up_refuses_the_bootstrap",
        because="the runner starts under a /tmp grant against the host's /tmp",
    ),
    Reversion(
        target=_SPAWN,
        find=(
            "    if not profile_name:\n"
            "        return None\n"
            "    return stashed_private_tmp(server)\n"
        ),
        replace="    return stashed_private_tmp(server)\n",
        test="test_an_unconfined_runner_never_enters_a_namespace",
        because=(
            "an unconfined slot enters one workspace's namespace and is later "
            "handed to another workspace's unconfined session"
        ),
    ),
    Reversion(
        target=_RS,
        find="        sandbox_utils.set_temp_roots(list(PRIVATE_TMP_TARGETS))\n",
        replace="        pass\n",
        test="test_a_private_tmp_runner_allows_and_pins_tmp",
        because="cli refuses the /tmp write the kernel would now allow",
    ),
    Reversion(
        target=_GITIGNORE,
        find='    ("/.tmp/", ".tmp/scratch", "the private /tmp (<ws>/.tmp, #1381)"),\n',
        replace="",
        test="test_the_gitignore_block_ignores_the_private_tmp",
        because="what sessions write to /tmp is one git add -A from a commit",
    ),
]

WS = "/srv/workspaces/demo"
TMP_DIR = f"{WS}/.tmp"


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    private_tmp._reset_for_tests()
    monkeypatch.setattr(sandbox_utils, "SYSTEM_TEMP_PATHS",
                        list(sandbox_utils.SYSTEM_TEMP_PATHS))
    yield
    private_tmp._reset_for_tests()


def _render(private_tmp_dir=None):
    return AppArmorManager(workspace_root="/srv/workspaces")._render_profile(
        "demo-abc", WS, private_tmp_dir=private_tmp_dir)


# ---------------------------------------------------------------- rendering

def test_the_tmp_grant_is_rendered_in_base_tool_hat_and_child():
    body = _render(TMP_DIR)
    # base, tool_hat and //child: three bodies, each with both grants.
    assert body.count("/tmp/**        rwkl,") == 3
    assert body.count("/var/tmp/**    rwkl,") == 3
    assert TMP_DIR in body


def test_no_tmp_grant_without_a_private_tmp():
    """The control: the broad grant never reaches a host-/tmp boundary."""
    body = _render(None)
    assert "/tmp/**        rwkl," not in body
    assert "/var/tmp/**" not in body
    assert body.count("# (no private /tmp for this boundary)") == 3


def test_a_private_tmp_boundary_has_its_own_confinement_id():
    mgr = AppArmorManager(workspace_root="/srv/workspaces")
    plain = mgr.confinement_id_for_boundary(WS)
    private = mgr.confinement_id_for_boundary(WS, private_tmp_dir=TMP_DIR)
    assert plain != private


def test_private_tmp_kwargs_leave_an_off_session_call_unchanged():
    assert private_tmp_kwargs(None) == {}
    assert private_tmp_kwargs(TMP_DIR) == {"private_tmp_dir": TMP_DIR}


# ---------------------------------------------------------------- slot key

def test_a_slot_in_one_namespace_is_not_handed_to_the_other_boundary():
    """#1033's rule: a confined slot cannot re-mount, so the key must keep
    it away from a session wanting the other /tmp.  The profile name --
    derived from the body, which names the directory -- is what does it."""
    mgr = AppArmorManager(workspace_root="/srv/workspaces")
    private_name = profile_name_for(
        mgr.confinement_id_for_boundary(WS, private_tmp_dir=TMP_DIR))
    plain_name = profile_name_for(mgr.confinement_id_for_boundary(WS))

    slot = PoolSlot(pid=1, sock=None)
    SlotKey.build(workspace_root=WS, profile_name=private_name).stamp(slot)

    assert SlotKey.build(workspace_root=WS, profile_name=private_name) \
        .accepts_unaffined(slot)
    assert not SlotKey.build(workspace_root=WS, profile_name=plain_name) \
        .accepts_unaffined(slot)


# ---------------------------------------------------------------- defaults

def test_a_daemon_managed_workspace_is_on_by_default(tmp_path):
    root = tmp_path / "root"
    ws = root / "ws"
    ws.mkdir(parents=True)
    assert private_tmp.private_tmp_requested(None, str(ws), str(root))


def test_a_users_own_checkout_is_opt_in(tmp_path):
    ws = tmp_path / "checkout"
    ws.mkdir()
    assert not private_tmp.private_tmp_requested(None, str(ws), None)
    assert private_tmp.private_tmp_requested(
        {"private_tmp": True}, str(ws), None)


def test_an_explicit_value_wins_and_a_typo_is_off(tmp_path, caplog):
    root = tmp_path / "root"
    ws = root / "ws"
    ws.mkdir(parents=True)
    assert not private_tmp.private_tmp_requested(
        {"private_tmp": False}, str(ws), str(root))
    with caplog.at_level(logging.WARNING):
        assert not private_tmp.private_tmp_requested(
            {"private_tmp": "yse"}, str(ws), str(root))
    assert "not a boolean" in caplog.text


def test_a_non_root_daemon_is_off_with_one_warning(monkeypatch, caplog):
    monkeypatch.setattr(private_tmp.os, "geteuid", lambda: 1000)
    profile = SimpleNamespace(plugin_configs={"cli": {"private_tmp": True}})
    with caplog.at_level(logging.WARNING):
        assert private_tmp.resolve_private_tmp(profile, WS, None) is None
        assert private_tmp.resolve_private_tmp(profile, WS, None) is None
    assert caplog.text.count("asked for a private /tmp") == 1


def test_a_root_daemon_resolves_the_workspace_tmp(monkeypatch):
    monkeypatch.setattr(private_tmp.os, "geteuid", lambda: 0)
    monkeypatch.setattr(private_tmp.sys, "platform", "linux")
    profile = SimpleNamespace(plugin_configs={"cli": {"private_tmp": True}})
    assert private_tmp.resolve_private_tmp(profile, WS, None) == \
        os.path.join(os.path.realpath(WS), ".tmp")


def test_a_workspace_under_tmp_cannot_have_one(monkeypatch, tmp_path):
    """Binding over /tmp would hide a workspace that lives there."""
    monkeypatch.setattr(private_tmp.os, "geteuid", lambda: 0)
    ws = tempfile.mkdtemp(dir="/tmp")
    try:
        profile = SimpleNamespace(plugin_configs={"cli": {"private_tmp": True}})
        assert private_tmp.resolve_private_tmp(profile, ws, None) is None
    finally:
        os.rmdir(ws)


# ---------------------------------------------------------------- refusal

def _envelope(profile_name="jaato-ws-demo-abc", tmp_dir=TMP_DIR):
    return SimpleNamespace(session_id="s1", profile_name=profile_name,
                           private_tmp_dir=tmp_dir)


def test_a_namespace_that_cannot_be_set_up_refuses_the_bootstrap(monkeypatch):
    def refuse(_):
        raise private_tmp.PrivateTmpError("unshare(CLONE_NEWNS) failed: EPERM")
    monkeypatch.setattr(runner_session, "enter_private_tmp", refuse)
    with pytest.raises(BootstrapError) as info:
        runner_session._enter_private_tmp(_envelope())
    assert info.value.stage == "private_tmp"
    assert "Refusing to start" in str(info.value)


def test_a_missing_directory_is_a_refusal_not_a_fallback(tmp_path):
    with pytest.raises(private_tmp.PrivateTmpError):
        private_tmp.enter_private_tmp(str(tmp_path / "nope"))


def test_the_cold_spawn_child_exits_naming_the_cause():
    """The forked child before exec: it cannot raise into the runner, so it
    exits with a status of its own and says why on stderr."""
    code = (
        "from jaato_server.server.runner_spawner import "
        "_enter_private_tmp_in_child as f; f('/nonexistent/.tmp')"
    )
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == PRIVATE_TMP_EXIT_CODE
    assert "refusing to start" in proc.stderr
    assert "/nonexistent/.tmp" in proc.stderr


def test_an_unconfined_runner_never_enters_a_namespace(monkeypatch):
    server = SimpleNamespace(_private_tmp_dir=TMP_DIR)
    assert session_private_tmp(server, "") is None
    assert session_private_tmp(server, "jaato-ws-demo-abc") == TMP_DIR
    entered = []
    monkeypatch.setattr(runner_session, "enter_private_tmp", entered.append)
    runner_session._enter_private_tmp(_envelope(profile_name=""))
    assert entered == [None]


def test_a_mock_server_never_reaches_a_mount():
    from unittest.mock import MagicMock
    assert session_private_tmp(MagicMock(), "jaato-ws-demo-abc") is None


def test_the_envelope_round_trips_and_refuses_a_relative_path():
    base = dict(session_id="s1", workspace_path=WS, profile_name="p",
                provider_name="echo", model_name="m", plugins=[],
                plugin_configs={}, system_instructions=None)
    env = SessionInitEnvelope(**base, private_tmp_dir=TMP_DIR)
    assert SessionInitEnvelope.from_dict(env.to_dict()).private_tmp_dir == TMP_DIR
    assert SessionInitEnvelope.from_dict(
        SessionInitEnvelope(**base).to_dict()).private_tmp_dir is None
    with pytest.raises(Exception):
        SessionInitEnvelope(**base, private_tmp_dir=".tmp")


def _writable_dir_outside_tmp():
    """A directory to host a throwaway workspace, not under /tmp.

    The suite's HOME and ``tmp_path`` are under /tmp, and a workspace there
    is exactly the one a private /tmp cannot serve.
    """
    import pwd
    for cand in (pwd.getpwuid(os.getuid()).pw_dir, "/dev/shm", "/run"):
        if cand and os.access(cand, os.W_OK) \
                and not private_tmp.under_a_target(cand):
            return cand
    return None


def test_the_namespace_really_binds_tmp():
    """The real thing, in a child process: /tmp and /var/tmp become the
    workspace's .tmp, and the host's /tmp is untouched."""
    parent = _writable_dir_outside_tmp()
    if parent is None:
        pytest.skip("no writable directory outside /tmp")
    ws = Path(tempfile.mkdtemp(dir=parent))
    try:
        (ws / ".tmp").mkdir()
        marker = f"jaato-1381-{os.getpid()}"
        code = (
            "import sys\n"
            "from jaato_server.shared import private_tmp as p\n"
            "try:\n"
            "    p.enter_private_tmp(sys.argv[1])\n"
            "except p.PrivateTmpError as e:\n"
            "    print('SKIP', e); sys.exit(0)\n"
            f"open('/tmp/{marker}', 'w').write('x')\n"
            "print('OK', p.describe()[0])\n"
        )
        proc = subprocess.run([sys.executable, "-c", code, str(ws / ".tmp")],
                              capture_output=True, text=True, timeout=60)
        if proc.stdout.startswith("SKIP"):
            pytest.skip(f"this host refuses a mount namespace: {proc.stdout}")
        assert proc.stdout.strip() == "OK True", proc.stderr
        assert (ws / ".tmp" / marker).exists()
        assert not Path("/tmp", marker).exists()
    finally:
        subprocess.run(["rm", "-rf", str(ws)], check=False)


# ---------------------------------------------------------------- TMPDIR

def test_a_private_tmp_runner_allows_and_pins_tmp(monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    monkeypatch.setenv("TMPDIR", os.environ.get("TMPDIR", "/tmp"))
    runner_session._pin_session_tmpdir(_envelope())
    assert os.environ["TMPDIR"] == "/tmp"
    assert tempfile.tempdir == "/tmp"
    assert sandbox_utils.is_under_temp_path("/tmp/x")
    assert sandbox_utils.is_under_temp_path("/var/tmp/x")


# ---------------------------------------------------------------- housekeeping

def test_the_files_panel_ignores_the_private_tmp():
    assert ".tmp/" in _WORKSPACE_HOME_IGNORE


def test_the_gitignore_block_ignores_the_private_tmp(tmp_path):
    (tmp_path / ".gitignore").write_text(gitignore.render_block())
    assert gitignore.assess(tmp_path).unignored_scratch == ()
    (tmp_path / ".gitignore").write_text("node_modules/\n")
    assert gitignore.assess(tmp_path).unignored_scratch


def test_the_runtime_aspect_says_whether_tmp_is_private(monkeypatch):
    assert runtime.private_tmp_report()["private"] is False
    monkeypatch.setattr(private_tmp, "_active_dir", TMP_DIR)
    monkeypatch.setattr(private_tmp, "private_tmp_in_effect", lambda d: True)
    report = runtime.private_tmp_report()
    assert report == {**report, "private": True, "backing_dir": TMP_DIR}
