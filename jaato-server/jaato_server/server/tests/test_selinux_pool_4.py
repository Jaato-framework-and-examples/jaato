"""SELinux sessions are served by pool slots forked into their boundary (phase 4).

A virgin pool slot is threaded, and SELinux refuses ``setcon`` in a
threaded process (phase 0), so it can never enter a session's domain.  A
freshly forked child has one thread, so the template forks a slot FOR the
session, and the child enters the private ``/tmp``, the uid and the
domain before it starts a thread (selinux-backend.md §7.2).  The slot then
returns to the pool and fits only that boundary.

No kernel: what is pinned is what is asked for, and in which order.
"""

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jaato_server.server import runner_spawn, runner_spawner
from jaato_server.server.confinement import ConfinementHandle
from jaato_server.server.runner import __main__ as runner_main
from jaato_server.server.runner_pool import PoolManager, PoolSlot, SlotKey
from jaato_server.server.runner_template import _fork_slot_line
from jaato_server.shared.privilege_drop import RunnerUser
from jaato_server.shared.tests.reversion import Reversion

_POOL = "jaato-server/jaato_server/server/runner_pool.py"
_SPAWN = "jaato-server/jaato_server/server/runner_spawn.py"
_SPAWNER = "jaato-server/jaato_server/server/runner_spawner.py"
_MAIN = "jaato-server/jaato_server/server/runner/__main__.py"

REVERSIONS = [
    Reversion(
        target=_POOL,
        find="            return self.selinux_boundary is None\n",
        replace="            return True\n",
        test="test_a_virgin_slot_never_serves_an_selinux_session",
        because="a threaded slot would be handed an SELinux session and "
                "refused at bootstrap (it cannot setcon)",
    ),
    Reversion(
        target=_POOL,
        find="            and (getattr(slot, \"selinux_boundary\", None) or None) == self.selinux_boundary\n",
        replace="",
        test="test_a_slot_serves_only_the_boundary_it_was_forked_into",
        because="a slot in one workspace's level would be handed another's",
    ),
    Reversion(
        target=_SPAWN,
        find="    return pool_manager.fork_slot_into(SlotKey.build(**fields), entry)\n",
        replace="    return None\n",
        test="test_an_selinux_miss_forks_a_slot_into_the_boundary",
        because="every SELinux session would cold-spawn, as before phase 4",
    ),
    Reversion(
        target=_MAIN,
        find="        if entry is not None:\n            enter_slot_boundary_in_child(entry)\n",
        replace="",
        test="test_the_child_enters_the_boundary_before_slot_mode",
        because="a slot forked for an SELinux session would serve it "
                "outside its domain",
    ),
    Reversion(
        target=_SPAWNER,
        find="        if actual != context:\n",
        replace="        if False:\n",
        test="test_a_context_the_kernel_did_not_apply_exits_the_child",
        because="a slot the kernel left outside the domain would serve",
    ),
    Reversion(
        target=_SPAWNER,
        find="    _redirect_output_in_child(entry[\"log_path\"])\n",
        replace="",
        test="test_the_child_enters_tmp_then_uid_then_domain",
        because="a confined slot would hold the daemon's log on fds 1 and 2, "
                "and the bootstrap's first flush would fail the session "
                "(phase 4 kernel run 2)",
    ),
    Reversion(
        target=_SPAWN,
        find="    _ensure_runner_log_dir(log_path)\n",
        replace="",
        test="test_the_spawn_path_makes_the_log_directory_first",
        because="a fresh workspace's first pool-served session would have "
                "no .jaato/logs to log into (phase 4 kernel run 2)",
    ),
    Reversion(
        target=_SPAWNER,
        find="    _enter_private_tmp_in_child(entry.get(\"private_tmp\") or None)\n"
             "    _drop_privileges_in_child(RunnerUser.from_dict(entry.get(\"runner_user\")))\n"
             "    _enter_selinux_domain_in_child(entry[\"context\"])\n",
        replace="    _enter_selinux_domain_in_child(entry[\"context\"])\n"
                "    _enter_private_tmp_in_child(entry.get(\"private_tmp\") or None)\n"
                "    _drop_privileges_in_child(RunnerUser.from_dict(entry.get(\"runner_user\")))\n",
        test="test_the_child_enters_tmp_then_uid_then_domain",
        because="the runner domain holds neither sys_admin nor setuid, so "
                "the mount and the drop must come first (Order A)",
    ),
]

_CTX = "unconfined_u:unconfined_r:jaato_runner_t:s0:c1,c2"
_USER = RunnerUser(uid=1000, gid=1000, groups=(1000,), username="u",
                   home="/home/u", source="workspace-owner")


def _handle():
    return ConfinementHandle(backend="selinux", label=_CTX, confinement_id="ws-b1",
                             child_label=_CTX.replace("runner", "child"))


def _pool(*slots):
    pool = PoolManager(template_manager=MagicMock(), target_size=0)
    pool._idle_slots.extend(slots)
    return pool


class _Exited(Exception):
    pass


def _virgin(pid=1):
    return PoolSlot(pid=pid, sock=MagicMock())


# ------------------------------------------------------------------ the key


def test_a_virgin_slot_never_serves_an_selinux_session():
    pool = _pool(_virgin())
    assert pool.acquire_slot(workspace_root="/w", selinux_boundary="ws-b1") is None
    # ...and still serves an unconfined one.
    assert pool.acquire_slot(workspace_root="/w") is not None


def test_a_slot_serves_only_the_boundary_it_was_forked_into():
    slot = _virgin()
    SlotKey.build(workspace_root="/w", selinux_boundary="ws-b1").stamp(slot)
    pool = _pool(slot)
    assert pool.acquire_slot(workspace_root="/w", selinux_boundary="ws-OTHER") is None
    assert pool.acquire_slot(workspace_root="/w") is None
    assert pool.acquire_slot(workspace_root="/w", selinux_boundary="ws-b1") is slot


def test_fork_slot_into_asks_the_template_and_stamps_the_key():
    pool = _pool()
    pool._template_manager.request_fork_slot.return_value = (42, MagicMock())
    key = SlotKey.build(workspace_root="/w", selinux_boundary="ws-b1", runner_uid=1000)
    entry = {"context": _CTX}
    slot = pool.fork_slot_into(key, entry)
    pool._template_manager.request_fork_slot.assert_called_once_with(entry=entry)
    assert SlotKey.of_slot(slot) == key and slot.has_served
    assert pool.get_telemetry()["pool_selinux_fork_total"] == 1


def test_fork_slot_on_demand_forks_a_virgin_slot_and_stamps_the_key():
    """The plain miss: no boundary entry, the key stamped, counted."""
    pool = _pool()
    pool._template_manager.request_fork_slot.return_value = (43, MagicMock())
    key = SlotKey.build(cascade_driver_id="c1", workspace_root="/w")
    slot = pool.fork_slot_on_demand(key)
    pool._template_manager.request_fork_slot.assert_called_once_with()
    assert SlotKey.of_slot(slot) == key and slot.pid == 43
    assert pool.get_telemetry()["pool_demand_fork_total"] == 1


def test_a_failed_demand_fork_is_a_cold_spawn():
    pool = _pool()
    pool._template_manager.request_fork_slot.return_value = None
    assert pool.fork_slot_on_demand(SlotKey.build(workspace_root="/w")) is None
    assert pool.get_telemetry()["pool_demand_fork_failures_total"] == 1


def test_a_failed_fork_is_a_miss():
    pool = _pool()
    pool._template_manager.request_fork_slot.return_value = None
    assert pool.fork_slot_into(SlotKey.build(selinux_boundary="ws-b1"), {"context": _CTX}) is None
    assert pool.get_telemetry()["pool_selinux_fork_failures_total"] == 1


# ------------------------------------------------------------------ spawn


def _server():
    return SimpleNamespace(config_root="/w/.jaato", _private_tmp_dir="/w/.tmp")


def test_an_selinux_session_may_use_the_pool(monkeypatch):
    monkeypatch.delenv("JAATO_RUNNER_POOL_ENABLED", raising=False)
    assert runner_spawn._pool_may_serve(object(), None) is True
    assert runner_spawn._pool_may_serve(object(), object()) is False


def test_an_selinux_miss_forks_a_slot_into_the_boundary():
    pool = MagicMock()
    pool.acquire_slot.return_value = None
    runner_spawn._acquire_pool_slot(
        pool, _server(), cascade_driver_id=None, workspace_path="/w",
        profile_name="", runner_user=_USER, confinement=_handle(),
        log_path="/w/.jaato/logs/runner-s1.log")
    key, entry = pool.fork_slot_into.call_args.args
    assert key.selinux_boundary == "ws-b1" and key.runner_uid == 1000
    assert entry == {"context": _CTX, "log_path": "/w/.jaato/logs/runner-s1.log",
                     "private_tmp": "/w/.tmp", "runner_user": _USER.to_dict()}


def test_an_unconfined_miss_forks_a_virgin_slot_not_into_a_boundary():
    pool = MagicMock()
    pool.acquire_slot.return_value = None
    runner_spawn._acquire_pool_slot(
        pool, _server(), cascade_driver_id=None, workspace_path="/w",
        profile_name="", runner_user=None, confinement=None)
    pool.fork_slot_into.assert_not_called()
    pool.fork_slot_on_demand.assert_called_once()


# ------------------------------------------------------------------ template


def test_the_template_line_round_trips():
    entry = {"context": _CTX, "log_path": "/w/l.log", "private_tmp": None,
             "runner_user": None}
    line = _fork_slot_line(entry).decode().strip()
    assert runner_main._fork_slot_entry(line) == entry
    assert runner_main._fork_slot_entry(_fork_slot_line(None).decode().strip()) is None


@pytest.mark.parametrize("cmd", ["FORK_SLOT {bad", "FORK_SLOT []", 'FORK_SLOT {"x":1}',
                                 'FORK_SLOT {"context":"c"}'])
def test_a_malformed_boundary_is_refused(cmd):
    assert runner_main._fork_slot_entry(cmd) is runner_main._MALFORMED


def test_the_child_enters_the_boundary_before_slot_mode(monkeypatch):
    order = []
    monkeypatch.setattr(os, "fork", lambda: 0)
    monkeypatch.setattr(os, "setsid", lambda: None)
    def exit_(code):
        # The child never returns into the template's branch, which would
        # close a real descriptor.
        raise _Exited(code)

    monkeypatch.setattr(os, "_exit", exit_)
    monkeypatch.setattr(runner_spawner, "enter_slot_boundary_in_child",
                        lambda entry: order.append(("enter", entry["context"])))
    monkeypatch.setattr(runner_main, "_run_slot_mode",
                        lambda fd, log: order.append(("slot", fd)))
    with pytest.raises(_Exited):
        runner_main._handle_fork_slot(MagicMock(), -1, MagicMock(), {"context": _CTX})
    assert order == [("enter", _CTX), ("slot", -1)]


# ------------------------------------------------------------------ child


def test_the_child_enters_tmp_then_uid_then_domain(monkeypatch):
    order = []
    monkeypatch.setattr(runner_spawner, "_redirect_output_in_child",
                        lambda p: order.append(("log", p)))
    monkeypatch.setattr(runner_spawner, "_enter_private_tmp_in_child",
                        lambda d: order.append(("tmp", d)))
    monkeypatch.setattr(runner_spawner, "_drop_privileges_in_child",
                        lambda u: order.append(("uid", u.uid if u else None)))
    monkeypatch.setattr(runner_spawner, "_enter_selinux_domain_in_child",
                        lambda c: order.append(("domain", c)))
    runner_spawner.enter_slot_boundary_in_child(
        {"context": _CTX, "log_path": "/w/l.log", "private_tmp": "/w/.tmp",
         "runner_user": _USER.to_dict()})
    assert order == [("log", "/w/l.log"), ("tmp", "/w/.tmp"), ("uid", 1000),
                     ("domain", _CTX)]


_REDIRECT = """
import sys
from jaato_server.server import runner_spawner
runner_spawner._redirect_output_in_child(sys.argv[1])
print("AFTER", flush=True)
sys.stderr.write("ERR-AFTER\\n")
"""


def test_the_slot_output_goes_to_its_log(tmp_path):
    import subprocess, sys
    log = tmp_path / "runner-s1.log"
    p = subprocess.run([sys.executable, "-c", _REDIRECT, str(log)],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0 and p.stdout == "" and p.stderr == ""
    assert "AFTER" in log.read_text() and "ERR-AFTER" in log.read_text()


def test_a_log_the_slot_cannot_open_exits_it(tmp_path):
    import subprocess, sys
    p = subprocess.run([sys.executable, "-c", _REDIRECT, str(tmp_path / "no" / "x.log")],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == runner_spawner.SLOT_LOG_EXIT_CODE
    assert "could not open its runner log" in p.stderr


def test_the_spawn_path_makes_the_log_directory_first(tmp_path):
    import ast
    from pathlib import Path

    log = tmp_path / ".jaato" / "logs" / "runner-s1.log"
    runner_spawn._ensure_runner_log_dir(str(log))
    assert log.parent.is_dir()
    tree = ast.parse(Path(runner_spawn.__file__).read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "spawn_session_runner")
    calls = [n.func.id for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert calls.index("_ensure_runner_log_dir") < calls.index("_prepare_runner_user")


def _attr(tmp_path, monkeypatch, reads_back):
    attr = tmp_path / "current"
    attr.write_text("")
    monkeypatch.setattr(runner_spawner, "_ATTR_CURRENT", str(attr))
    real_open = open

    def fake_open(path, *a, **kw):
        if str(path) == str(attr) and "r" in (a[0] if a else kw.get("mode", "r")):
            attr.write_text(reads_back)
        return real_open(path, *a, **kw)

    monkeypatch.setattr("builtins.open", fake_open)

    def exit_(code):
        raise _Exited(code)

    monkeypatch.setattr(os, "_exit", exit_)
    return attr


def test_the_child_writes_its_context(tmp_path, monkeypatch):
    _attr(tmp_path, monkeypatch, reads_back=_CTX + "\x00")
    runner_spawner._enter_selinux_domain_in_child(_CTX)


def test_a_context_the_kernel_did_not_apply_exits_the_child(tmp_path, monkeypatch):
    _attr(tmp_path, monkeypatch, reads_back="unconfined_u:unconfined_r:unconfined_t:s0")
    with pytest.raises(_Exited) as exc:
        runner_spawner._enter_selinux_domain_in_child(_CTX)
    assert exc.value.args == (runner_spawner.SELINUX_ENTRY_EXIT_CODE,)
