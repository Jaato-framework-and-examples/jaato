"""A root daemon can run each runner as its user (#1168, step 3).

THE STATE THIS GUARDS AGAINST.  A root daemon forked and exec'd every
runner as root, so every file an agent wrote into a workspace was
root-owned.  The umask (step 2) cannot widen a file created with an
explicit mode (``curated.jsonl`` is ``0600``) nor one written by
``mkstemp`` + ``os.replace``; on the issue, an app owning its workspace as
uid 1001 could not delete the ``.jaato/memories`` its own sessions wrote.
``--runner-uid-policy peer|workspace-owner`` drops the runner to a user.

What is guarded, and why each is its own case:

* **the policy fails closed to the daemon's uid, never open to a guess** —
  a non-root daemon, a connection with no OS principal (WS, #1074), a
  root target;
* **a user who cannot read what a runner imports is REFUSED**, not handed
  a runner that dies on import, and not handed a root runner either — both
  transports turn the refusal into a refused session, never the in-process
  fallback (which would run the model's tools in the root daemon);
* **the order** — cgroup and private ``/tmp`` (both need root) before the
  drop, the drop before ``aa_change_profile`` (Order A: the confined runner
  then holds no capability to become root again).  Checked by recording
  the bootstrap's calls, and by walking the cold-spawn child's AST;
* **a slot that dropped is that uid for life** — ``SlotKey.runner_uid``;
* **what the daemon makes for the runner is handed over, and nothing else
  is taken** — only daemon-owned paths are chowned;
* **the drop itself** is irreversible, proven by ``setuid(0)`` failing.

THE ROOT-ONLY CASES ARE NOT THE LOAD-BEARING ONES.  CI runs as an ordinary
user, where a reversion whose only detector is skipped would read as
decorative to the meta-guard.  So every reversion names a case that runs
without root: the drop's syscalls are substituted, the chown is recorded
rather than performed.  The cases that DO fork a child and drop to
``nobody`` run where the suite is root (this container) and are skipped
cleanly elsewhere; they check what a substitution cannot — that the kernel
agrees.

NOT VERIFIED HERE: an enforcing AppArmor kernel.  Order A was measured on
one in the issue; ``kernel.apparmor_restrict_unprivileged_unconfined=1``
and an unprivileged already-confined re-transition on a reused slot were
not, and are on the PR's checklist.
"""

from __future__ import annotations

import ast
import logging
import os
import pathlib
import stat
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jaato_server.shared.peer_identity import PeerCredentials
from jaato_server.shared.privilege_drop import (
    PrivilegeDropError, RunnerUser, drop_to,
)
from jaato_server.shared.tests.reversion import Reversion


REVERSIONS = [
    Reversion(
        target="jaato-server/jaato_server/server/runner_user.py",
        find="    if policy == POLICY_DAEMON or announce_if_unprivileged(policy):\n",
        replace="    if policy == POLICY_DAEMON:\n",
        test="TestThePolicy::test_a_non_root_daemon_keeps_its_uid",
        because="a non-root daemon asked to drop resolving a user anyway, "
                "so every spawn fails at setgroups instead of the policy "
                "saying once that it cannot apply",
    ),
    Reversion(
        target="jaato-server/jaato_server/server/runner_user.py",
        find="    if uid == 0 or uid == os.getuid():\n",
        replace="    if False:\n",
        test="TestThePolicy::test_a_root_peer_is_not_a_drop_target",
        because="a root peer (or a root-owned workspace) producing a "
                "RunnerUser for uid 0 -- a 'drop' that is none, which "
                "drop_to then refuses and so fails the session",
    ),
    Reversion(
        target="jaato-server/jaato_server/server/runner_user.py",
        find="        if path_reachable_by(p, peer, ACCESS_EXECUTE) is not True\n",
        replace="        if False\n",
        test="TestThePolicy::test_an_unreachable_install_refuses_the_session",
        because="a user who cannot read the interpreter being handed a "
                "runner that dies on its first lazy import, several layers "
                "from the cause",
    ),
    Reversion(
        target="jaato-server/jaato_server/server/runner_pool.py",
        find='        return getattr(slot, "runner_uid", None) == self.runner_uid\n',
        replace="        return True\n",
        test="TestTheSlotKey::test_a_served_slot_fits_only_its_own_uid",
        because="a slot that dropped to one user being handed to another "
                "user's session (or to a root one): the drop is "
                "irreversible, so the bootstrap refuses, or worse, runs",
    ),
    Reversion(
        target="jaato-server/jaato_server/server/runner/session.py",
        find="    _drop_to_runner_user(envelope)\n\n",
        replace="\n",
        test="TestTheOrder::test_the_pool_slot_drops_between_tmp_and_confine",
        because="a pool slot never dropping: every session it serves runs "
                "as root although the policy named a user",
    ),
    Reversion(
        target="jaato-server/jaato_server/server/runner_spawner.py",
        find="        _drop_privileges_in_child(runner_user)\n",
        replace="        pass\n",
        test="TestTheOrder::test_the_cold_spawn_child_drops_before_exec",
        because="a cold-spawned runner exec'd as root although the policy "
                "named a user",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/privilege_drop.py",
        find=(
            "    else:\n"
            "        raise PrivilegeDropError(\n"
            "            f\"dropped to {user.describe()} but setuid(0) still succeeds; \"\n"
        ),
        replace=(
            "    else:\n"
            "        pass\n"
            "    if False:\n"
            "        raise PrivilegeDropError(\n"
            "            f\"dropped to {user.describe()} but setuid(0) still succeeds; \"\n"
        ),
        test="TestTheDrop::test_a_drop_that_did_not_stick_is_refused",
        because="a drop that left a way back to root reported as a success",
    ),
    Reversion(
        target="jaato-server/jaato_server/server/runner_user.py",
        find="    if st.st_uid != os.getuid() or st.st_uid == user.uid:\n",
        replace="    if st.st_uid == user.uid:\n",
        test="TestTheOwnedPaths::test_another_users_file_is_not_taken",
        because="the daemon chowning a path some other user owns to the "
                "session's user -- taking a file away from its owner",
    ),
    Reversion(
        target="jaato-server/jaato_server/server/session_manager.py",
        find="        if isinstance(exc, RunnerUserRefused):\n",
        replace="        if False:\n",
        test="TestTheRefusal::test_the_ipc_path_refuses_rather_than_falling_back",
        because="a refused drop falling back to in-process tool execution, "
                "i.e. running the model's tools in the ROOT daemon",
    ),
    Reversion(
        target="jaato-server/jaato_server/server/runner/session.py",
        find="        ensure_dumpable(user)\n",
        replace="",
        test="TestDumpableAfterInProcessDrop::test_the_slot_restores_dumpable_after_the_drop",
        because="a pool slot that dropped in process staying non-dumpable: "
                "its /proc is root-owned and every owner /proc/*/... rule "
                "denies the bootstrap read-back (#1499)",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/privilege_drop.py",
        find="    if now != 1:\n",
        replace="    if False:\n",
        test="TestDumpableAfterInProcessDrop::test_a_prctl_that_did_not_stick_refuses",
        because="a slot left non-dumpable reported as fixed, refused by "
                "AppArmor later and far from the cause",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/privilege_drop.py",
        find="    except OSError as exc:\n        raise PrivilegeDropError(\n            f\"dropped to {user.describe()} but could not make",
        replace="    except ValueError as exc:\n        raise PrivilegeDropError(\n            f\"dropped to {user.describe()} but could not make",
        test="TestDumpableAfterInProcessDrop::test_a_failed_prctl_refuses_the_bootstrap",
        because="a prctl failure escaping as a bare OSError instead of the "
                "privilege_drop bootstrap refusal",
    ),
]


NOBODY = RunnerUser(uid=65534, gid=65534, groups=(65534,), username="nobody",
                    home="/nonexistent", source="peer")


@pytest.fixture(autouse=True)
def _fresh_announcements():
    """The once-per-reason latch is process state; clear it per test."""
    from jaato_server.server import runner_user
    runner_user._announced.clear()
    yield
    runner_user._announced.clear()


# ------------------------------------------------------------------ policy

class TestThePolicy:
    """Who the runner becomes: named, or the daemon's uid -- never guessed."""

    def test_the_default_is_daemon_and_changes_nothing(self, monkeypatch):
        from jaato_server.server import runner_user
        monkeypatch.delenv("JAATO_RUNNER_UID_POLICY", raising=False)
        assert runner_user.resolve_policy(None) == "daemon"
        assert runner_user.resolve_runner_user(
            peer=PeerCredentials(uid=1001, gid=1001), workspace_path="/w",
            policy="daemon") is None

    def test_the_flag_outranks_the_env_and_a_typo_keeps_the_default(
        self, monkeypatch, caplog,
    ):
        from jaato_server.server import runner_user
        monkeypatch.setenv("JAATO_RUNNER_UID_POLICY", "workspace-owner")
        assert runner_user.resolve_policy(None) == "workspace-owner"
        assert runner_user.resolve_policy("peer") == "peer"
        with caplog.at_level(logging.ERROR):
            assert runner_user.resolve_policy("pere") == "daemon"
        assert [r for r in caplog.records if r.levelno == logging.ERROR]

    def test_a_non_root_daemon_keeps_its_uid(self, monkeypatch, caplog):
        from jaato_server.server import runner_user
        monkeypatch.setattr(runner_user, "_running_as_root", lambda: False)
        monkeypatch.setattr(runner_user, "refuse_if_unreachable",
                            lambda *a, **k: None)
        with caplog.at_level(logging.WARNING):
            got = runner_user.resolve_runner_user(
                peer=PeerCredentials(uid=1001, gid=1001),
                workspace_path="/w", policy="peer")
        assert got is None
        assert any("not root" in r.getMessage() for r in caplog.records)

    def test_a_connection_with_no_principal_keeps_the_daemon_uid(
        self, monkeypatch, caplog,
    ):
        """WS under ``peer``: announced once, never a guessed uid."""
        from jaato_server.server import runner_user
        monkeypatch.setattr(runner_user, "_running_as_root", lambda: True)
        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                assert runner_user.resolve_runner_user(
                    peer=None, workspace_path="/w", policy="peer") is None
        lines = [r for r in caplog.records if "no OS principal" in r.getMessage()]
        assert len(lines) == 1

    def test_a_root_peer_is_not_a_drop_target(self, monkeypatch):
        from jaato_server.server import runner_user
        monkeypatch.setattr(runner_user, "_running_as_root", lambda: True)
        monkeypatch.setattr(runner_user, "refuse_if_unreachable",
                            lambda *a, **k: None)
        assert runner_user.resolve_runner_user(
            peer=PeerCredentials(uid=0, gid=0), workspace_path="/w",
            policy="peer") is None

    def test_a_peer_names_its_own_uid(self, monkeypatch):
        from jaato_server.server import runner_user
        monkeypatch.setattr(runner_user, "_running_as_root", lambda: True)
        monkeypatch.setattr(runner_user, "refuse_if_unreachable",
                            lambda *a, **k: None)
        monkeypatch.setattr(runner_user, "announce_daemon_credentials_hidden",
                            lambda: None)
        monkeypatch.setattr(os, "getuid", lambda: 0)
        user = runner_user.resolve_runner_user(
            peer=PeerCredentials(uid=65534, gid=65534), workspace_path="/w",
            policy="peer")
        assert user is not None and user.uid == 65534 and user.gid == 65534
        assert user.source == "peer"

    def test_workspace_owner_reads_the_directory(self, monkeypatch, tmp_path):
        from jaato_server.server import runner_user
        uid, gid, why = runner_user._target_uid_gid(
            "workspace-owner", None, str(tmp_path))
        assert (uid, gid, why) == (
            tmp_path.stat().st_uid, tmp_path.stat().st_gid, "")
        uid, _, why = runner_user._target_uid_gid(
            "workspace-owner", None, str(tmp_path / "missing"))
        assert uid is None and "stat" in why

    def test_an_unreachable_install_refuses_the_session(self, tmp_path,
                                                        monkeypatch):
        """``nobody`` cannot enter a 0700 directory it does not own."""
        from jaato_server.server import runner_user
        private = tmp_path / "private-venv"
        private.mkdir()
        (private / "python").write_text("")
        os.chmod(private, 0o700)
        monkeypatch.setattr(runner_user, "runner_import_paths",
                            lambda: [str(private / "python")])
        if os.getuid() == 65534:
            pytest.skip("the suite runs as nobody")
        with pytest.raises(runner_user.RunnerUserRefused, match="cannot read"):
            runner_user.refuse_if_unreachable(NOBODY, None)

    def test_the_import_paths_name_the_interpreter_and_jaato(self):
        import sys

        import jaato_server
        from jaato_server.server import runner_user
        paths = runner_user.runner_import_paths()
        assert os.path.realpath(sys.executable) in paths
        assert any(os.path.realpath(p) in paths for p in jaato_server.__path__)


# -------------------------------------------------------------- slot key

class TestTheSlotKey:
    """``SlotKey.runner_uid``: a dropped slot is that uid for life."""

    @staticmethod
    def _pool(*slots):
        from jaato_server.server.runner_pool import PoolManager
        pool = PoolManager(template_manager=MagicMock(), target_size=8)
        pool._idle_slots = list(slots)
        return pool

    @staticmethod
    def _slot(pid, *, runner_uid=None, has_served=True):
        from jaato_server.server.runner_pool import PoolSlot
        return PoolSlot(pid=pid, sock=MagicMock(), runner_uid=runner_uid,
                        has_served=has_served)

    def test_a_virgin_slot_fits_any_uid(self):
        from jaato_server.server.runner_pool import SlotKey
        slot = self._slot(1, has_served=False)
        assert SlotKey.build(runner_uid=1001).accepts_unaffined(slot)
        assert SlotKey.build(runner_uid=None).accepts_unaffined(slot)

    def test_a_served_slot_fits_only_its_own_uid(self):
        from jaato_server.server.runner_pool import SlotKey
        slot = self._slot(1, runner_uid=1001)
        assert SlotKey.build(runner_uid=1001).accepts_unaffined(slot)
        assert not SlotKey.build(runner_uid=1002).accepts_unaffined(slot)
        assert not SlotKey.build(runner_uid=None).accepts_unaffined(slot)

    def test_acquire_passes_over_another_uid_and_counts_it(self):
        pool = self._pool(self._slot(1, runner_uid=1001))
        assert pool.acquire_slot(runner_uid=1002) is None
        counters = pool.get_telemetry()
        assert counters["pool_uid_mismatch_skips_total"] == 1
        assert pool.acquire_slot(runner_uid=1001) is not None

    def test_the_stamp_records_the_uid(self):
        from jaato_server.server.runner_pool import SlotKey
        slot = self._slot(1, has_served=False)
        SlotKey.build(runner_uid=1001).stamp(slot)
        assert slot.runner_uid == 1001 and slot.has_served

    def test_the_cascade_path_compares_the_uid_too(self):
        from jaato_server.server.runner_pool import SlotKey
        slot = self._slot(1, runner_uid=1001)
        slot.cascade_id = "cid"
        assert SlotKey.of_slot(slot) != SlotKey.build(
            cascade_driver_id="cid", runner_uid=1002)


# ------------------------------------------------------------------- order

class TestTheOrder:
    """cgroup + private /tmp (root) -> drop -> aa_change_profile."""

    def test_the_pool_slot_drops_between_tmp_and_confine(self, monkeypatch):
        from jaato_server.server.runner import session as session_mod
        from jaato_server.shared.session_envelope import SessionInitEnvelope

        calls = []
        for name in ("_enter_private_tmp", "_drop_to_runner_user",
                     "_maybe_self_confine", "_pin_session_tmpdir"):
            monkeypatch.setattr(
                session_mod, name,
                lambda *a, _n=name, **k: calls.append(_n))
        env = SessionInitEnvelope(
            session_id="s1", workspace_path=None, profile_name="",
            provider_name="echo", model_name="m", plugins=None,
            runner_user=NOBODY.to_dict(),
        )
        session_mod.bootstrap_session(env, runtime_factory=None)
        assert calls == ["_enter_private_tmp", "_drop_to_runner_user",
                         "_maybe_self_confine", "_pin_session_tmpdir"]

    @staticmethod
    def _calls_in(func: ast.AST):
        names = []
        for node in ast.walk(func):
            if isinstance(node, ast.Call):
                f = node.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                names.append((node.lineno, name))
        return [n for _, n in sorted(names)]

    def _function(self, name):
        from jaato_server.server import runner_spawner
        tree = ast.parse(pathlib.Path(runner_spawner.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        raise AssertionError(f"{name} not found")

    def test_the_cold_spawn_child_drops_before_exec(self):
        spawn = self._calls_in(self._function("spawn"))
        assert spawn.index("cgroup_attach") < spawn.index(
            "_enter_private_tmp_in_child") < spawn.index("_exec_runner")
        exec_calls = self._calls_in(self._function("_exec_runner"))
        assert "_drop_privileges_in_child" in exec_calls, (
            "the cold-spawn child no longer drops before exec")
        drop = exec_calls.index("_drop_privileges_in_child")
        assert exec_calls.index("dup2") < drop < exec_calls.index("execvpe")

    def test_the_cold_spawn_child_exits_rather_than_run_as_root(
        self, monkeypatch,
    ):
        from jaato_server.server import runner_spawner

        def refuse(_user):
            raise PrivilegeDropError("nope")

        exits = []
        monkeypatch.setattr(runner_spawner, "drop_to", refuse)
        monkeypatch.setattr(os, "_exit", lambda code: exits.append(code))
        monkeypatch.setattr(os, "write", lambda fd, data: len(data))
        runner_spawner._drop_privileges_in_child(NOBODY)
        assert exits == [runner_spawner.PRIVILEGE_DROP_EXIT_CODE]


# -------------------------------------------------------------------- drop

class TestTheDrop:
    """``drop_to`` with its syscalls substituted, then (as root) for real."""

    @staticmethod
    def _fake_kernel(monkeypatch, *, setuid0_succeeds=False):
        from jaato_server.shared import privilege_drop
        state = {"res": (0, 0, 0), "gid": (0, 0, 0), "groups": None}
        fake = SimpleNamespace(
            getresuid=lambda: state["res"], getresgid=lambda: state["gid"],
            geteuid=lambda: state["res"][1],
            setgroups=lambda g: state.__setitem__("groups", list(g)),
            setresgid=lambda *g: state.__setitem__("gid", g),
            setresuid=lambda *u: state.__setitem__("res", u),
        )

        def setuid(uid):
            if not setuid0_succeeds:
                raise PermissionError(1, "EPERM")
            state["res"] = (uid,) * 3
        fake.setuid = setuid
        for name, value in vars(fake).items():
            monkeypatch.setattr(privilege_drop.os, name, value)
        return state

    def test_groups_then_gid_then_uid(self, monkeypatch):
        state = self._fake_kernel(monkeypatch)
        assert drop_to(NOBODY) is True
        assert state == {"res": (65534,) * 3, "gid": (65534,) * 3,
                         "groups": [65534]}

    def test_a_drop_that_did_not_stick_is_refused(self, monkeypatch):
        self._fake_kernel(monkeypatch, setuid0_succeeds=True)
        with pytest.raises(PrivilegeDropError, match="setuid\\(0\\)"):
            drop_to(NOBODY)

    def test_uid_zero_is_never_a_target(self):
        with pytest.raises(PrivilegeDropError, match="uid 0"):
            drop_to(RunnerUser(uid=0, gid=0))

    def test_already_that_user_is_a_no_op(self, monkeypatch):
        state = self._fake_kernel(monkeypatch)
        state["res"] = (65534,) * 3
        state["gid"] = (65534,) * 3
        assert drop_to(NOBODY) is False

    def test_the_wire_form_round_trips_and_a_bad_one_refuses(self):
        assert RunnerUser.from_dict(NOBODY.to_dict()) == NOBODY
        assert RunnerUser.from_dict(None) is None
        with pytest.raises(PrivilegeDropError):
            RunnerUser.from_dict({"uid": "x"})

    def test_the_envelope_carries_it(self):
        from jaato_server.shared.session_envelope import SessionInitEnvelope
        env = SessionInitEnvelope(
            session_id="s", workspace_path=None, profile_name="",
            provider_name="p", model_name="m", plugins=None,
            runner_user=NOBODY.to_dict())
        back = SessionInitEnvelope.from_dict(env.to_dict())
        assert RunnerUser.from_dict(back.runner_user) == NOBODY

    def test_the_bootstrap_step_refuses_rather_than_stay_root(self,
                                                              monkeypatch):
        from jaato_server.server.runner import session as session_mod

        def refuse(_user):
            raise PrivilegeDropError("no")
        monkeypatch.setattr(session_mod, "drop_to", refuse)
        env = SimpleNamespace(runner_user=NOBODY.to_dict())
        with pytest.raises(session_mod.BootstrapError) as info:
            session_mod._drop_to_runner_user(env)
        assert info.value.stage == "privilege_drop"

    @pytest.mark.skipif(os.geteuid() != 0, reason="needs root to drop")
    def test_a_real_child_drops_to_nobody_and_writes_as_nobody(self):
        """Fork, drop for real, write a file, report the uid.

        In a fresh directory under the system temp root rather than
        ``tmp_path``, whose ``0700`` parents ``nobody`` cannot traverse.
        """
        import shutil
        import tempfile
        workdir = pathlib.Path(tempfile.mkdtemp(prefix="jaato-1168-"))
        self._cleanup = lambda: shutil.rmtree(workdir, ignore_errors=True)
        os.chmod(workdir, 0o777)
        target = workdir / "written-by-runner"
        read_end, write_end = os.pipe()
        pid = os.fork()
        if pid == 0:  # pragma: no cover - child
            code = 1
            try:
                drop_to(NOBODY)
                target.write_text("x")
                os.write(write_end, f"{os.getuid()} {os.getgid()}".encode())
                code = 0
            except BaseException as exc:  # noqa: BLE001 - report it
                os.write(write_end, repr(exc).encode()[:200])
            finally:
                os._exit(code)
        os.close(write_end)
        _, status = os.waitpid(pid, 0)
        report = os.read(read_end, 64).decode()
        os.close(read_end)
        assert os.waitstatus_to_exitcode(status) == 0, report
        try:
            assert report == "65534 65534"
            assert target.stat().st_uid == 65534
        finally:
            self._cleanup()


# ----------------------------------------------------------- dumpable (#1499)

class TestDumpableAfterInProcessDrop:
    """A pool slot drops without exec, so it must restore dumpable itself.

    ``setresuid`` clears the dumpable flag and the kernel then owns the
    process's ``/proc/<pid>/`` as root, so the profile's ``owner`` rules
    never match.  ``execve`` restores the flag for a cold spawn; a slot
    calls ``ensure_dumpable`` between the drop and ``aa_change_profile``.
    """

    @staticmethod
    def _fake_prctl(monkeypatch, *, start=2, sticks=True, fails=False):
        from jaato_server.shared import privilege_drop
        state = {"dumpable": start, "calls": []}

        def prctl(option, arg=0):
            state["calls"].append((option, arg))
            if fails:
                raise OSError(1, "EPERM")
            if option == privilege_drop.PR_SET_DUMPABLE and sticks:
                state["dumpable"] = arg
            return state["dumpable"] if option == privilege_drop.PR_GET_DUMPABLE else 0
        monkeypatch.setattr(privilege_drop, "_prctl", prctl)
        return state

    def test_the_slot_restores_dumpable_after_the_drop(self, monkeypatch):
        from jaato_server.server.runner import session as session_mod
        order = []
        monkeypatch.setattr(session_mod, "drop_to",
                            lambda u: order.append("drop_to") or True)
        state = self._fake_prctl(monkeypatch)
        monkeypatch.setattr(session_mod, "apply_user_env", lambda *a: None)
        session_mod._drop_to_runner_user(
            SimpleNamespace(runner_user=NOBODY.to_dict()))
        assert state["dumpable"] == 1
        assert order == ["drop_to"]

    def test_the_restore_is_after_setresuid_and_inside_the_drop_step(self):
        """AST: inside ``_drop_to_runner_user``, drop_to precedes
        ensure_dumpable; the step itself precedes ``_maybe_self_confine``
        (checked by ``TestTheOrder``)."""
        from jaato_server.server.runner import session as session_mod
        tree = ast.parse(pathlib.Path(session_mod.__file__).read_text())
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == "_drop_to_runner_user")
        calls = TestTheOrder._calls_in(fn)
        assert "ensure_dumpable" in calls
        assert calls.index("drop_to") < calls.index("ensure_dumpable")

    def test_the_cold_spawn_path_does_not_call_it(self):
        from jaato_server.server import runner_spawner
        src = pathlib.Path(runner_spawner.__file__).read_text()
        assert "ensure_dumpable" not in src and "PR_SET_DUMPABLE" not in src

    def test_already_dumpable_is_a_no_op(self, monkeypatch):
        from jaato_server.shared import privilege_drop
        state = self._fake_prctl(monkeypatch, start=1)
        assert privilege_drop.ensure_dumpable(NOBODY) is False
        assert [c[0] for c in state["calls"]] == [privilege_drop.PR_GET_DUMPABLE]

    def test_a_prctl_that_did_not_stick_refuses(self, monkeypatch):
        from jaato_server.shared import privilege_drop
        self._fake_prctl(monkeypatch, sticks=False)
        with pytest.raises(PrivilegeDropError, match="PR_GET_DUMPABLE"):
            privilege_drop.ensure_dumpable(NOBODY)

    def test_a_failed_prctl_refuses_the_bootstrap(self, monkeypatch):
        from jaato_server.server.runner import session as session_mod
        monkeypatch.setattr(session_mod, "drop_to", lambda u: True)
        self._fake_prctl(monkeypatch, fails=True)
        with pytest.raises(session_mod.BootstrapError) as info:
            session_mod._drop_to_runner_user(
                SimpleNamespace(runner_user=NOBODY.to_dict()))
        assert info.value.stage == "privilege_drop"
        assert "dumpable" in str(info.value)

    @pytest.mark.skipif(os.geteuid() != 0, reason="needs root to drop")
    def test_a_real_child_drops_without_exec_and_owns_its_proc(self):
        from jaato_server.shared import privilege_drop
        read_end, write_end = os.pipe()
        pid = os.fork()
        if pid == 0:  # pragma: no cover - child
            code = 1
            try:
                drop_to(NOBODY)
                before = privilege_drop._prctl(privilege_drop.PR_GET_DUMPABLE)
                privilege_drop.ensure_dumpable(NOBODY)
                after = privilege_drop._prctl(privilege_drop.PR_GET_DUMPABLE)
                owner = os.stat(f"/proc/{os.getpid()}/attr/current").st_uid
                os.write(write_end, f"{before} {after} {owner}".encode())
                code = 0
            except BaseException as exc:  # noqa: BLE001 - report it
                os.write(write_end, repr(exc).encode()[:200])
            finally:
                os._exit(code)
        os.close(write_end)
        _, status = os.waitpid(pid, 0)
        report = os.read(read_end, 256).decode()
        os.close(read_end)
        assert os.waitstatus_to_exitcode(status) == 0, report
        before, after, owner = report.split()
        assert after == "1" and owner == "65534", report


# ------------------------------------------------------------ owned paths

class TestTheOwnedPaths:
    """Hand over what the daemon made for the runner; take nothing else."""

    @staticmethod
    def _record_chowns(monkeypatch, daemon_uid):
        from jaato_server.server import runner_user
        calls = []
        monkeypatch.setattr(runner_user.os, "lchown",
                            lambda p, u, g: calls.append((str(p), u, g)))
        monkeypatch.setattr(runner_user.os, "getuid", lambda: daemon_uid)
        return calls

    def test_the_daemons_paths_are_handed_over_and_created(self, tmp_path,
                                                            monkeypatch):
        from jaato_server.server import runner_user
        mine = os.getuid()
        calls = self._record_chowns(monkeypatch, mine)
        ws = tmp_path / "ws"
        ws.mkdir()
        dirs, files = runner_user.runner_owned_paths(
            session_id="s1", workspace_path=str(ws),
            session_tmp=str(tmp_path / "tmp" / "s1"), private_tmp=None,
            workspace_home=None, log_path=str(ws / ".jaato/logs/runner-s1.log"))
        runner_user.prepare_runner_owned_paths(NOBODY, dirs, files)
        chowned = {p for p, _, _ in calls}
        for rel in (".jaato", ".jaato/logs", ".jaato/sessions",
                    ".jaato/sessions/s1", ".jaato/logs/runner-s1.log"):
            assert str(ws / rel) in chowned, rel
        assert str(ws) not in chowned, "the user's own workspace root"
        assert stat.S_IMODE((ws / ".jaato/logs/runner-s1.log").stat().st_mode) == 0o600

    def test_another_users_file_is_not_taken(self, tmp_path, monkeypatch):
        from jaato_server.server import runner_user
        # The daemon is "uid 4242"; the file is owned by whoever runs the
        # suite, i.e. somebody else.
        calls = self._record_chowns(monkeypatch, 4242)
        existing = tmp_path / "logs"
        existing.mkdir()
        runner_user.prepare_runner_owned_paths(NOBODY, [str(existing)], [])
        assert calls == []

    def test_no_user_touches_nothing(self, tmp_path, monkeypatch):
        from jaato_server.server import runner_user
        calls = self._record_chowns(monkeypatch, os.getuid())
        runner_user.prepare_runner_owned_paths(None, [str(tmp_path / "x")], [])
        assert calls == [] and not (tmp_path / "x").exists()

    @pytest.mark.skipif(os.geteuid() != 0, reason="needs root to chown")
    def test_for_real_as_root(self, tmp_path):
        from jaato_server.server import runner_user
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "users-own").write_text("")
        dirs, files = runner_user.runner_owned_paths(
            session_id="s1", workspace_path=str(ws), session_tmp=None,
            private_tmp=None, workspace_home=None, log_path=None)
        runner_user.prepare_runner_owned_paths(NOBODY, dirs, files)
        assert (ws / ".jaato/logs").stat().st_uid == 65534
        assert (ws / ".jaato/sessions/s1").stat().st_uid == 65534
        assert (ws / "users-own").stat().st_uid == 0


# ---------------------------------------------------------------- refusal

class TestTheRefusal:
    """A refused drop refuses the session on both transports."""

    def test_the_ipc_path_refuses_rather_than_falling_back(self):
        from jaato_server.server.runner_user import RunnerUserRefused
        from jaato_server.server.session_manager import SessionManager
        manager = SessionManager.__new__(SessionManager)
        manager._notify_apparmor = MagicMock()
        server = MagicMock()
        manager._report_runner_spawn_failure(
            server, "c1", "s1", RunnerUserRefused("cannot read"))
        server.note_runner_bootstrap_outcome.assert_called_once()
        manager._notify_apparmor.assert_not_called()

    def test_an_ordinary_spawn_failure_still_falls_back(self):
        from jaato_server.server.session_manager import SessionManager
        manager = SessionManager.__new__(SessionManager)
        manager._notify_apparmor = MagicMock()
        server = MagicMock()
        manager._report_runner_spawn_failure(server, "c1", "s1", OSError("x"))
        server.note_runner_bootstrap_outcome.assert_not_called()
        manager._notify_apparmor.assert_called_once()

    def test_the_ws_path_refuses_even_when_unconfined(self):
        from jaato_server.server import websocket
        from jaato_server.server.runner_user import RunnerUserRefused
        server = MagicMock()
        websocket._report_confined_spawn_failure(
            server, "s1", RunnerUserRefused("cannot read"),
            confinement_required=False)
        server.note_runner_bootstrap_outcome.assert_called_once()

    def test_the_ipc_spawn_hands_the_peer_and_the_pool_the_uid(self):
        """The two call sites the mechanism is inert without (#735)."""
        from jaato_server.server import runner_spawn, session_manager
        sm_src = pathlib.Path(session_manager.__file__).read_text()
        assert "peer=self._client_peer(client_id)," in sm_src
        rs_src = pathlib.Path(runner_spawn.__file__).read_text()
        assert "runner_uid=runner_uid_of(runner_user)," in rs_src
        assert "runner_user=stashed_runner_user(server)," in rs_src
        assert "runner_user=_runner_user_wire(server)," in rs_src
