"""Phase 5 — session-scoped TMPDIR for the confined runner subprocess.

The AppArmor profile only allows writes under
``/tmp/jaato-{session_id}/**``.  Python's ``tempfile.gettempdir()``
probes ``/tmp``, ``/var/tmp``, ``/usr/tmp`` (and cwd) with a generic
sanity write — all denied → ``FileNotFoundError: No usable temporary
directory found``.  Fix: ``RunnerSpawner`` pre-sets ``TMPDIR`` to the
session-scoped path + ``mkdir -p``'s it before fork so the confined
runner's tempfile probe lands inside the profile's allow rule.

See ``project_backlog_runner_apparmor_tmpdir`` memory.

Lives in its own test file (not ``test_runner_spawn.py``) because
that file has an ``autouse=True`` fixture that swaps the real
``RunnerSpawner`` for a stub — these tests need the real class.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from server.runner_spawner import RunnerSpawner


class TestSessionTmpdirHelper:
    def test_returns_canonical_path(self):
        assert RunnerSpawner._session_tmpdir("sess-A") == "/tmp/jaato-sess-A"

    def test_handles_full_session_id(self):
        assert RunnerSpawner._session_tmpdir("20260512_120000") == (
            "/tmp/jaato-20260512_120000"
        )

    def test_isolated_subagent_session_id(self):
        """Isolated subagent ids use the ``{parent}__sub_{sub}`` template
        — the tmpdir convention extends to them naturally."""
        assert RunnerSpawner._session_tmpdir("sess-A__sub_agent-1") == (
            "/tmp/jaato-sess-A__sub_agent-1"
        )


class TestBuildEnvTmpdir:
    def test_sets_tmpdir_to_session_scoped_path(self):
        spawner = RunnerSpawner()
        env = spawner._build_env(
            profile_name="jaato-ws-sess-X",
            session_id="sess-X",
            workspace_path="/tmp/ws",
            log_path="/tmp/ws/.jaato/logs/runner-sess-X.log",
            max_output_chars=None,
            tool_timeout_seconds=None,
            disable_confine=False,
        )
        assert env["TMPDIR"] == "/tmp/jaato-sess-X"

    def test_sets_tmpdir_even_when_unconfined(self):
        """The TMPDIR convention applies in both confined and
        unconfined modes — consistency simplifies operator forensics
        across the two deployments."""
        spawner = RunnerSpawner()
        env = spawner._build_env(
            profile_name="",
            session_id="sess-Y",
            workspace_path=None,
            log_path=None,
            max_output_chars=None,
            tool_timeout_seconds=None,
            disable_confine=True,
        )
        assert env["TMPDIR"] == "/tmp/jaato-sess-Y"
        assert env.get("JAATO_RUNNER_DISABLE_CONFINE") == "1"


class TestSpawnCreatesTmpdirBeforeFork:
    def test_mkdir_lands_before_fork_raises(self, tmp_path):
        """``spawn`` must create the session tmpdir BEFORE fork — the
        confined child can't mkdir outside the allow list, and
        Python's tempfile probe happens at plugin-import time
        (post-self-confinement).

        Mocking ``os.fork`` to raise verifies the mkdir landed first.
        ``_session_tmpdir`` is patched to point at a sandbox path so
        we don't pollute the real ``/tmp``."""
        sandbox = tmp_path / "fake-tmp"
        target = sandbox / "jaato-sess-Z"

        with patch.object(
            RunnerSpawner, "_session_tmpdir",
            staticmethod(lambda session_id: str(target)),
        ):
            spawner = RunnerSpawner()
            with patch.object(spawner, "_assert_daemon_unconfined"):
                with patch(
                    "os.fork",
                    side_effect=RuntimeError("fork-blocked"),
                ):
                    with pytest.raises(RuntimeError, match="fork-blocked"):
                        spawner.spawn(
                            profile_name="jaato-ws-sess-Z",
                            session_id="sess-Z",
                            workspace_path="/tmp/ws",
                            log_path=None,
                            max_output_chars=None,
                            tool_timeout_seconds=None,
                            disable_confine=False,
                        )

        assert target.is_dir(), (
            "session tmpdir should have been created before fork; "
            "fork raised but the mkdir lives ahead of it in spawn()"
        )

    def test_mkdir_exist_ok_handles_stale_dir(self, tmp_path):
        """``exist_ok=True`` handles the rare case of a stale tmpdir
        from a prior session with the same id (e.g., session-restore
        path or rapid sequential spawns).  Pin that the spawn doesn't
        EEXIST when the dir is already there."""
        sandbox = tmp_path / "fake-tmp"
        target = sandbox / "jaato-sess-existing"
        target.mkdir(parents=True)
        # Drop a file so we can verify the dir wasn't replaced
        sentinel = target / "stale-file"
        sentinel.write_text("from-prior-session")

        with patch.object(
            RunnerSpawner, "_session_tmpdir",
            staticmethod(lambda session_id: str(target)),
        ):
            spawner = RunnerSpawner()
            with patch.object(spawner, "_assert_daemon_unconfined"):
                with patch(
                    "os.fork",
                    side_effect=RuntimeError("fork-blocked"),
                ):
                    with pytest.raises(RuntimeError, match="fork-blocked"):
                        spawner.spawn(
                            profile_name="jaato-ws-sess-existing",
                            session_id="sess-existing",
                            workspace_path="/tmp/ws",
                            log_path=None,
                            max_output_chars=None,
                            tool_timeout_seconds=None,
                            disable_confine=False,
                        )

        # Dir still exists + contents preserved (exist_ok behaviour)
        assert target.is_dir()
        assert sentinel.read_text() == "from-prior-session"
