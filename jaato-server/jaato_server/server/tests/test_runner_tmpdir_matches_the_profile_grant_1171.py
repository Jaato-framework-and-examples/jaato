"""The two sides of one agreement, asserted together (#1171).

A confined runner's temp files have to land where its AppArmor profile
says they may.  That is ONE fact spread across two modules:

* ``RunnerSpawner`` decides the path and puts it in ``TMPDIR``;
* ``AppArmorManager`` renders the rule that grants it.

Each side had a test.  Neither test read the other side, so when #1037
renamed the profile after the BOUNDARY rather than after the session,
the agreement broke and both suites stayed green — for the whole life of
0.16, 0.17 and 0.18, until a real confined cold-spawned runner died at
plugin-import time with::

    [Errno 2] No usable temporary directory found in
    ['/tmp/jaato-20260921_053346', '/tmp', '/var/tmp', '/usr/tmp', '/']

while the kernel logged the other half::

    apparmor="DENIED" operation="mknod" profile="jaato-ws-runtime-cdd58a0cee68"
      name="/tmp/jaato-20260921_053346/…"

**This file needs no kernel.**  That matters: every confinement test in
this tree runs with ``is_available()`` and ``apparmor_parser`` stubbed,
because CI has no AppArmor LSM — so nothing in CI can observe an AVC,
and a test that waited for one would never run.  The agreement above is
a property of two strings, and a string comparison would have failed the
day #1037 landed.

The second half of the fix has its own guard here too.  Making the paths
agree is not sufficient on the POOL path, because Python never reads
``TMPDIR`` there: the pre-warm template walks plugin discovery
unconfined, ``sandbox_utils`` resolves ``tempfile.gettempdir()`` at
module scope, and every forked slot inherits that cached ``/tmp`` — a
directory the base profile does not grant either.  So the runner pins
:data:`tempfile.tempdir` itself.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import pytest

from jaato_server.server.apparmor import AppArmorManager
from jaato_server.server.confinement_id import (
    confinement_id_from_profile_name,
    profile_name_for,
    session_tmpdir,
)
from jaato_server.server.runner_spawner import RunnerSpawner
from jaato_server.shared.tests.reversion import Reversion


REVERSIONS = [
    Reversion(
        target="jaato-server/jaato_server/server/confinement_id.py",
        find='    return f"{TMPDIR_PREFIX}{confinement_id}/{session_id}"',
        replace='    return f"{TMPDIR_PREFIX}{session_id}"',
        test="TestTheTwoSidesAgree::test_the_spawners_tmpdir_is_granted_by_the_profile",
        because=(
            "the runner's tmpdir keyed on the session id again, so a "
            "confined runner writes outside every rule its profile grants "
            "and dies on the tempfile probe"
        ),
    ),
    Reversion(
        target="jaato-server/jaato_server/server/runner_spawner.py",
        find='env["TMPDIR"] = self._session_tmpdir(session_id, profile_name)',
        replace='env["TMPDIR"] = self._session_tmpdir(session_id)',
        test="TestTheTwoSidesAgree::test_build_env_hands_the_runner_a_granted_path",
        because=(
            "the spawner ignoring the profile it was handed, so the "
            "convention is right and the value reaching the runner is not"
        ),
    ),
    Reversion(
        target="jaato-server/jaato_server/server/runner/session.py",
        find="    tempfile.tempdir = path",
        replace="    pass  # reversion",
        test=(
            "TestAForkedSlotDoesNotInheritTheTemplatesTempdir::"
            "test_the_pin_overrides_an_inherited_value"
        ),
        because=(
            "a pool slot keeping the template's cached /tmp, which the "
            "profile does not grant either -- the half of #1171 that "
            "matching the two ids does not fix"
        ),
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/plugins/sandbox_utils.py",
        find="    except Exception:  # noqa: BLE001",
        replace="    except ZeroDivisionError:  # noqa: BLE001",
        test=(
            "TestSandboxUtilsImportsWithoutAUsableTempDir::"
            "test_module_scope_resolution_is_not_fatal"
        ),
        because=(
            "a module-scope temp-dir PROBE turning 'I cannot write a temp "
            "file' back into 'this module cannot be imported', which fails "
            "plugin discovery and with it the whole bootstrap"
        ),
    ),
]


# The real thing, from the incident: a workspace slug plus a body digest.
CONFINEMENT_ID = "runtime-cdd58a0cee68"
PROFILE_NAME = profile_name_for(CONFINEMENT_ID)
SESSION_ID = "20260921_053346"


def _tmp_rules(rendered: str) -> list:
    """Every ``/tmp/...`` allow rule in a rendered profile body."""
    return [
        line.strip() for line in rendered.splitlines()
        if line.strip().startswith("/tmp/") and line.strip().endswith(",")
    ]


def _granted(rendered: str, path: str) -> bool:
    """Would one of the profile's ``/tmp`` rules cover a write to *path*?

    A deliberately narrow reading of AppArmor globbing — ``**`` crosses
    ``/``, ``*`` does not — because the question here is only whether
    the two sides of OUR convention line up.  It is not an AppArmor
    implementation, and it is not trying to be: a test that re-derived
    the kernel's matcher would be asserting its own model.
    """
    import fnmatch

    for rule in _tmp_rules(rendered):
        pattern = rule.rsplit(" ", 1)[0].strip().rstrip(",").strip()
        if pattern.endswith("/**"):
            if path.startswith(pattern[:-2]):
                return True
        elif "*" in pattern:
            if fnmatch.fnmatchcase(path, pattern):
                return True
        elif path == pattern.rstrip("/"):
            return True
    return False


@pytest.fixture
def rendered_profile():
    """The profile body as the daemon renders it for a boundary.

    ``_render_profile`` is handed the RENDER id — which is the
    confinement id, not the session id — exactly as
    ``provision_profile`` does at ``apparmor.py:1287``.  Getting that
    argument right is the whole point of the fixture.
    """
    mgr = AppArmorManager(workspace_root="/srv/ws")
    return mgr._render_profile(CONFINEMENT_ID, "/srv/ws/runtime")


class TestTheTwoSidesAgree:
    def test_the_spawners_tmpdir_is_granted_by_the_profile(self, rendered_profile):
        """The assertion that was missing.

        Both sides are computed here — no literal path is written down —
        so this keeps holding if the convention moves, and stops holding
        the moment the two stop agreeing.
        """
        path = RunnerSpawner._session_tmpdir(SESSION_ID, PROFILE_NAME)
        assert _granted(rendered_profile, path), (
            f"the runner's TMPDIR {path!r} is not covered by any /tmp rule "
            f"in the profile rendered for {CONFINEMENT_ID!r}: "
            f"{_tmp_rules(rendered_profile)}"
        )

    def test_a_file_inside_that_tmpdir_is_granted(self, rendered_profile):
        """The directory is not enough — the runner writes files in it,
        and that is the operation the kernel denied (``mknod``)."""
        path = RunnerSpawner._session_tmpdir(SESSION_ID, PROFILE_NAME)
        assert _granted(rendered_profile, f"{path}/tmpabcd1234")

    def test_the_pre_1171_path_is_the_one_that_was_denied(self, rendered_profile):
        """A control, so the test above cannot pass vacuously.

        If the profile happened to grant ``/tmp`` broadly, every path
        would be "granted" and the guard would prove nothing.  The
        session-keyed path is what the incident's AVC names, so it must
        still be refused.
        """
        assert not _granted(rendered_profile, f"/tmp/jaato-{SESSION_ID}/tmpabcd1234")

    def test_build_env_hands_the_runner_a_granted_path(self, rendered_profile):
        """End to end on the daemon side: what actually reaches the
        runner's environment, not just what the helper computes."""
        env = RunnerSpawner()._build_env(
            profile_name=PROFILE_NAME,
            session_id=SESSION_ID,
            workspace_path="/srv/ws/runtime",
            log_path=None,
            max_output_chars=None,
            tool_timeout_seconds=None,
            disable_confine=False,
        )
        assert _granted(rendered_profile, env["TMPDIR"])

    def test_an_unconfined_runner_is_unchanged(self):
        """The opt-out keeps the pre-#1171 path, so a deployment with no
        AppArmor sees no change at all."""
        assert RunnerSpawner._session_tmpdir(SESSION_ID, "") == (
            f"/tmp/jaato-{SESSION_ID}"
        )

    def test_the_id_round_trips_through_the_profile_name(self):
        """The spawner derives the id by inverting the name it was
        handed, so the two functions have to be inverses.  A second
        spelling of the prefix is how they stop being."""
        assert confinement_id_from_profile_name(PROFILE_NAME) == CONFINEMENT_ID
        assert AppArmorManager.profile_name_for_confinement_id(
            CONFINEMENT_ID) == PROFILE_NAME


class TestEverySpawnBranchCreatesTheDirectory:
    """The runner cannot make it: the profile grants
    ``/tmp/jaato-<id>/**`` but not ``/tmp/``, so the confined child may
    create its own subdirectory and not the boundary directory above it.
    """

    def test_spawn_session_runner_creates_it_before_either_branch(self, tmp_path):
        from jaato_server.server import runner_spawn

        target = tmp_path / "boundary" / "session"
        with patch.object(
            runner_spawn, "session_tmpdir",
            lambda session_id, confinement_id=None: str(target),
        ):
            runner_spawn._ensure_session_tmpdir(SESSION_ID, PROFILE_NAME)
        assert target.is_dir()

    def test_a_failure_to_create_it_does_not_raise(self, tmp_path):
        """Best-effort: a session that would otherwise run must not be
        taken down by this, and the failure is logged rather than
        swallowed silently."""
        from jaato_server.server import runner_spawn

        blocker = tmp_path / "not-a-dir"
        blocker.write_text("")
        with patch.object(
            runner_spawn, "session_tmpdir",
            lambda session_id, confinement_id=None: str(blocker / "under-a-file"),
        ):
            runner_spawn._ensure_session_tmpdir(SESSION_ID, PROFILE_NAME)


class TestAForkedSlotDoesNotInheritTheTemplatesTempdir:
    """#1171's second half.

    ``tempfile.gettempdir()`` resolves ONCE and caches into a module
    global.  The pre-warm template resolves it unconfined during plugin
    discovery, so every forked slot inherits ``/tmp`` and never consults
    ``TMPDIR`` again — measured, not assumed.
    """

    def test_cpython_caches_the_resolution_across_an_env_change(self):
        """The property the whole second half rests on.  If this ever
        stops being true, the pin becomes redundant rather than wrong —
        but the reader should know it was checked, not guessed."""
        before = tempfile.tempdir
        try:
            tempfile.tempdir = "/tmp"
            with patch.dict(os.environ, {"TMPDIR": "/tmp/somewhere-else"}):
                assert tempfile.gettempdir() == "/tmp"
        finally:
            tempfile.tempdir = before

    def test_the_pin_overrides_an_inherited_value(self):
        from jaato_server.shared.session_envelope import SessionInitEnvelope
        from jaato_server.server.runner.session import _pin_session_tmpdir

        before = tempfile.tempdir
        try:
            tempfile.tempdir = "/tmp"          # what a slot inherits
            envelope = SessionInitEnvelope(
                session_id=SESSION_ID,
                profile_name=PROFILE_NAME,
                model_name="echo",
                workspace_path="/srv/ws/runtime",
                provider_name="echo",
            )
            with patch.dict(os.environ, {}, clear=False):
                _pin_session_tmpdir(envelope)
                expected = session_tmpdir(SESSION_ID, CONFINEMENT_ID)
                assert tempfile.gettempdir() == expected
                # The subprocesses the model drives inherit the same
                # answer rather than resolving one of their own.
                assert os.environ["TMPDIR"] == expected
        finally:
            tempfile.tempdir = before

    def test_the_pin_agrees_with_the_spawner(self):
        """Daemon side and runner side derive the path independently.
        They must land on the same string — deriving it twice is what
        #1171 is."""
        from jaato_server.shared.session_envelope import SessionInitEnvelope
        from jaato_server.server.runner.session import _pin_session_tmpdir

        before = tempfile.tempdir
        try:
            tempfile.tempdir = None
            _pin_session_tmpdir(SessionInitEnvelope(
                session_id=SESSION_ID,
                profile_name=PROFILE_NAME,
                model_name="echo",
                workspace_path="/srv/ws/runtime",
                provider_name="echo",
            ))
            assert tempfile.gettempdir() == RunnerSpawner._session_tmpdir(
                SESSION_ID, PROFILE_NAME)
        finally:
            tempfile.tempdir = before


    def test_a_directory_that_cannot_exist_declines_to_pin(self, tmp_path):
        """``tempfile`` does not CREATE ``tempdir`` — it fails on use.

        So a pin whose directory is missing is worse than no pin: it
        refuses every temp operation, where the unpinned probe might
        still find somewhere usable.  The failure declines rather than
        half-applying.
        """
        from jaato_server.shared.session_envelope import SessionInitEnvelope
        from jaato_server.server.runner import session as runner_session

        blocker = tmp_path / "not-a-dir"
        blocker.write_text("")

        before = tempfile.tempdir
        try:
            tempfile.tempdir = "/tmp"
            with patch.object(
                runner_session, "session_tmpdir",
                lambda session_id, confinement_id=None: str(
                    blocker / "under-a-file"),
            ):
                runner_session._pin_session_tmpdir(SessionInitEnvelope(
                    session_id=SESSION_ID,
                    profile_name=PROFILE_NAME,
                    model_name="echo",
                    workspace_path="/srv/ws/runtime",
                    provider_name="echo",
                ))
            assert tempfile.tempdir == "/tmp", (
                "a pin that could not create its directory must leave the "
                "previous resolution alone"
            )
        finally:
            tempfile.tempdir = before


class TestSandboxUtilsImportsWithoutAUsableTempDir:
    """Defence in depth, and the module where the crash actually landed.

    ``SYSTEM_TEMP_PATHS`` classifies paths; it never writes a temp file.
    But ``gettempdir()`` PROBES, so a module-scope call turned "I cannot
    write a temp file" into "this module cannot be imported" — which
    failed plugin discovery, and with it the whole bootstrap.
    """

    def test_module_scope_resolution_is_not_fatal(self):
        from jaato_server.shared.plugins import sandbox_utils

        def denied():
            raise FileNotFoundError(
                2, "No usable temporary directory found in []")

        with patch.object(tempfile, "_get_default_tempdir", denied):
            before = tempfile.tempdir
            try:
                tempfile.tempdir = None
                assert sandbox_utils._default_temp_paths() == ["/tmp"]
            finally:
                tempfile.tempdir = before

    def test_a_usable_temp_dir_is_still_reported(self):
        """The fallback must not become the answer on a healthy host."""
        from jaato_server.shared.plugins import sandbox_utils

        before = tempfile.tempdir
        try:
            tempfile.tempdir = "/tmp/jaato-somewhere"
            assert sandbox_utils._default_temp_paths() == [
                "/tmp", "/tmp/jaato-somewhere",
            ]
        finally:
            tempfile.tempdir = before
