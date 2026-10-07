"""A confined session's /tmp allowance is the directory its profile grants (#1361).

``sandbox_utils`` allowed the whole of ``/tmp``; a confined runner's AppArmor
profile grants only its session tmpdir under ``/tmp/jaato-<id>/`` (#1171).
So ``cat > /tmp/x`` passed ``cli``'s pre-flight and then failed with
``Permission denied``, and both fresh-workspace assessments concluded "/tmp
is not writable" and moved their scratch files into the workspace.  The
runner now narrows the allowance to the directory it pinned, and the
``runtime`` aspect and the ``cli`` description name ``$TMPDIR``.

CI has no AppArmor kernel; the profile name is only a string here.
"""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

from jaato_server.server.runner import session as runner_session
from jaato_server.shared.plugins import sandbox_utils
from jaato_server.shared.plugins.cli.plugin import CLIToolPlugin
from jaato_server.shared.plugins.environment import runtime
from jaato_server.shared.tests.reversion import Reversion

_RS = "jaato-server/jaato_server/server/runner/session.py"
_RUNTIME = "jaato-server/jaato_server/shared/plugins/environment/runtime.py"

REVERSIONS = [
    Reversion(
        target=_RS,
        find="        sandbox_utils.narrow_temp_roots(path)\n",
        replace="        pass\n",
        test="test_a_confined_runner_allows_only_its_session_tmpdir",
        because="cli's pre-flight allows /tmp paths the kernel will refuse",
    ),
    Reversion(
        target=_RS,
        find="        sandbox_utils.restore_temp_roots()\n",
        replace="        pass\n",
        test="test_an_unconfined_runner_keeps_the_whole_of_tmp",
        because="an unconfined session inherits a confined one's narrowed /tmp",
    ),
    Reversion(
        target=_RUNTIME,
        find='        "tmpdir": env.get("TMPDIR"),\n',
        replace="",
        test="test_the_runtime_aspect_reports_the_session_tmpdir",
        because="the model cannot ask where it may write temp files",
    ),
]


def _pin(monkeypatch, tmp_path, profile_name):
    """Run the runner's pin with the session tmpdir under ``tmp_path``."""
    session_dir = tmp_path / "jaato-boundary" / "sess1"
    monkeypatch.setattr(sandbox_utils, "SYSTEM_TEMP_PATHS",
                        list(sandbox_utils.SYSTEM_TEMP_PATHS))
    monkeypatch.setattr(runner_session, "session_tmpdir",
                        lambda sid, cid: str(session_dir))
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    monkeypatch.setenv("TMPDIR", os.environ.get("TMPDIR", "/tmp"))
    runner_session._pin_session_tmpdir(
        SimpleNamespace(session_id="sess1", profile_name=profile_name))
    return str(session_dir)


def test_a_confined_runner_allows_only_its_session_tmpdir(monkeypatch, tmp_path):
    session_dir = _pin(monkeypatch, tmp_path, "jaato-ws-abc")

    assert sandbox_utils.is_under_temp_path(os.path.join(session_dir, "x"))
    assert not sandbox_utils.is_under_temp_path("/tmp/x")

    workspace = tmp_path / "ws"
    workspace.mkdir()
    assert not sandbox_utils.check_path_with_jaato_containment(
        "/tmp/x", str(workspace))
    assert sandbox_utils.check_path_with_jaato_containment(
        os.path.join(session_dir, "x"), str(workspace))


def test_an_unconfined_runner_keeps_the_whole_of_tmp(monkeypatch, tmp_path):
    _pin(monkeypatch, tmp_path, "jaato-ws-abc")
    assert not sandbox_utils.is_under_temp_path("/tmp/x")
    _pin(monkeypatch, tmp_path, "")
    assert sandbox_utils.is_under_temp_path("/tmp/x")


def test_the_runtime_aspect_reports_the_session_tmpdir():
    cli = SimpleNamespace(_build_subprocess_env=lambda: (
        {"PATH": "/usr/bin", "TMPDIR": "/tmp/jaato-b/s1"}, None))
    report = runtime.subprocess_report(cli)
    assert report["tmpdir"] == "/tmp/jaato-b/s1"
    assert ("tmpdir", "/tmp/jaato-b/s1") in runtime._subprocess_lines(report)


def test_the_cli_description_names_tmpdir():
    text = CLIToolPlugin().get_system_instructions()
    assert "$TMPDIR" in text
