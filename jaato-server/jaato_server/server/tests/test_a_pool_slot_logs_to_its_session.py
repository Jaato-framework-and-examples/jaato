"""A pool slot writes its session's runner log, not the daemon's.

The SELinux phase 4 kernel run (b8e2dc42): a pool slot is forked from the
template and kept the template's stdout and stderr, which are the daemon's
own log.  Under AppArmor and unconfined its lines landed there; under
SELinux ``jaato_runner_t`` may not write that file, so ~700 writes per run
were refused and the slot's whole log was lost.  ``runner-<id>.log`` was
created and stayed empty.  A cold spawn has always had one, because its
child opens the file onto fds 1 and 2 before exec.

Every bootstrap now points fds 1 and 2 at ``envelope.runner_log_path``.
"""

import ast
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from jaato_server.server.runner_spawn import build_session_envelope, runner_log_path
from jaato_server.server.tests.test_envelope_carries_gc import _profile
from jaato_server.shared.session_envelope import SessionInitEnvelope
from jaato_server.shared.tests.reversion import Reversion

_SPAWN = "jaato-server/jaato_server/server/runner_spawn.py"
_RUNNER = "jaato-server/jaato_server/server/runner/session.py"

REVERSIONS = [
    Reversion(
        target=_SPAWN,
        find="        runner_log_path=runner_log_path(workspace_path, session_id),\n",
        replace="",
        test="test_the_envelope_names_the_sessions_log",
        because="a pool slot would keep writing the daemon's log",
    ),
    Reversion(
        target=_RUNNER,
        find="    _point_log_at_session(envelope)\n",
        replace="",
        test="test_bootstrap_points_the_log_first",
        because="the path would cross the wire and nothing would use it",
    ),
    Reversion(
        target=_RUNNER,
        find="        os.dup2(fd, 2)\n",
        replace="",
        test="test_stdout_and_a_stderr_handler_land_in_the_file",
        because="a slot logs through a stderr handler, so its lines would "
                "still go to the daemon's log",
    ),
]


def test_the_envelope_names_the_sessions_log():
    env = build_session_envelope(
        server=SimpleNamespace(_profile=_profile(), config_root=None,
                               _main_agent_id="main", _cascade_driver_id=None),
        session_id="s1", workspace_path="/tmp/ws", profile_name="p")
    assert env.runner_log_path == "/tmp/ws/.jaato/logs/runner-s1.log"
    assert runner_log_path(None, "s1") is None


def test_the_path_round_trips():
    env = SessionInitEnvelope(session_id="s1", workspace_path="/w", profile_name="",
                              model_name="m", provider_name="echo", plugins=[],
                              runner_log_path="/w/.jaato/logs/runner-s1.log")
    assert SessionInitEnvelope.from_dict(env.to_dict()).runner_log_path == env.runner_log_path


def test_bootstrap_points_the_log_first():
    tree = ast.parse(Path(__file__).resolve().parents[1].joinpath(
        "runner", "session.py").read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "bootstrap_session")
    calls = [n.func.id for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert "_point_log_at_session" in calls
    assert calls.index("_point_log_at_session") < calls.index("_apply_envelope_session_env")


_CHILD = """
import logging, sys
from jaato_server.server.runner import session
from jaato_server.shared.session_envelope import SessionInitEnvelope
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stderr)])
env = SessionInitEnvelope(session_id="s1", workspace_path="/w", profile_name="",
                          model_name="m", provider_name="echo", plugins=[],
                          runner_log_path=sys.argv[1])
session._point_log_at_session(env)
print("STDOUT-LINE", flush=True)
logging.getLogger("slot").info("HANDLER-LINE")
"""


def test_stdout_and_a_stderr_handler_land_in_the_file(tmp_path):
    log = tmp_path / "runner-s1.log"
    p = subprocess.run([sys.executable, "-c", _CHILD, str(log)],
                       capture_output=True, text=True, timeout=60)
    text = log.read_text()
    assert "STDOUT-LINE" in text and "HANDLER-LINE" in text, (p.stdout, p.stderr)
    assert "HANDLER-LINE" not in p.stderr
