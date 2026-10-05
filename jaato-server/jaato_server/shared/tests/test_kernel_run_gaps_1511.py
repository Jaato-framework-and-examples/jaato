"""Four AppArmor gaps found on an enforcing kernel (#1511), and what closed them.

From the 2026-10-04 runs on Ubuntu (kernel 7.0.0-31):

* ``shell_spawn`` got no pty in any confined session: the base profile
  denied ``/dev/ptmx`` (``wr``) to the ``runner-rpc-work`` thread.  pexpect
  opens the pty master IN-PROCESS, in the runner, before its child execs,
  so base (and ``tool_hat``, which mirrors it) needs the grant and
  ``//child`` does not.
* ``python3 -m venv`` failed in ``//child``: Debian's ``ensurepip`` reads
  its wheels from ``/usr/share/python-wheels/``.
* ``git`` in ``//child`` warned on ``/usr/share/git-core/templates/``.
* A daemon on a uv-managed Python could start no confined session: the
  stdlib under ``~/.local/share/uv/python/<build>`` was unreadable.  The
  interpreter installation is now derived from ``sys.base_prefix`` /
  ``sys.base_exec_prefix`` at render time, like ``{venv_path}``.

Template v45.  The fifth AppArmor bullet (``ldconfig`` / ``collect2`` exec
denials from ``ctypes.util.find_library``) was the seccomp compile, which
#1508 moved to the daemon; the one runner-side ``find_library`` left on a
path that can run confined (``privilege_drop._prctl``) is removed here.

These tests read rendered profile text through the #1348 rule matcher;
CI has no AppArmor kernel.  Where ``apparmor_parser`` is installed they
also compile the render.
"""

from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from jaato_server.server.apparmor import (
    AppArmorManager,
    interpreter_install_roots,
)
from jaato_server.shared.confinement_grants import ConfinementGrants
from jaato_server.shared.tests.reversion import Reversion

_AA = "jaato-server/jaato_server/server/apparmor.py"
_PD = "jaato-server/jaato_server/shared/privilege_drop.py"

# A uv-shaped interpreter root.  It need not exist: the renderer only
# writes the path into the rules.
_UV_ROOT = Path("/home/u/.local/share/uv/python/cpython-3.12.7-linux-x86_64-gnu")

REVERSIONS = [
    Reversion(
        target=_AA,
        find="\n  /dev/ptmx            rw,\n",
        replace="\n",
        test="test_the_runner_can_open_a_pty_master",
        because="the base profile denies /dev/ptmx, so shell_spawn gets no pty",
    ),
    Reversion(
        target=_AA,
        find="    /usr/share/python-wheels/**       r,\n",
        replace="",
        test="test_child_reads_python_wheels_and_git_templates",
        because="//child cannot read the ensurepip wheels; python3 -m venv fails",
    ),
    Reversion(
        target=_AA,
        find="    /usr/share/git-core/templates/**  r,\n",
        replace="",
        test="test_child_reads_python_wheels_and_git_templates",
        because="//child cannot read git's templates",
    ),
    Reversion(
        target=_AA,
        find="            interpreter_rules=self._interpreter_rules(workspace_path, \"  \"),\n",
        replace="            interpreter_rules=\"\",\n",
        test="test_every_body_reads_the_interpreter_installation",
        because="the base profile cannot read a uv-managed stdlib; no confined session starts",
    ),
    Reversion(
        target=_AA,
        find="        if workspace_root is not None and _is_within(root, workspace_root):",
        replace="        if False:",
        test="test_an_interpreter_inside_a_workspace_is_never_granted",
        because="an exec grant would land on a model-writable directory",
    ),
    Reversion(
        target=_PD,
        find="    libc = ctypes.CDLL(None, use_errno=True)",
        replace="    import ctypes.util\n"
                "    libc = ctypes.CDLL(ctypes.util.find_library(\"c\") or None, use_errno=True)",
        test="test_privilege_drop_does_not_search_for_libc",
        because="find_library execs ldconfig and collect2, which a confined runner is refused",
    ),
]


def _manager(root: Path, interpreter_roots: List[Path]) -> AppArmorManager:
    manager = AppArmorManager(workspace_root=str(root))
    manager._interpreter_roots = list(interpreter_roots)
    return manager


def _workspace(root: Path) -> Path:
    workspace = root / "Test env"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def _body_rules(text: str) -> List[str]:
    """Rule lines of one rendered body (no comments, headers or braces)."""
    rules = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("profile "):
            continue
        if line in ("{", "}") or line.endswith("{"):
            continue
        rules.append(line)
    return rules


def _bodies(root: Path, fragments: Optional[List[str]],
            interpreter_roots: List[Path] = (_UV_ROOT,)) -> Dict[str, str]:
    """Render every body for a workspace under *root*, keyed by name."""
    workspace = _workspace(root)
    manager = _manager(root, list(interpreter_roots))
    text = manager._render_profile("sid", str(workspace),
                                   requested_fragments=fragments)
    hat = text.index("profile tool_hat {")
    child = text.index("profile child {")
    isolated = manager._render_sub_profile("sid", "sub_1", str(workspace))
    return {
        "base": text[:hat],
        "tool_hat": text[hat:child],
        "child": text[child:],
        "isolated": isolated,
        "_all": text,
    }


def _verdict(body: str, path: str, need: str) -> Optional[bool]:
    grants = ConfinementGrants("p", None, _body_rules(body))
    # A line the matcher cannot read (a signal or capability rule it does
    # not classify) would make every answer unknown; drop those, since
    # only file rules decide these paths.
    grants._parsed.unreadable = []
    return grants.verdict(path, need)


def test_the_template_version_moved():
    assert AppArmorManager._TEMPLATE_VERSION >= 45


def test_the_runner_can_open_a_pty_master(tmp_path):
    bodies = _bodies(tmp_path, None)
    for name in ("base", "tool_hat"):
        for need in ("r", "w"):
            assert _verdict(bodies[name], "/dev/ptmx", need) is True, (
                f"{name} cannot open /dev/ptmx for {need}")
        assert _verdict(bodies[name], "/dev/pts/3", "w") is True


def test_the_pty_master_is_not_granted_where_nothing_opens_it(tmp_path):
    for fragments in (None, []):
        bodies = _bodies(tmp_path / str(fragments is None), fragments)
        assert _verdict(bodies["child"], "/dev/ptmx", "w") is False
        assert _verdict(bodies["isolated"], "/dev/ptmx", "w") is False


@pytest.mark.parametrize("fragments", [None, []], ids=["unscoped", "scoped"])
def test_child_reads_python_wheels_and_git_templates(tmp_path, fragments):
    child = _bodies(tmp_path, fragments)["child"]
    for path in (
        "/usr/share/python-wheels/pip-24.0-py3-none-any.whl",
        "/usr/share/git-core/templates/description",
        "/usr/share/git-core/templates/hooks/pre-commit.sample",
    ):
        assert _verdict(child, path, "r") is True, path
        assert _verdict(child, path, "w") is False, path
        assert _verdict(child, path, "x") is False, path


@pytest.mark.parametrize("fragments", [None, []], ids=["unscoped", "scoped"])
def test_every_body_reads_the_interpreter_installation(tmp_path, fragments):
    bodies = _bodies(tmp_path, fragments)
    stdlib = f"{_UV_ROOT}/lib/python3.12/os.py"
    ext = f"{_UV_ROOT}/lib/python3.12/lib-dynload/_ssl.cpython-312-x86_64-linux-gnu.so"
    interp = f"{_UV_ROOT}/bin/python3.12"
    for name in ("base", "tool_hat", "child", "isolated"):
        body = bodies[name]
        assert _verdict(body, stdlib, "r") is True, f"{name}: stdlib"
        assert _verdict(body, ext, "m") is True, f"{name}: extension map"
        assert _verdict(body, interp, "x") is True, f"{name}: interpreter exec"
        assert _verdict(body, stdlib, "w") is False, f"{name}: stdlib write"


def test_a_system_interpreter_adds_no_rule(tmp_path):
    bodies = _bodies(tmp_path, None, interpreter_roots=[])
    for name in ("base", "tool_hat", "child", "isolated"):
        assert "the resolved interpreter installation" not in bodies[name]


def test_the_roots_come_from_the_running_interpreter(tmp_path):
    venv = tmp_path / "venv"
    venv.mkdir()
    ws_root = tmp_path / "workspaces"
    ws_root.mkdir()
    uv = tmp_path / "uv" / "cpython-3.12"
    uv.mkdir(parents=True)
    roots = interpreter_install_roots(
        venv.resolve(), ws_root.resolve(),
        prefixes=[str(uv), str(uv), "/usr", "/usr/local", "/", str(venv / "x")],
    )
    assert roots == [uv.resolve()]
    # The default reads sys.base_prefix / sys.base_exec_prefix.
    import sys
    default = interpreter_install_roots(Path(sys.prefix).resolve())
    for root in default:
        assert root in {Path(sys.base_prefix).resolve(),
                        Path(sys.base_exec_prefix).resolve()}


def test_an_interpreter_inside_a_workspace_is_never_granted(tmp_path):
    ws_root = tmp_path / "workspaces"
    planted = ws_root / "sessions" / "x" / "python"
    planted.mkdir(parents=True)
    roots = interpreter_install_roots(
        (tmp_path / "venv").resolve(), ws_root.resolve(),
        prefixes=[str(planted)],
    )
    assert roots == []
    # And at render time, a root inside the session's own workspace.
    root = tmp_path / "r"
    workspace = _workspace(root)
    inside = workspace / ".python"
    manager = _manager(root, [inside.resolve()])
    text = manager._render_profile("sid", str(workspace))
    assert str(inside.resolve()) + "/bin/*" not in text


def test_privilege_drop_does_not_search_for_libc():
    source = Path(__file__).resolve().parents[2] / "shared" / "privilege_drop.py"
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "find_library":
            pytest.fail(f"privilege_drop.py:{node.lineno} calls find_library")


@pytest.mark.skipif(shutil.which("apparmor_parser") is None,
                    reason="apparmor_parser not installed")
@pytest.mark.parametrize("fragments", [None, []], ids=["unscoped", "scoped"])
def test_the_render_compiles(tmp_path: Path, fragments):
    bodies = _bodies(tmp_path / "ws", fragments)
    for key in ("_all", "isolated"):
        profile = tmp_path / f"profile-{key}"
        profile.write_text(bodies[key])
        result = subprocess.run(
            ["apparmor_parser", "-Q", "-K", str(profile)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
