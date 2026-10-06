"""Every ``python -m <module>`` a test launches names a module that exists (#1462).

``jaato-server/tests/integration/test_phase2_multitenant_apparmor.py``
started its daemon with ``[sys.executable, "-m", "server", ...]``.  The
package is ``jaato_server``, so on every host where the file runs the
daemon exited at once and the test failed before any session existed.
Nobody saw it: the file is marked ``apparmor`` and skips wherever the LSM
is absent, which is every CI runner and every dev container, and a skip
reads like a pass in a summary line.  The #1253 host probe depends on that
test running.

A launch of a module that does not exist is checkable WITHOUT the
environment the test needs, so this guard checks it in ordinary CI: it
parses every test file under the test directories below, finds each list
or tuple literal that starts with ``sys.executable`` and carries ``"-m",
<module>``, and asks a fresh interpreter whether that module resolves.
The module may be a string literal or a module-level string constant of
the same file (the integration test names its daemon ``DAEMON_MODULE``).
A ``-m`` argument this guard cannot resolve to a string is reported, not
skipped, so a launch cannot escape the check by being spelled indirectly.

Only launches through ``sys.executable`` are judged: a literal such as
``["python", "-m", "server", ...]`` in a doctor test is a fabricated
``/proc/<pid>/cmdline``, data rather than a launch.

The probe runs in a subprocess with a neutral working directory, because
the test process's own ``sys.path`` holds pytest's rootdir insertions and
would answer for directories the launched interpreter never sees.  The
child inherits ``PYTHONPATH``, so under the reversion meta-guard it
resolves against the sandbox's copy.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from jaato_server.shared.tests.reversion import Reversion

#: The checkout this file sits in (the sandbox's, under the meta-guard).
ROOT = Path(__file__).resolve().parents[4]

_INTEGRATION = "jaato-server/tests/integration/test_phase2_multitenant_apparmor.py"

REVERSIONS = [
    Reversion(
        target=_INTEGRATION,
        find='DAEMON_MODULE = "jaato_server"',
        replace='DAEMON_MODULE = "server"',
        because="the AppArmor gate launched its daemon as `python -m server`, "
                "a module that no longer exists, and skipped everywhere CI "
                "could have noticed",
        test="test_every_dash_m_launch_names_an_importable_module",
    ),
]

#: Directories holding test files that may launch interpreters.
_TEST_DIRS = (
    "jaato-server/tests",
    "jaato-server/jaato_server",
    "jaato-sdk/jaato_sdk",
    "jaato-tui",
)

_SKIP_PARTS = {".venv", "venv", "node_modules", "__pycache__", ".git"}


def _test_files() -> List[Path]:
    files = []
    for rel in _TEST_DIRS:
        base = ROOT / rel
        if not base.is_dir():
            continue
        for path in base.rglob("test_*.py"):
            if _SKIP_PARTS.intersection(path.relative_to(ROOT).parts):
                continue
            files.append(path)
    return sorted(files)


def _is_sys_executable(node: ast.AST) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == "executable"
            and isinstance(node.value, ast.Name) and node.value.id == "sys")


def _module_constants(tree: ast.Module) -> Dict[str, str]:
    """Module-level ``NAME = "string"`` assignments."""
    consts: Dict[str, str] = {}
    for stmt in tree.body:
        targets: List[ast.expr] = []
        value: Optional[ast.expr] = None
        if isinstance(stmt, ast.Assign):
            targets, value = stmt.targets, stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            targets, value = [stmt.target], stmt.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            for t in targets:
                if isinstance(t, ast.Name):
                    consts[t.id] = value.value
    return consts


def _launches(path: Path) -> Tuple[List[Tuple[str, int, str]], List[str]]:
    """``(file, line, module)`` per ``sys.executable -m`` launch, plus the
    locations whose module could not be resolved to a string."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    consts = _module_constants(tree)
    rel = str(path.relative_to(ROOT))
    found: List[Tuple[str, int, str]] = []
    unresolved: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.List, ast.Tuple)) or not node.elts:
            continue
        if not _is_sys_executable(node.elts[0]):
            continue
        elts = node.elts
        for i, elt in enumerate(elts[:-1]):
            if not (isinstance(elt, ast.Constant) and elt.value == "-m"):
                continue
            arg = elts[i + 1]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                found.append((rel, node.lineno, arg.value))
            elif isinstance(arg, ast.Name) and arg.id in consts:
                found.append((rel, node.lineno, consts[arg.id]))
            else:
                unresolved.append(f"{rel}:{node.lineno}")
            break
    return found, unresolved


def _all_launches() -> Tuple[List[Tuple[str, int, str]], List[str]]:
    found: List[Tuple[str, int, str]] = []
    unresolved: List[str] = []
    for path in _test_files():
        f, u = _launches(path)
        found.extend(f)
        unresolved.extend(u)
    return found, unresolved


def _unresolvable_modules(modules: List[str], cwd: Path) -> List[str]:
    probe = (
        "import importlib.util, json, sys\n"
        "out = []\n"
        "for m in json.loads(sys.argv[1]):\n"
        "    try:\n"
        "        ok = importlib.util.find_spec(m) is not None\n"
        "    except (ImportError, ValueError):\n"
        "        ok = False\n"
        "    if not ok:\n"
        "        out.append(m)\n"
        "print(json.dumps(out))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe, json.dumps(sorted(set(modules)))],
        capture_output=True, text=True, timeout=120, cwd=str(cwd),
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_the_apparmor_gate_is_among_the_launches_found() -> None:
    """The scan reaches the file the defect was in.

    Without this, a scan that found nothing (a wrong root, a renamed
    directory) would pass the next test vacuously.
    """
    found, _ = _all_launches()
    assert any(rel == _INTEGRATION for rel, _, _ in found), found


def test_every_dash_m_launch_names_an_importable_module(tmp_path: Path) -> None:
    found, unresolved = _all_launches()
    assert not unresolved, (
        "`sys.executable -m <x>` launches whose module is not a string "
        f"literal or a module-level string constant: {unresolved}"
    )
    missing = set(_unresolvable_modules([m for _, _, m in found], tmp_path))
    bad = [f"{rel}:{line} -m {mod}" for rel, line, mod in found
           if mod in missing]
    assert not bad, (
        "tests launch modules that do not exist (the daemon is "
        f"`python -m jaato_server`): {bad}"
    )
