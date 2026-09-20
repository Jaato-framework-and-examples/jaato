"""A session a test built must not be visible to the next test.

WHAT WENT WRONG (#974).  ``shared.session_context._current_session`` is
a process-global ``ContextVar``, and ``JaatoSession`` publishes ``self``
into it from three places -- ``configure()`` and both tool-dispatch
paths.  Nothing ever took it back out.

Dozens of guards in this tree drive those paths against a session built
with ``JaatoSession.__new__(JaatoSession)``: a shell with the two or
three attributes the contract under test reads and none of the rest.
The first such test to run therefore left a half-built ``JaatoSession``
visible to every test that ran after it, in any package, for the rest of
the pytest process.

``PermissionPlugin`` is the reader that found it.  It resolves its
per-session policy off the current session (#957) and asks it for
``reliability:escalated_tools`` through ``get_session_state`` -- an
attribute the shell does not have, because ``_state_providers`` is
assigned in ``__init__``::

    AttributeError: 'JaatoSession' object has no attribute '_state_providers'

Measured on ``main`` at ``4e5c95ba``, one pytest process each::

    shared/plugins/permission/                      505 passed,     0 failed
    test_unreadable_tool_args_are_not_executed.py
      + shared/plugins/permission/                  429 passed,   113 failed
    shared/tests/ + shared/plugins/permission/     5534 passed,   118 failed
      (the same, with this fix)                    5653 passed,     5 failed

The last pair is the whole scope, with ``--continue-on-collection-errors``
and ``test_every_guard_detects_its_own_reversion.py`` left out -- it
spawns a pytest subprocess per case and measures nothing about a
session.  118 - 113 = 5, and those five are
``test_jaato_session.py::TestForceNarrationBetweenToolsQuirk``, which
fail identically when that file is run on its own: a different,
pre-existing defect that this change neither causes nor fixes.

WHY IT IS WORSE THAN A NOISY RUN.  The damage lands hundreds of tests
away from its cause, in files that are individually clean, so the
obvious reading is "the permission package is broken".  And it destroys
the measurement several changes in this repository are evaluated by: a
real one- or two-test regression is invisible against a hundred-failure
floor that appears only in combination.  The issue's own advice -- diff
failing test *ids*, never totals -- is a workaround for this bug, not a
law of nature.

THE THREE PIECES, and each is guarded below because each can be removed
on its own:

1. :func:`shared.session_context.isolated_current_session` -- the one
   definition of "put the variable back", including back to *unset*,
   which only the ``Token`` of the originating ``set()`` can express.
2. The leak site itself: the fixture in
   ``test_unreadable_tool_args_are_not_executed.py`` that wraps its
   shell.  Fixing it where it is set is the actual fix.
3. The autouse net in ``jaato-server/conftest.py``, so the *next* leak
   cannot escape the test that causes it.  Defence in depth, not a
   substitute for (2).

DELIBERATELY NOT DONE: making ``get_session_state`` defensive with
``getattr(self, "_state_providers", {})``.  The ``AttributeError`` is
the only thing that makes this class of leak visible at all; silencing
it would leave a shell session answering permission questions for a
whole test run with nobody the wiser.
"""

from __future__ import annotations

import ast
import os
import pathlib
import subprocess
import sys

import pytest

from shared.session_context import (
    _current_session,
    get_current_session,
    isolated_current_session,
    set_current_session,
)
from shared.tests.reversion import Reversion


ROOT = pathlib.Path(__file__).resolve().parents[3]
CONFTEST = ROOT / "jaato-server" / "conftest.py"
LEAK_SITE = (
    ROOT / "jaato-server" / "shared" / "tests"
    / "test_unreadable_tool_args_are_not_executed.py"
)

#: The victim used by the end-to-end pin.  ``test_scoped_policy_957.py``
#: is the permission file whose whole point is resolving a policy off
#: the current session, so it is the one that goes red first and the one
#: a reader should be pointed at.
_VICTIM = (
    "jaato-server/shared/plugins/permission/tests/test_scoped_policy_957.py"
)
_LEAKER = (
    "jaato-server/shared/tests/"
    "test_unreadable_tool_args_are_not_executed.py"
)


# ==================== The primitive ====================


def test_the_helper_returns_an_unset_variable_to_unset():
    """The case a save-and-restore cannot do.

    ``ContextVar`` has no delete.  Reading the value and writing it back
    afterwards leaves the variable *set* when it started unset, which is
    a different state: ``get_current_session`` raises ``LookupError``
    for one and not the other, and every caller in the tree branches on
    exactly that.
    """
    with pytest.raises(LookupError):
        get_current_session()

    with isolated_current_session():
        set_current_session(object())
        assert get_current_session() is not None

    with pytest.raises(LookupError):
        get_current_session()


def test_the_helper_restores_a_previous_session():
    """Nesting must not clear an outer session either."""
    outer = object()
    with isolated_current_session():
        set_current_session(outer)
        with isolated_current_session():
            set_current_session(object())
        assert get_current_session() is outer


def test_none_reads_as_no_session():
    """The branch that makes the restore expressible.

    :func:`isolated_current_session` writes ``None`` to hold a token it
    can reset from.  If ``get_current_session`` handed that ``None``
    back to a caller instead of raising, every ``except LookupError``
    guard in the tree would fall through and call a method on ``None`` --
    the same crash as the leak, from the fix.
    """
    token = _current_session.set(None)
    try:
        with pytest.raises(LookupError):
            get_current_session()
    finally:
        _current_session.reset(token)


# ==================== The leak site ====================


def _fixture_source(path: pathlib.Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(
                path.read_text(encoding="utf-8"), node) or ""
    raise AssertionError(
        f"{path.name} no longer defines {name}() -- if the shell moved, "
        "this guard must move with it, not be deleted"
    )


def test_the_shell_is_handed_out_only_under_isolation():
    """The fix at the site that sets the variable.

    Asserted on the source rather than on behaviour because the autouse
    net in ``conftest.py`` would mask a behavioural check: with the net
    in place, removing this fixture's isolation changes nothing anyone
    can observe from inside the suite -- right up until someone runs
    those tests without the net, or adds a second leak the net has to
    absorb.  Two independent pieces, two independent guards.
    """
    source = _fixture_source(LEAK_SITE, "stub_session")
    assert "isolated_current_session" in source, (
        "stub_session() hands out a JaatoSession shell whose dispatch "
        "paths call set_current_session(self); without "
        "isolated_current_session() that shell outlives the test (#974)"
    )


def test_the_package_conftest_installs_the_net():
    """Defence in depth, and it must be autouse to be depth at all."""
    source = CONFTEST.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        body = ast.get_source_segment(source, node) or ""
        if "isolated_current_session" not in body:
            continue
        decorators = [ast.get_source_segment(source, d) or ""
                      for d in node.decorator_list]
        assert any("autouse=True" in d for d in decorators), (
            f"{node.name}() restores the session ContextVar but is not "
            "autouse, so it protects only the tests that ask for it"
        )
        return
    raise AssertionError(
        "jaato-server/conftest.py declares no fixture restoring the "
        "current-session ContextVar; a leak in any one test then reaches "
        "every package that runs after it (#974)"
    )


# ==================== The end-to-end pin ====================


def _run(args: list) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / name) for name in ("jaato-server", "jaato-sdk")]
        + [env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *args],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=900,
    )


def test_a_shell_session_does_not_poison_the_next_package():
    """The property the issue is about, measured rather than argued.

    A subprocess, because the leak is per-process state and this process
    has already run the isolation above.  The two files are given in the
    order that reproduced the failure -- the leaker first.  Before the
    fix this produced ``AttributeError: 'JaatoSession' object has no
    attribute '_state_providers'`` from ``PermissionPlugin``; the
    assertion names that string so a future failure for some unrelated
    reason does not read as this bug returning.
    """
    proc = _run([_LEAKER, _VICTIM])
    assert proc.returncode == 0, (
        "running a session-shell guard ahead of the permission package "
        "still poisons it (#974)\n"
        f"--- stdout ---\n{proc.stdout[-4000:]}\n"
        f"--- stderr ---\n{proc.stderr[-2000:]}"
    )
    assert "_state_providers" not in proc.stdout


REVERSIONS = [
    Reversion(
        target="jaato-server/shared/session_context.py",
        find=(
            "    session = _current_session.get(None)\n"
            "    if session is None:\n"
            "        raise LookupError('current_session')\n"
            "    return session"
        ),
        replace="    return _current_session.get()",
        because=(
            "without the None branch, the value isolated_current_session "
            "writes to hold its token is handed to callers as a session, "
            "and every `except LookupError` guard falls through onto None"
        ),
        test="test_none_reads_as_no_session",
    ),
    Reversion(
        target=(
            "jaato-server/shared/tests/"
            "test_unreadable_tool_args_are_not_executed.py"
        ),
        find=(
            "    with isolated_current_session():\n"
            "        yield _session_with_stub_executor()"
        ),
        replace="    yield _session_with_stub_executor()",
        because=(
            "the shell's dispatch paths call set_current_session(self); "
            "unwrapped, the shell outlives the test that built it"
        ),
        test="test_the_shell_is_handed_out_only_under_isolation",
    ),
    Reversion(
        target="jaato-server/conftest.py",
        find="""@pytest.fixture(autouse=True)
def isolated_session_context():""",
        replace="""@pytest.fixture
def isolated_session_context():""",
        because=(
            "a net that is not autouse protects only the tests that ask "
            "for it, which is none of them"
        ),
        test="test_the_package_conftest_installs_the_net",
    ),
]
