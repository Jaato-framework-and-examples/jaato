"""Entry points of ``jaato-doctor`` and ``jaato-scaffold`` (#1643).

Both tools introspect the framework installed beside their own
interpreter.  Inside a session they are reached through symlinks to the
daemon venv's console scripts (``jaato_server.shared.plugins.jaato_tools_path``,
#1273), and a symlink cannot reset the environment: the command inherits the
subprocess ``PYTHONPATH``, which a workspace venv prepends its site-packages
to (``apply_venv_to_env``).  When that venv held another ``jaato-sdk`` (an
agent writing a cascade driver installs one), the daemon's own
``jaato-scaffold`` imported the WORKSPACE's ``jaato_sdk``, exited 0 and
described a framework that is not the installed one: three plugins dropped
out with only a line on stderr each.

The fix is here, at the first import, because it is the one place every
launch passes through whatever started it (``cli``, ``interactive_shell``,
a notebook cell's ``!jaato-scaffold``).  :func:`isolate` removes from
``sys.path`` every ``PYTHONPATH`` entry the session did not inherit from its
runner, before ``jaato_sdk`` is imported, and narrows ``PYTHONPATH`` the
same way so the tool's own child processes agree.

What is trusted, and who says so
--------------------------------
The session's subprocess environment carries
:data:`TRUSTED_PYTHONPATH_ENV`, stamped by the runner with ITS OWN
``PYTHONPATH`` (usually empty).  Those entries are what the runner -- and so
the framework the session runs on -- imports from, so they stay.  Anything
else on ``PYTHONPATH`` (the workspace venv, a ``PYTHONPATH=$WS`` prefix in
the command) is dropped.

With no marker the launch is not a session's (a developer's shell, a CI
job) and nothing changes: a ``PYTHONPATH`` set there is the caller's own
choice of framework, which ``jaato-doctor``'s client-side checks exist to
diagnose.

This module is deliberately TOP-LEVEL and not inside ``jaato_sdk``: the
console script imports it before anything else, and ``from jaato_sdk...``
would already have loaded the shadowing package's ``__init__``.  A
``jaato-sdk`` older than this module does not ship it, so an older shadow
cannot replace it; a newer one ships the same rule.

Not a security boundary: the model can unset the marker or run
``python -m jaato_sdk.scaffold.cli`` itself.  What it fixes is the accident
the ``jaato-sdk`` skill invites -- the tool the model is told to trust
answering for whatever it last installed.
"""

from __future__ import annotations

import os
import sys
from typing import List, MutableMapping, Optional

#: Set by the runner on a model-driven subprocess environment to the
#: runner's own ``PYTHONPATH`` (possibly empty).  Its presence is what makes
#: a launch a session's; its value is the set of entries that stay.
TRUSTED_PYTHONPATH_ENV = "JAATO_TOOLS_TRUSTED_PYTHONPATH"


def _norm(entry: str) -> str:
    """One spelling per directory, so a trailing slash is not a new entry."""
    return os.path.normcase(os.path.abspath(entry))


def _entries(value: str) -> List[str]:
    return [e for e in value.split(os.pathsep) if e]


def _own_dirs() -> set:
    """The interpreter's own library directories, never removed.

    A ``PYTHONPATH`` that happens to name one of them (someone added the
    daemon venv's site-packages by hand) must not take it away.
    """
    import sysconfig
    dirs = set()
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        try:
            path = sysconfig.get_path(key)
        except (KeyError, ValueError):
            path = None
        if path:
            dirs.add(_norm(path))
    return dirs


def _untrusted(environ: MutableMapping[str, str]) -> set:
    """The normalised ``PYTHONPATH`` entries the marker does not name.

    Empty when the launch carries no marker (not a session's) or every entry
    is trusted or one of the interpreter's own library directories.
    """
    trusted_raw = environ.get(TRUSTED_PYTHONPATH_ENV)
    if trusted_raw is None:
        return set()
    trusted = {_norm(e) for e in _entries(trusted_raw)}
    given = {_norm(e) for e in _entries(environ.get("PYTHONPATH", ""))}
    return given - trusted - _own_dirs()


def _narrow_pythonpath(environ: MutableMapping[str, str], untrusted: set) -> None:
    """Keep only trusted ``PYTHONPATH`` entries, so the tool's children agree."""
    kept = [e for e in _entries(environ.get("PYTHONPATH", ""))
            if _norm(e) not in untrusted]
    if kept:
        environ["PYTHONPATH"] = os.pathsep.join(kept)
    else:
        environ.pop("PYTHONPATH", None)


def isolate(
    environ: Optional[MutableMapping[str, str]] = None,
    path: Optional[List[str]] = None,
) -> List[str]:
    """Drop the ``PYTHONPATH`` entries a session did not inherit from its runner.

    Mutates ``path`` (default ``sys.path``) and ``environ`` (default
    ``os.environ``) in place, before any jaato module is imported.  Returns
    the ``sys.path`` entries removed, in order; an empty list when the launch
    carries no marker or nothing untrusted was on ``PYTHONPATH``.
    """
    environ = os.environ if environ is None else environ
    path = sys.path if path is None else path
    untrusted = _untrusted(environ)
    if not untrusted:
        return []
    removed = [e for e in path if e and _norm(e) in untrusted]
    path[:] = [e for e in path if e not in removed]
    _narrow_pythonpath(environ, untrusted)
    if removed:
        import importlib
        importlib.invalidate_caches()
    return removed


def doctor() -> int:
    """``jaato-doctor``: :func:`isolate`, then ``jaato_sdk.doctor.main``."""
    isolate()
    from jaato_sdk.doctor import main
    return main()


def scaffold() -> int:
    """``jaato-scaffold``: :func:`isolate`, then ``jaato_sdk.scaffold.cli.main``."""
    isolate()
    from jaato_sdk.scaffold.cli import main
    return main()
