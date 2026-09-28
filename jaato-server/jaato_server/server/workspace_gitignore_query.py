"""Ask git whether a path is ignored, so the Files panel agrees with it.

The workspace monitor filtered paths through :class:`GitignoreParser`, which
reads only ``<workspace>/.gitignore``.  A workspace is not a repository; the
web coder clones each project to ``<ws>/<name>``, and each clone has its own
``.gitignore`` and ``.git/info/exclude`` that the parser never saw.  So a
clone's ``node_modules/`` and ``build/`` showed up in the panel, however
correctly the project ignored them.

This oracle answers for a path INSIDE a checkout by asking that checkout's
own git, through one long-lived ``git check-ignore --stdin`` process per
checkout (~40us per query once warm, measured).  A path outside every
checkout keeps the parser answer, so the workspace-home extras
(``.home/``, ``.tmp/``, the tool venv) and the ``.git/`` default still apply
everywhere.

**Every git call is hardened**, because a checkout is model-writable and its
``.git/config`` is not.  A repo-local config value such as ``core.fsmonitor``
or ``core.hooksPath`` makes even a read-only git call run a program the
config names, so the process is launched with global and system config
neutralised (``GIT_CONFIG_GLOBAL`` / ``GIT_CONFIG_SYSTEM`` = ``/dev/null``)
and with the hook-bearing knobs forced off on the command line
(``-c core.fsmonitor=false -c core.hooksPath=/dev/null``).  ``check-ignore``
reads a repository's own ``.gitignore`` and ``.git/info/exclude`` — which is
exactly the view we want — and touches no network.

A checkout here is the same set the panel already means: the workspace root
when it holds ``.git``, and each immediate non-hidden child that holds
``.git`` (matching :mod:`workspace_sources`).  The set is resolved once when
the oracle is built and refreshed when git reports a path it cannot place
(a clone that appeared mid-session), bounded to one refresh per miss.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

#: How many immediate children to consider as checkouts (the panel's own
#: bound; a workspace with more than this many top-level directories is not
#: the shape this serves).
_MAX_CHECKOUTS = 256


def _hardened_git_env() -> Dict[str, str]:
    """Environment for a read-only git that trusts no config but the repo's own.

    Global and system config are the two files this process's account
    controls, so a repo cannot reach a hook through them; the repo's own
    ``.git/config`` is neutralised on the command line instead, since git
    must still read the repository to answer at all.
    """
    env = dict(os.environ)
    env.update({
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
        "LC_ALL": "C",
    })
    return env


#: Command-line overrides for the config knobs a repo-local value could use to
#: run a program.  ``-c`` beats the repo's ``.git/config``.
_HARDENING_FLAGS = (
    "-c", "core.fsmonitor=false",
    "-c", "core.hooksPath=/dev/null",
)


class _CheckIgnore:
    """One long-lived ``git check-ignore`` process for a single checkout.

    ``--stdin -z -v -n`` streams a NUL-separated path in and a four-field
    NUL-separated record out per path: ``source``, ``linenum``, ``pattern``,
    ``pathname``.  ``-n`` (non-matching) makes git answer for every path, so
    the record is present whether or not the path is ignored, and the verdict
    is read from the fields: a match with a non-empty ``pattern`` that does
    not start with ``!`` means ignored.
    """

    def __init__(self, checkout: Path):
        self._checkout = checkout
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def _spawn(self) -> Optional[subprocess.Popen]:
        try:
            return subprocess.Popen(
                ["git", "-C", str(self._checkout), *_HARDENING_FLAGS,
                 "check-ignore", "--stdin", "-z", "-v", "-n"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, env=_hardened_git_env(),
            )
        except OSError as exc:  # git missing, or the checkout vanished
            logger.debug("check-ignore spawn failed for %s: %s", self._checkout, exc)
            return None

    def _read_field(self, proc: subprocess.Popen) -> Optional[bytes]:
        buf = bytearray()
        assert proc.stdout is not None
        while True:
            byte = proc.stdout.read(1)
            if not byte:
                return None  # process died mid-record
            if byte == b"\0":
                return bytes(buf)
            buf += byte

    def is_ignored(self, rel_posix: str) -> Optional[bool]:
        """``True`` / ``False`` for a repo-relative POSIX path, ``None`` if git could not answer."""
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._proc = self._spawn()
            proc = self._proc
            if proc is None or proc.stdin is None:
                return None
            try:
                proc.stdin.write(rel_posix.encode("utf-8", "surrogateescape") + b"\0")
                proc.stdin.flush()
                fields = [self._read_field(proc) for _ in range(4)]
            except (BrokenPipeError, OSError):
                self._proc = None
                return None
            if any(f is None for f in fields):
                self._proc = None
                return None
            pattern = fields[2].decode("utf-8", "surrogateescape")
            # Non-empty pattern not starting with '!' → the path is ignored.
            # An empty pattern (no rule matched) or a '!' negation → not.
            return bool(pattern) and not pattern.startswith("!")

    def close(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()


class GitIgnoreOracle:
    """Answers ``is_ignored`` for a workspace, per-checkout, via git.

    Thread-safe.  ``is_ignored`` returns ``None`` for a path that belongs to
    no checkout or that git could not place, and the caller falls back to its
    own :class:`GitignoreParser`.
    """

    def __init__(self, workspace_path: str):
        self._root = Path(os.path.abspath(workspace_path))
        self._lock = threading.Lock()
        self._checks: Dict[Path, _CheckIgnore] = {}
        self._checkouts: List[Path] = []
        self._scan_checkouts()

    def _scan_checkouts(self) -> None:
        found: List[Path] = []
        if (self._root / ".git").exists():
            found.append(self._root)
        try:
            names = sorted(os.listdir(self._root))
        except OSError:
            names = []
        for name in names[:_MAX_CHECKOUTS]:
            if name.startswith("."):
                continue
            child = self._root / name
            if (child / ".git").exists():
                found.append(child)
        # Longest path first so a nested checkout wins over its parent.
        self._checkouts = sorted(found, key=lambda p: len(p.parts), reverse=True)

    def _checkout_for(self, abs_path: Path) -> Optional[Path]:
        for checkout in self._checkouts:
            try:
                abs_path.relative_to(checkout)
                return checkout
            except ValueError:
                continue
        return None

    def is_ignored(self, abs_path: str) -> Optional[bool]:
        """Git's verdict for *abs_path*, or ``None`` if no checkout owns it."""
        p = Path(abs_path)
        with self._lock:
            checkout = self._checkout_for(p)
            if checkout is None:
                self._scan_checkouts()  # a clone may have appeared
                checkout = self._checkout_for(p)
            if checkout is None:
                return None
            check = self._checks.get(checkout)
            if check is None:
                check = _CheckIgnore(checkout)
                self._checks[checkout] = check
        rel = p.relative_to(checkout).as_posix()
        if not rel or rel == ".":
            return None
        return check.is_ignored(rel)

    def close(self) -> None:
        with self._lock:
            checks = list(self._checks.values())
            self._checks.clear()
        for check in checks:
            check.close()
