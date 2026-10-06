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

#: Paths per write in :meth:`_CheckIgnore.is_ignored_many`: small enough
#: that their records fit git's stdout pipe while we are not reading it.
_BATCH = 128

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
        # Bytes git wrote past the last record read (see ``_read_record``).
        self._buf = bytearray()

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

    def _read_record(self, proc: subprocess.Popen) -> Optional[List[bytes]]:
        """Read one four-field record, or ``None`` if the process died mid-record.

        Reads in chunks and keeps what follows the record for the next
        call.  It used to read ONE BYTE per Python call, about a hundred
        calls per path; a baseline walk asks for every file in a clone,
        and several sessions walking at once contended for the GIL with
        nothing else, which is how four ``session.new`` took a minute
        (#1553).
        """
        assert proc.stdout is not None
        while self._buf.count(b"\0") < 4:
            chunk = proc.stdout.read1(65536)
            if not chunk:
                self._buf.clear()
                return None
            self._buf += chunk
        fields = []
        for _ in range(4):
            end = self._buf.index(b"\0")
            fields.append(bytes(self._buf[:end]))
            del self._buf[:end + 1]
        return fields

    def is_ignored(self, rel_posix: str) -> Optional[bool]:
        """``True`` / ``False`` for a repo-relative POSIX path, ``None`` if git could not answer."""
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._proc = self._spawn()
                self._buf.clear()
            proc = self._proc
            if proc is None or proc.stdin is None:
                return None
            try:
                proc.stdin.write(rel_posix.encode("utf-8", "surrogateescape") + b"\0")
                proc.stdin.flush()
                fields = self._read_record(proc)
            except (BrokenPipeError, OSError):
                self._proc = None
                return None
            if fields is None:
                self._proc = None
                return None
            pattern = fields[2].decode("utf-8", "surrogateescape")
            # Non-empty pattern not starting with '!' → the path is ignored.
            # An empty pattern (no rule matched) or a '!' negation → not.
            return bool(pattern) and not pattern.startswith("!")

    def is_ignored_many(self, rels: List[str]) -> List[Optional[bool]]:
        """:meth:`is_ignored` for many paths, in one exchange per chunk.

        A baseline walk asks about a whole directory at once.  One round
        trip per FILE meant one blocking read per file, and with several
        sessions walking at once each read waited out the interpreter's
        switch interval to get the GIL back (#1553).  Paths are written
        ``_BATCH`` at a time and their records read back before the next
        chunk, so git's stdout pipe cannot fill while we are still writing
        its stdin.
        """
        out: List[Optional[bool]] = []
        with self._lock:
            for start in range(0, len(rels), _BATCH):
                chunk = rels[start:start + _BATCH]
                if self._proc is None or self._proc.poll() is not None:
                    self._proc = self._spawn()
                    self._buf.clear()
                proc = self._proc
                if proc is None or proc.stdin is None:
                    out.extend([None] * len(chunk))
                    continue
                try:
                    proc.stdin.write(b"".join(
                        r.encode("utf-8", "surrogateescape") + b"\0"
                        for r in chunk))
                    proc.stdin.flush()
                    for _ in chunk:
                        fields = self._read_record(proc)
                        if fields is None:
                            raise OSError("check-ignore died mid-batch")
                        pattern = fields[2].decode("utf-8", "surrogateescape")
                        out.append(bool(pattern) and not pattern.startswith("!"))
                except (BrokenPipeError, OSError):
                    self._proc = None
                    self._buf.clear()
                    out.extend([None] * (len(chunk) - (len(out) - start)))
        return out

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
        self._root_str = str(self._root)
        self._lock = threading.Lock()
        self._checks: Dict[Path, _CheckIgnore] = {}
        self._checkouts: List[Path] = []
        # ``(checkout, str(checkout) + os.sep)``, longest first, for the
        # lookup below: a string prefix test per path, where
        # ``Path.relative_to`` per checkout per path was most of a baseline
        # walk's cost (#1553).
        self._prefixes: List[tuple] = []
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
        self._prefixes = [(c, str(c) + os.sep) for c in self._checkouts]

    def _checkout_for(self, abs_path: str) -> Optional[Path]:
        for checkout, prefix in self._prefixes:
            if abs_path.startswith(prefix):
                return checkout
        return None

    def _new_checkout_for(self, abs_path: str) -> bool:
        """Whether *abs_path* lies in a checkout that appeared after the scan.

        Only the path's own top-level child can be such a checkout, so that
        one ``.git`` is looked for.  The old rule rescanned the whole root
        (a listdir and a stat per child) for EVERY path outside a checkout,
        which a walk of a workspace whose files are not in a clone paid
        once per file (#1553).
        """
        if not abs_path.startswith(self._root_str + os.sep):
            return False
        first = abs_path[len(self._root_str) + 1:].split(os.sep, 1)
        if len(first) < 2 or not first[0] or first[0].startswith("."):
            return False  # a file directly under the root, or a hidden dir
        child = self._root / first[0]
        if child in self._checkouts or not (child / ".git").exists():
            return False
        self._scan_checkouts()
        return True

    def is_ignored(self, abs_path: str) -> Optional[bool]:
        """Git's verdict for *abs_path*, or ``None`` if no checkout owns it."""
        abs_path = os.fspath(abs_path)
        with self._lock:
            checkout = self._checkout_for(abs_path)
            if checkout is None and self._new_checkout_for(abs_path):
                checkout = self._checkout_for(abs_path)  # a clone appeared
            if checkout is None:
                return None
            check = self._checks.get(checkout)
            if check is None:
                check = _CheckIgnore(checkout)
                self._checks[checkout] = check
        rel = abs_path[len(str(checkout)) + 1:].replace(os.sep, "/")
        if not rel or rel == ".":
            return None
        return check.is_ignored(rel)

    def is_ignored_many(self, abs_paths: List[str]) -> List[Optional[bool]]:
        """:meth:`is_ignored` for several paths, one git exchange per checkout."""
        verdicts: List[Optional[bool]] = [None] * len(abs_paths)
        groups: Dict[Path, List[tuple]] = {}
        with self._lock:
            for i, abs_path in enumerate(abs_paths):
                abs_path = os.fspath(abs_path)
                checkout = self._checkout_for(abs_path)
                if checkout is None and self._new_checkout_for(abs_path):
                    checkout = self._checkout_for(abs_path)
                if checkout is None:
                    continue
                rel = abs_path[len(str(checkout)) + 1:].replace(os.sep, "/")
                if rel and rel != ".":
                    groups.setdefault(checkout, []).append((i, rel))
            checks = {}
            for checkout in groups:
                check = self._checks.get(checkout)
                if check is None:
                    check = _CheckIgnore(checkout)
                    self._checks[checkout] = check
                checks[checkout] = check
        for checkout, items in groups.items():
            answers = checks[checkout].is_ignored_many([rel for _, rel in items])
            for (i, _), answer in zip(items, answers):
                verdicts[i] = answer
        return verdicts

    def close(self) -> None:
        with self._lock:
            checks = list(self._checks.values())
            self._checks.clear()
        for check in checks:
            check.close()
