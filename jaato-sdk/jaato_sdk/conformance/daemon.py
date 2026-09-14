"""Start a real daemon, wait until it can actually answer, tear it down.

The fixture is the load-bearing part of a live suite: a flaky daemon start
produces failures indistinguishable from the defects the suite is looking for,
which is worse than no suite.  Three rules follow from that, and each is here
because the alternative fails silently:

* **Readiness is a successful CONNECT, never a sleep and never the socket
  file's existence.**  A stale socket file is present and dead — that state is
  the single thing ``jaato_sdk.doctor`` exists to detect — so a fixture that
  waits for the path would proceed against a corpse.
* **Every wait is bounded and says what it was waiting for.**  A hung daemon
  must fail the job in seconds with a named cause, not hold a CI runner until
  the platform kills it with nothing to read.
* **Teardown is unconditional and escalates.**  A leaked daemon holds its
  socket, and the next run inherits a process that is not the one it thinks it
  is testing — the failure mode that makes a suite lie about which code it
  exercised.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

#: Seconds to wait for a cold daemon to accept a connection.  Generous
#: because CI runners are slow and plugin discovery is real work; bounded
#: because the alternative to a bound is a job that hangs.
STARTUP_TIMEOUT = 90.0

#: Seconds between connect attempts while waiting.
POLL_INTERVAL = 0.25


#: The modules the daemon needs, and which tree they must come from.
#: ``jaato_sdk`` is what the test code drives the daemon THROUGH; ``server``
#: and ``shared`` are what the daemon IS.  A suite in which these resolve to
#: a checkout nobody chose is asserting about a tree nobody ran (jaato #1050).
TREE_MODULES = ("jaato_sdk", "server", "shared")

#: The two server-tier packages, and the directory that holds them in a
#: source checkout of this repository.
_SERVER_PACKAGES = ("server", "shared")
_SERVER_SIBLING = "jaato-server"

#: Printed by the child, with the env the daemon is about to be given, in the
#: cwd the daemon is about to be given -- both matter, since ``-c`` and
#: ``-m`` alike put the cwd on ``sys.path`` ahead of ``PYTHONPATH``.  It
#: RESOLVES rather than imports: importing ``server`` costs plugin discovery,
#: and a preflight slower than the thing it checks will be deleted.
_PREFLIGHT_SRC = """\
import importlib.util, json
out = {}
for name in %r:
    try:
        spec = importlib.util.find_spec(name)
    except BaseException:
        spec = None
    out[name] = spec.origin if spec is not None and spec.origin else None
print(json.dumps(out))
"""

#: Seconds allowed for the preflight.  It is one interpreter start and three
#: ``find_spec`` calls; a bound this generous can only be hit by a machine in
#: trouble, and the alternative to a bound is a fixture that hangs.
PREFLIGHT_TIMEOUT = 30.0


class DaemonTreeMismatch(RuntimeError):
    """The daemon would import a different checkout than the tests came from.

    Raised BEFORE the daemon is started, because the failure it prevents is
    not a crash -- it is a suite that passes or fails about code it never
    ran.  Names both sides, since the whole difficulty of jaato #1050 was
    that nothing in the output said which tree had been loaded.
    """


def _resolve(name: str) -> Optional[str]:
    """Where does THIS process resolve *name*, without importing it?

    Resolution, not import: the fixture must be able to ask where ``server``
    lives without paying plugin discovery for the answer, and without it
    depending on whether some earlier test happened to import it.

    ``None`` for anything this helper cannot answer for -- a namespace
    package, a zipimport, a layout it does not understand.  ``find_spec``
    imports parent packages, so it can raise anything an import can raise,
    and a fixture helper is not worth a collection failure.
    """
    try:
        spec = importlib.util.find_spec(name)
    except BaseException:
        return None
    if spec is None or not spec.origin:
        return None
    return str(Path(spec.origin).resolve())


def _package_root(origin: str) -> str:
    """The directory to put on ``PYTHONPATH`` so *origin* is importable.

    For a package (``<root>/pkg/__init__.py``) that is the grandparent; for a
    single-module file it is the parent.  Anything else would put the package
    itself on the path, which imports its submodules as top-level names.
    """
    path = Path(origin).resolve()
    return str(path.parent.parent if path.name == "__init__.py"
               else path.parent)


def tree_roots() -> Dict[str, str]:
    """``{"sdk": <dir>, "server": <dir>}`` -- the tree the tests came from.

    THE ANCHOR IS ``jaato_sdk``, and everything follows from it.  That module
    is where this very file was imported from, so it identifies the checkout
    the test code came from BY CONSTRUCTION -- no configuration, no guessing
    which of several trees the author meant.

    The server tier is then taken from the SAME checkout when one is there
    (``<checkout>/jaato-server`` holding both ``server`` and ``shared``).
    That is a layout assumption, and a deliberate one: the alternative --
    "wherever ``server`` happens to resolve" -- is what the defect already
    does.  Measured in the configuration this was found in: pytest running
    the SDK leg from a worktree inserts ``<worktree>/jaato-sdk`` on
    ``sys.path`` and NOT ``<worktree>/jaato-server``, so ``jaato_sdk``
    resolves to the worktree while ``server`` and ``shared`` still resolve
    through the editable install.  A fix that merely propagated what this
    process resolves would faithfully reproduce that split.

    Nothing is lost where the assumption does not hold: an installed tree
    with no co-located checkout falls back to resolution, which is correct
    there because there is only one of everything.

    Safe here specifically because the conformance package imports neither
    ``server`` nor ``shared`` in-process -- the daemon is their only
    consumer, so pointing it at the co-located checkout cannot put the two
    halves of a test run out of step.
    """
    roots: Dict[str, str] = {}

    sdk_origin = _resolve("jaato_sdk")
    if sdk_origin:
        roots["sdk"] = _package_root(sdk_origin)

    checkout = Path(roots["sdk"]).parent if "sdk" in roots else None
    if checkout is not None:
        sibling = checkout / _SERVER_SIBLING
        if all((sibling / pkg / "__init__.py").is_file()
               for pkg in _SERVER_PACKAGES):
            roots["server"] = str(sibling)

    if "server" not in roots:
        for name in _SERVER_PACKAGES:
            origin = _resolve(name)
            if origin:
                roots["server"] = _package_root(origin)
                break
    return roots


def tree_pythonpath() -> List[str]:
    """The directories of :func:`tree_roots`, in search order, deduplicated."""
    out: List[str] = []
    for root in tree_roots().values():
        if root not in out:
            out.append(root)
    return out


def expected_origins() -> Dict[str, str]:
    """What a child given :func:`tree_pythonpath` OUGHT to resolve.

    Stated rather than assumed, so the preflight below checks the child
    against the tree that was chosen for it -- not against whatever this
    process happens to resolve, which in the worktree case is the split
    :func:`tree_roots` exists to repair.  It therefore also catches a child
    whose search path is shadowed by something ahead of ours: its cwd, a
    stale ``.pth``, a conflicting install.
    """
    roots = tree_roots()
    expected: Dict[str, str] = {}
    if "sdk" in roots:
        expected["jaato_sdk"] = str(
            Path(roots["sdk"]) / "jaato_sdk" / "__init__.py")
    if "server" in roots:
        for pkg in _SERVER_PACKAGES:
            candidate = Path(roots["server"]) / pkg / "__init__.py"
            if candidate.is_file():
                expected[pkg] = str(candidate)
    return expected


def daemon_env(base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """*base* (default ``os.environ``) with this tree prepended to PYTHONPATH.

    PREPENDED, and an existing ``PYTHONPATH`` is kept after it: an operator
    who exported one meant it, and dropping it would trade this bug for the
    opposite one.  Ours wins, because the suite's whole claim is that the
    daemon runs the code the tests came from.
    """
    env = dict(os.environ if base is None else base)
    existing = env.get("PYTHONPATH", "")
    parts = tree_pythonpath()
    if existing:
        parts = parts + [existing]
    if parts:
        env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


class DaemonStartupError(RuntimeError):
    """The daemon did not become answerable.  Carries what was captured.

    The daemon's own output is attached because a startup failure with no
    log is the least actionable failure a CI job can produce -- the reader
    knows only that something did not happen.
    """


def _can_connect(socket_path: str) -> bool:
    """Does something ANSWER on this socket right now?

    Not ``os.path.exists``: a stale socket file is present and dead, and
    treating presence as readiness is how a suite ends up asserting against a
    daemon that is not running.
    """
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            sock.connect(socket_path)
        return True
    except (OSError, socket.timeout):
        return False


def echo_workspace(root: Path, *, usage: Optional[dict] = None,
                   tool_call: Optional[dict] = None,
                   response: Optional[str] = None,
                   completion_schema: Optional[dict] = None,
                   retry_tool_call: bool = False,
                   processor: Optional[str] = None,
                   processor_entry: Optional[dict] = None,
                   plugins: Optional[list] = None,
                   runtime_limits: Optional[dict] = None,
                   plugin_configs: Optional[dict] = None,
                   name: str = "conformance") -> Path:
    """Write a workspace with one echo-backed profile and return its path.

    ``usage`` is what makes budget invariants possible: echo reports the spend
    it is told to, identically every turn, so "how many turns to the ceiling"
    is arithmetic rather than observation.

    ``retry_tool_call`` + ``processor`` together are what let a profile model
    a REFUSED agent rather than a satisfied one.  Echo normally calls its tool
    once and then answers in prose, which is indistinguishable from the tool
    having succeeded; with the flag it re-claims completion every time, so a
    gate that refuses forms the loop jaato #768 is about and something else
    has to end it.  ``processor`` is the module body, written under
    ``.jaato/scripts/processors/<name>.py``, and ``processor_entry`` carries
    the ``completion_processors`` keys (``max_refusals``, ``on_exhausted``, …)
    the profile declares for it.

    ``completion_schema`` matters more than it looks.  A profile carrying one
    ends its run by calling ``signal_completion``, which terminates the
    session INSIDE a tool-use turn -- and that terminus is the condition under
    which a consumer measured event delivery silently stopping.  A suite whose
    profiles all end in prose never reaches it and reports everything healthy;
    that is not hypothetical, it is how the first repro of that defect
    exonerated the daemon.

    ``plugins`` / ``runtime_limits`` / ``plugin_configs`` are what let a
    profile drive a REAL tool rather than only the lifecycle surface.
    They exist for jaato #735, where the question is whether a declared
    ``runtime_limits.tool_timeout_seconds`` bounds an actual subprocess:
    that cannot be asked of a profile carrying ``plugins: []``, and it
    cannot be answered by any amount of wire assertion, because on the
    pre-fix tree the value travelled correctly and landed on an executor
    the session never used.  ``plugin_configs`` merges UNDER the echo
    section this function writes, so a caller can tune ``cli`` (notably
    ``auto_background_threshold`` — see that test) without having to
    reproduce the echo wiring.
    """
    profiles = root / ".jaato" / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)

    echo_cfg: dict = {}
    if usage is not None:
        echo_cfg["usage"] = usage
    if tool_call is not None:
        echo_cfg["tool_call"] = tool_call
    if response is not None:
        echo_cfg["response"] = response

    profile: dict = {
        "name": name,
        "description": "echo-backed profile for live conformance",
        "model": "echo",
        "provider": "echo",
        "plugins": list(plugins or []),
    }
    if runtime_limits is not None:
        profile["runtime_limits"] = dict(runtime_limits)
    if retry_tool_call:
        echo_cfg["retry_tool_call"] = True
    configs: dict = dict(plugin_configs or {})
    if echo_cfg:
        configs["echo"] = {**configs.get("echo", {}), **echo_cfg}
    if configs:
        profile["plugin_configs"] = configs
    if completion_schema is not None:
        profile["completion_payload_schema"] = completion_schema
    if processor is not None:
        script = root / ".jaato" / "scripts" / "processors" / f"{name}.py"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(processor, encoding="utf-8")
        entry = {"script": f"scripts/processors/{name}.py", "name": name}
        entry.update(processor_entry or {})
        profile["completion_processors"] = [entry]

    (profiles / f"{name}.json").write_text(
        json.dumps(profile, indent=2), encoding="utf-8")
    return root


class ConformanceDaemon:
    """A daemon owned by the test run, or one the operator supplied.

    ``JAATO_CONFORMANCE_SOCKET`` points the suite at an already-running daemon
    -- the consumer-facing mode, where the question is "does MY deployment
    conform?" rather than "did we regress?".  In that mode nothing is started
    and nothing is torn down, because a suite that kills an operator's daemon
    to tidy up is worse than one that never ran.
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self._external = os.environ.get("JAATO_CONFORMANCE_SOCKET")
        self.socket_path: str = self._external or ""
        self._proc: Optional[subprocess.Popen] = None
        self._tmpdir: Optional[str] = None
        self._log: Optional[Path] = None
        self._out: Optional[Path] = None
        self._out_handle = None

    # ------------------------------------------------------------- lifecycle

    def start(self) -> "ConformanceDaemon":
        if self._external:
            if not _can_connect(self._external):
                raise DaemonStartupError(
                    f"JAATO_CONFORMANCE_SOCKET={self._external} but nothing "
                    "answers there. The suite does not start a daemon in "
                    "external mode -- start yours, or unset the variable to "
                    "have the suite run its own."
                )
            return self

        self._tmpdir = tempfile.mkdtemp(prefix="jaato-conformance-")
        self.socket_path = os.path.join(self._tmpdir, "d.sock")
        self._log = Path(self._tmpdir) / "daemon.log"

        # NOT --daemon: the fixture owns this process and must be able to kill
        # it deterministically.  A forked daemon outlives a failed test run and
        # the next one inherits it.
        # --pid-file IS NOT OPTIONAL HERE.  Without it the daemon uses the
        # DEFAULT pidfile, sees any other jaato daemon on the machine, and
        # refuses to start -- "Jaato server is already running (PID ...)".
        # A unique socket is not enough: the running-instance check is keyed
        # on the pidfile, not the socket.  Found the first time this fixture
        # met a machine that already had a daemon on it, which in CI is a
        # previous job's leftover and in development is the operator's own.
        cmd = [sys.executable, "-m", "server",
               "--ipc-socket", self.socket_path,
               "--pid-file", os.path.join(self._tmpdir, "d.pid"),
               "--log-file", str(self._log)]
        # OUTPUT GOES TO A FILE, NOT A PIPE.  The daemon is chatty at startup
        # (plugin discovery, pool warm-up, extension load).  With
        # ``stdout=PIPE`` and nobody reading, it fills the 64KB pipe buffer
        # and BLOCKS -- so the fixture's readiness wait times out on a daemon
        # that is healthy and merely gagged, and reports it as a startup
        # failure.  A file has no buffer limit and is readable after the
        # process dies, which is exactly when the error path needs it.
        #
        # ``--log-file`` is NOT sufficient: the daemon's startup diagnostics
        # and its refusal messages go to stdout, and that file stayed empty
        # through the failure that motivated this.
        self._out = Path(self._tmpdir) / "daemon.out"
        self._out_handle = open(self._out, "wb")

        # THE DAEMON MUST IMPORT THE TREE THESE TESTS CAME FROM.  It is a
        # separate process, so it does not inherit pytest's rootdir
        # insertion: with a bare ``os.environ`` it resolved ``server`` /
        # ``shared`` / ``jaato_sdk`` through the editable install, which
        # points at one fixed checkout whatever tree pytest is running from.
        # In a ``git worktree`` that is a different tree, and the suite then
        # passes or fails about code it never ran (jaato #1050).  This
        # module's own docstring already names the class -- "the next run
        # inherits a process that is not the one it thinks it is testing" --
        # and had it only from the leaked-daemon direction.
        env = daemon_env()
        self._verify_child_tree(env)

        self._proc = subprocess.Popen(
            cmd,
            cwd=str(self.workspace),
            env=env,
            stdout=self._out_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,          # so teardown can kill the group
        )
        self._await_ready()
        return self

    def _verify_child_tree(self, env: Dict[str, str]) -> None:
        """Ask a child, with the daemon's own env and cwd, what it resolves.

        Run BEFORE the daemon, because the point is to refuse rather than to
        explain afterwards, and a 90-second startup spent on the wrong tree
        is time nobody gets back.

        Checked against :func:`expected_origins` -- the tree that was CHOSEN
        for the child -- rather than against whatever this process resolves,
        which in the worktree case is the very split :func:`tree_roots`
        exists to repair.

        ONLY POSITIVE EVIDENCE COUNTS, the rule jaato #1023 states for a
        thread label and which holds for the same reason here: a resolution
        read successfully that names a different file is proof the daemon
        would run other code, while a preflight that could not run proves
        nothing about the tree and must not fail a suite on a machine whose
        only fault is being unusual.  So divergence RAISES and an unreadable
        preflight WARNS -- into the daemon's own output file, which the
        startup error path already attaches.

        External mode never reaches here: ``start`` returns before this for
        ``JAATO_CONFORMANCE_SOCKET``, because there the operator has
        deliberately taken responsibility for what is running.
        """
        expected = expected_origins()
        theirs = self._preflight(env)
        if theirs is None:
            return

        divergent = {
            name: (expected[name], theirs[name])
            for name in sorted(set(expected) & set(theirs))
            if theirs[name] and expected[name] != theirs[name]
        }
        self._note(
            "tree check: " + ", ".join(
                f"{name}={theirs.get(name) or '<unresolved>'}"
                for name in TREE_MODULES
            )
        )
        if not divergent:
            return

        detail = "\n".join(
            f"  {name}\n    the tests' tree: {here}\n"
            f"    the daemon:     {there}"
            for name, (here, there) in divergent.items()
        )
        raise DaemonTreeMismatch(
            "the daemon would import a different checkout than the tests "
            "came from.\n"
            f"{detail}\n"
            "Nothing in a suite run distinguishes this from a real result, "
            "so the fixture refuses rather than report one (jaato #1050)."
        )

    def _preflight(self, env: Dict[str, str]) -> Optional[Dict[str, Optional[str]]]:
        """Where would the daemon resolve :data:`TREE_MODULES`?  ``None`` if unknown."""
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _PREFLIGHT_SRC % (list(TREE_MODULES),)],
                cwd=str(self.workspace),
                env=env,
                capture_output=True,
                text=True,
                timeout=PREFLIGHT_TIMEOUT,
            )
            resolved = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception as exc:                    # noqa: BLE001 - see docstring
            self._note(f"tree check SKIPPED: {type(exc).__name__}: {exc}")
            return None
        if not isinstance(resolved, dict):
            self._note(f"tree check SKIPPED: unparseable preflight output")
            return None
        return {
            name: (str(Path(origin).resolve()) if origin else None)
            for name, origin in resolved.items()
        }

    def _note(self, line: str) -> None:
        """Record a fixture-level fact in the daemon's own output file.

        The file, not a logger: a pytest run swallows logging by default, and
        ``DaemonStartupError`` already attaches this file's contents -- so
        anything written here reaches the reader on exactly the path that
        needs it.  Best-effort; a diagnostic that raises is worse than none.
        """
        try:
            if self._out_handle is not None:
                self._out_handle.write(
                    f"[conformance fixture] {line}\n".encode())
                self._out_handle.flush()
        except Exception:                           # noqa: BLE001
            pass

    def _await_ready(self) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise DaemonStartupError(
                    f"daemon exited with code {self._proc.returncode} before "
                    f"accepting a connection.\n--- daemon output ---\n"
                    f"{self._captured()}"
                )
            if _can_connect(self.socket_path):
                return
            time.sleep(POLL_INTERVAL)
        raise DaemonStartupError(
            f"daemon did not accept a connection on {self.socket_path} within "
            f"{STARTUP_TIMEOUT}s (process still alive: it started but never "
            f"answered).\n--- daemon output ---\n{self._captured()}"
        )

    def _captured(self) -> str:
        """Everything the daemon said, from BOTH channels.

        The refusal that motivated this went to the process's STDOUT and
        never reached the log file -- so the first version, which flushed
        the pipe without reading it, reported "(no daemon output captured)"
        while the answer sat unread in the pipe.  A startup failure with no
        log is the least actionable failure a CI job can produce, which this
        method's whole purpose is to prevent, and it was not delivering it.

        Reads non-blockingly: the daemon may still be alive (the timeout
        path), and a blocking read on a live process's pipe would hang the
        error path -- turning a legible failure into the hang it is
        diagnosing.
        """
        parts = []
        if self._out is not None and self._out.exists():
            try:
                self._out_handle.flush()
            except Exception:
                pass
            text = self._out.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                # Tail, not head: a startup failure announces itself at the
                # END of the output, after however much discovery chatter.
                parts.append("[stdout] " + "\n".join(text.splitlines()[-40:]))
        if self._log is not None and self._log.exists():
            text = self._log.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                parts.append("[log] " + text)
        return "\n".join(parts) or "(no daemon output captured)"

    def stop(self) -> None:
        """Unconditional, escalating teardown.

        A leaked daemon holds its socket and the next run inherits a process
        that is not the one it thinks it is testing.
        """
        if self._external:
            return
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), 15)     # TERM the group
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), 9)
                    proc.wait(timeout=10)
                except Exception:
                    pass
            except (ProcessLookupError, PermissionError):
                pass
        if self._out_handle is not None:
            try:
                self._out_handle.close()
            except Exception:
                pass
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    def __enter__(self) -> "ConformanceDaemon":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
