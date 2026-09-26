"""``workspace.clone``: clone GitHub repositories into a workspace (1.27).

The WS handler resolves the caller's workspace (the same visibility and
containment rules as ``select`` / ``delete``) and the workspace's own
``GH_TOKEN``, then hands both to :func:`clone_repos`, which clones the
requested repos ONE AT A TIME and reports each through an ``emit``
coroutine as a sequence of states:

``queued`` (every repo, up front) -> ``cloning`` (with ``percent`` from
git's ``--progress``) -> ``checkout`` -> ``done`` | ``failed``.

The rules this module holds to:

* **The token never reaches argv or a URL.**  It rides the environment as
  git's own ``GIT_CONFIG_COUNT`` / ``GIT_CONFIG_KEY_n`` / ``GIT_CONFIG_VALUE_n``
  mechanism -- an ``http.https://github.com/.extraheader`` of
  ``AUTHORIZATION: basic <b64(x-access-token:TOKEN)>`` -- so ``ps`` and
  ``/proc/<pid>/cmdline`` show nothing, and a remote URL written into the
  new checkout's ``.git/config`` is the bare ``https://github.com/o/r.git``.
* **The daemon user's own git credentials are never offered.**  The same
  mechanism resets ``credential.helper`` to empty, and prompting is off
  (``GIT_TERMINAL_PROMPT=0``, ``GIT_ASKPASS`` = ``true``), so a private repo
  with no workspace token FAILS with git's auth error rather than being
  cloned with whatever the daemon's account can read.
* **Error text is scrubbed** of the token (and its base64 form) before it
  is sent to a client.
* **A failed clone leaves nothing behind**: its partial target directory is
  removed.  A target that already exists is refused, never overwritten.
* **No network in tests**: the URL builder is injectable, so a test clones
  from a local bare repository.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

#: ``owner/name`` as the request carries it.
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
#: Characters git refuses in a ref name, plus whitespace.
_BAD_BRANCH_CHARS = re.compile(r"[\s~^:?*\[\\\x00-\x1f\x7f]")
#: ``Receiving objects:  56% (...)`` -- the phase that dominates a clone.
_RECEIVING = re.compile(r"Receiving objects:\s+(\d{1,3})%")
#: Lines that mark the working-tree phase.
_CHECKOUT_MARKERS = ("Updating files:", "Checking out files:")
#: Forges this verb can clone from today.
SUPPORTED_FORGES = frozenset({"github"})
#: At most this many ``cloning`` progress events per repo per second.
PROGRESS_EVENTS_PER_SECOND = 5
#: Hard cap on one clone, seconds.  A clone that has not finished by then
#: is killed and reported ``failed`` rather than holding the request open.
CLONE_TIMEOUT_SECONDS = 1800.0

Emit = Callable[[Dict[str, Any]], Awaitable[None]]
UrlFor = Callable[[str], str]


def github_url(repo: str) -> str:
    """The credential-free HTTPS clone URL for a GitHub ``owner/name``."""
    return f"https://github.com/{repo}.git"


@dataclass(frozen=True)
class CloneSpec:
    """One validated entry of a ``WorkspaceCloneRequest.repos`` list."""
    repo: str
    branch: str
    forge: str

    @property
    def dirname(self) -> str:
        """The target directory name inside the workspace: the repo's name."""
        return self.repo.split("/", 1)[1]


def spec_error(repo: str, branch: str, forge: str) -> str:
    """Why an entry cannot be cloned, or ``""`` when it can."""
    if forge not in SUPPORTED_FORGES:
        return f"forge not supported: {forge or '(none)'}"
    if not REPO_RE.match(repo):
        return f"invalid repo {repo!r}: expected owner/name"
    owner, name = repo.split("/", 1)
    if owner in (".", "..") or name.startswith("."):
        return f"invalid repo {repo!r}: '.'/'..' segments and dot-names are refused"
    if branch and (branch.startswith("-") or ".." in branch
                   or _BAD_BRANCH_CHARS.search(branch)):
        return f"invalid branch {branch!r}"
    return ""


def parse_specs(repos: List[Dict[str, Any]]) -> List[tuple]:
    """``[(CloneSpec, error), ...]`` for a request's ``repos`` list."""
    parsed = []
    for raw in repos:
        entry = raw if isinstance(raw, dict) else {}
        repo = str(entry.get("repo") or "").strip()
        branch = str(entry.get("branch") or "").strip()
        forge = str(entry.get("forge") or "github").strip().lower()
        parsed.append((CloneSpec(repo, branch, forge), spec_error(repo, branch, forge)))
    return parsed


def git_auth_env(token: Optional[str], host_url: str = "https://github.com/") -> Dict[str, str]:
    """The environment additions for one non-interactive, token-carrying clone.

    Always resets ``credential.helper`` (so the daemon account's own helper
    is never consulted) and disables prompting.  With a *token*, adds the
    ``extraheader`` that authenticates to *host_url*.
    """
    askpass = shutil.which("true") or "/bin/true"
    env = {
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": askpass,
        "SSH_ASKPASS": askpass,
        "GCM_INTERACTIVE": "never",
        "LC_ALL": "C",
        "GIT_CONFIG_KEY_0": "credential.helper",
        "GIT_CONFIG_VALUE_0": "",
        "GIT_CONFIG_COUNT": "1",
    }
    if token:
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        env.update({
            "GIT_CONFIG_KEY_1": f"http.{host_url}.extraheader",
            "GIT_CONFIG_VALUE_1": f"AUTHORIZATION: basic {basic}",
            "GIT_CONFIG_COUNT": "2",
        })
    return env


def scrub(text: str, token: Optional[str]) -> str:
    """*text* with the token -- raw or inside its basic-auth form -- masked."""
    if not token:
        return text
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return text.replace(basic, "***").replace(token, "***")


class _ProgressReader:
    """Consumes git's ``--progress`` stderr and throttles progress events.

    git redraws a progress line with ``\\r``, so the stream is split on both
    ``\\r`` and ``\\n``.  Non-progress lines are kept (last 20) as the error
    text for a failure.
    """

    def __init__(self, on_percent: Callable[[int], Awaitable[None]],
                 on_checkout: Callable[[], Awaitable[None]]) -> None:
        self._on_percent = on_percent
        self._on_checkout = on_checkout
        self._last_emit = 0.0
        self._last_percent = -1
        self.saw_checkout = False
        self.messages: List[str] = []

    async def feed_line(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        match = _RECEIVING.search(line)
        if match:
            await self._percent(min(100, int(match.group(1))))
            return
        if any(marker in line for marker in _CHECKOUT_MARKERS):
            if not self.saw_checkout:
                self.saw_checkout = True
                await self._on_checkout()
            return
        if "%" not in line:
            self.messages = (self.messages + [line])[-20:]

    async def _percent(self, percent: int) -> None:
        now = time.monotonic()
        due = now - self._last_emit >= 1.0 / PROGRESS_EVENTS_PER_SECOND
        if percent != self._last_percent and (due or percent == 100):
            self._last_percent = percent
            self._last_emit = now
            await self._on_percent(percent)


async def _drain_stderr(stream: asyncio.StreamReader, reader: _ProgressReader) -> None:
    """Feed *stream* to *reader* split on ``\\r`` / ``\\n`` until EOF."""
    pending = ""
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            break
        pending += chunk.decode("utf-8", errors="replace")
        parts = re.split(r"[\r\n]", pending)
        pending = parts.pop()
        for part in parts:
            await reader.feed_line(part)
    await reader.feed_line(pending)


async def _run_git_clone(
    argv: List[str], env: Dict[str, str], reader: _ProgressReader,
) -> int:
    """Run the clone, streaming stderr through *reader*; returns the exit code."""
    proc = await asyncio.create_subprocess_exec(
        *argv, env=env,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        await asyncio.wait_for(_drain_stderr(proc.stderr, reader),
                               timeout=CLONE_TIMEOUT_SECONDS)
        return await proc.wait()
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        reader.messages.append(f"clone timed out after {CLONE_TIMEOUT_SECONDS:g}s")
        return -1


async def clone_one(
    spec: CloneSpec,
    workspace: Path,
    token: Optional[str],
    emit: Emit,
    url_for: UrlFor = github_url,
) -> str:
    """Clone one repo into ``<workspace>/<name>``; ``""`` on success, else the error.

    Emits ``cloning`` (with percent) and ``checkout`` states through *emit*
    (only the state-specific fields; the caller adds counters).  The
    terminal ``done`` / ``failed`` event is the caller's.
    """
    target = workspace / spec.dirname
    if target.exists() or target.is_symlink():
        return f"{spec.dirname} already exists in the workspace"

    async def on_percent(percent: int) -> None:
        await emit({"state": "cloning", "percent": percent})

    async def on_checkout() -> None:
        await emit({"state": "checkout", "percent": 100})

    reader = _ProgressReader(on_percent, on_checkout)
    await emit({"state": "cloning", "percent": 0})
    argv = ["git", "clone", "--progress"]
    if spec.branch:
        argv += ["--branch", spec.branch]
    argv += ["--", url_for(spec.repo), str(target)]
    env = {**os.environ, **git_auth_env(token)}
    try:
        code = await _run_git_clone(argv, env, reader)
    except OSError as exc:
        code, reader.messages = -1, [f"could not run git: {exc}"]
    if code == 0:
        if not reader.saw_checkout:
            await on_checkout()
        return ""
    shutil.rmtree(target, ignore_errors=True)
    return scrub("\n".join(reader.messages) or f"git clone exited {code}", token)


async def clone_repos(
    workspace: Path,
    repos: List[Dict[str, Any]],
    token: Optional[str],
    emit: Emit,
    url_for: UrlFor = github_url,
) -> None:
    """Clone every entry of *repos* into *workspace*, sequentially.

    *emit* receives dicts with ``repo``, ``branch``, ``state``, ``percent``,
    ``error``, ``done`` and ``total`` -- the fields of a
    ``WorkspaceCloneProgressEvent`` minus the name and request id.  Every
    repo is announced ``queued`` first; an invalid entry is then reported
    ``failed`` in its turn without running git.
    """
    parsed = parse_specs(repos)
    total = len(parsed)
    done = 0
    for spec, _ in parsed:
        await emit({"repo": spec.repo, "branch": spec.branch, "state": "queued",
                    "percent": 0, "error": "", "done": 0, "total": total})
    for spec, error in parsed:

        async def step(fields: Dict[str, Any], _spec: CloneSpec = spec) -> None:
            await emit({"repo": _spec.repo, "branch": _spec.branch, "error": "",
                        "done": done, "total": total, **fields})

        if not error:
            error = await clone_one(spec, workspace, token, step, url_for)
        done += 1
        if error:
            logger.info("workspace.clone: %s failed in %s: %s", spec.repo, workspace, error)
            await step({"state": "failed", "percent": 0, "error": error})
        else:
            logger.info("workspace.clone: cloned %s into %s", spec.repo, workspace)
            await step({"state": "done", "percent": 100})


def resolve_workspace_token(
    workspace: Path,
    resolver: Any = None,
) -> Optional[str]:
    """The workspace's ``GH_TOKEN``, resolved as a session there would be.

    Reads ``<workspace>/.env`` (``GH_TOKEN``, then ``GITHUB_TOKEN``).  An
    ``app://<name>`` reference is resolved through *resolver* (the WS
    server's :class:`~server.app_secret.AppSecretResolver`) for the
    workspace's OWNER, exactly as ``JaatoServer._apply_app_secret_references``
    does at spawn; any other secret URI (``pass://`` / ``vault://``) through
    the registered resolvers.  Blocking (an ``app://`` round trip): call it
    off the event loop.  ``None`` when nothing resolves -- the clone then runs
    anonymously, which is enough for a public repository.
    """
    env_file = workspace / ".env"
    if not env_file.is_file():
        return None
    from dotenv import dotenv_values
    values = dotenv_values(env_file)
    raw = values.get("GH_TOKEN") or values.get("GITHUB_TOKEN")
    if not raw:
        return None
    return _resolve_token_value(raw, str(workspace), resolver)


def _resolve_token_value(raw: str, workspace_path: str, resolver: Any) -> Optional[str]:
    """Turn one ``.env`` value into a token: literal, ``app://``, or secret URI."""
    from jaato_server.shared.plugins.subagent.config import (
        SecretResolveContext,
        _resolve_secret_uri,
        parse_app_secret_reference,
    )
    ref = parse_app_secret_reference(raw)
    if ref is None:
        try:
            return _resolve_secret_uri(raw) or None
        except Exception as exc:  # noqa: BLE001 -- an unresolvable URI is "no token"
            logger.warning("workspace.clone: GH_TOKEN did not resolve: %s", exc)
            return None
    if resolver is None:
        logger.warning("workspace.clone: GH_TOKEN is %s and no app:// resolver "
                       "is wired; cloning anonymously", raw)
        return None
    context = SecretResolveContext(
        workspace_path=workspace_path,
        workspace_owner=resolver.owner_for(workspace_path),
        session_id="",
    )
    answer = resolver.resolve_reference(ref, context)
    if answer.ok:
        return answer.value
    logger.warning("workspace.clone: %s did not resolve (%s); cloning anonymously",
                   raw, answer.detail or answer.status)
    return None
