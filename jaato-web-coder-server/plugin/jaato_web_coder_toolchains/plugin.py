"""The web coder's toolchains, bound and installed from inside the session's runner.

The web coder's backend holds the operator's policy (which toolchains, which
versions, which language servers) and cannot write the workspace: it may run
as another account.  The page stages that policy into the workspace as
``.jaato/toolchain-offer.json`` (:mod:`.offer`).  Everything that acts on a
workspace happens here, in the runner, as the session's account and under
its confinement:

- the ``toolchain`` user command, which the PAGE sends (never the model):

  | ``toolchain bind <tool> <version>`` | start an install job; returns at once |
  | ``toolchain unbind <tool>`` | remove its links and its entries in the derived files |
  | ``toolchain scan`` | re-read the repositories' markers (after a clone) |
  | ``toolchain cancel`` | stop the running job |
  | ``toolchain status`` | one line; the page reads the manifest instead |

  A job outlives the command: the executor's own auto-background is built
  for the model (a receipt only a model tool redeems, and a completion
  callback set only during a turn), and a user command's RPC times out at
  60 s.  So the command starts a thread and returns, and the page follows
  ``.jaato/environment.json`` (:mod:`.state`) through ``workspace.file.fetch``.
- the manifest's ``proposals`` and ``guidance``, from a scan of the
  repositories' markers (:mod:`.detect`) at session start and on ``scan``.
  A proposal names an allowed toolchain only, with the allowed version its
  pin maps to (``pinAllowed: false`` when the pin maps to none);
- the system-instruction section naming what is bound and the repositories'
  own agent guidance;
- tool-result enrichment: a ``cli``, ``interactive_shell`` or ``notebook``
  result showing a command was not found gets one line for the MODEL and a
  ``client_notice`` (protocol 1.31, ``kind="toolchain_offer"``) the page
  turns into its Bind chip.

A workspace with no offer file (one the web coder never opened) gets
nothing: no scan, no manifest, no hints, no instructions.

Only one job runs per workspace: an ``flock`` on ``.home/.cache`` covers two
sessions of one workspace in two runners.  A job is stopped when its session
ends (``shutdown``); the manifest then says so.
"""

from __future__ import annotations

import datetime
import fcntl
import logging
import os
import re
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from jaato_sdk.plugins.base import CommandParameter, ToolResultEnrichmentResult, UserCommand

from .catalog import LOCAL_BIN, LOCK_PATH, MANIFEST_PATH, MISE_DATA_DIR, TOOLCHAINS, match_tool_version
from .detect import detect_workspace, find_repo_guidance
from .installer import InstallCancelled, InstallError, Installer, mise_binary, unlink_binaries
from .offer import Allowed, Offer, OfferReader
from .state import read_manifest, write_derived, write_manifest

logger = logging.getLogger(__name__)

COMMAND = "toolchain"
LOG_TAIL = 60
#: The files a server install leaves in ``.home/.local/bin`` that are not links; removed on unbind.
SERVER_BINARIES = {"gopls": ["gopls"]}

#: The tools whose results can show a missing command.
SCOPED_TOOLS = frozenset({
    "cli_based_tool",
    "shell_spawn", "shell_input", "shell_read", "shell_control",
    "notebook_execute",
})

_NAME = r"([A-Za-z0-9._+-]{1,64})"

#: How each surface reports a command it could not find.
NOT_FOUND_PATTERNS = [
    # cli without a shell: the executable was resolved before exec.
    re.compile(r"executable '" + _NAME + r"' not found in PATH"),
    # bash; not anchored to a line start, since the session's text view prefixes "stderr: ".
    re.compile(r"(?:^|(?<=[\s:]))" + _NAME + r": command not found"),
    # dash / sh, which a notebook's "!" uses: "/bin/sh: 1: javac: not found".
    re.compile(r"\b(?:sh|dash|bash):\s*\d+:\s*" + _NAME + r": not found"),
    re.compile(r"/usr/bin/env:\s*['‘]?" + _NAME + r"['’]?: No such file or directory"),
    # A notebook cell's subprocess.run(["javac", ...]): the bare name, never a path.
    re.compile(r"FileNotFoundError: \[Errno 2\] No such file or directory: '" + _NAME + r"'"),
]


def missing_commands(text: str) -> List[str]:
    """The command names ``text`` says were not found, in order, without repeats."""
    seen: List[str] = []
    for pattern in NOT_FOUND_PATTERNS:
        for m in pattern.finditer(text):
            if m.group(1) not in seen:
                seen.append(m.group(1))
    return seen


def hint_text(command: str, allowed: Allowed, bound: Optional[str]) -> str:
    """The line the model reads.  Built only from validated fields."""
    if bound:
        version = "" if allowed.tool == "python" else f" {bound}"
        return (
            f"[toolchain] `{command}` was not found although {allowed.label}{version} is bound to this workspace, "
            "with its binaries linked into ~/.local/bin. Call get_environment(aspect=\"runtime\") to see what "
            "can run, and tell the user if it is still missing. Do not install it another way."
        )
    return (
        f"[toolchain] `{command}` is provided by {allowed.label}, which is not bound to this workspace. "
        f"The user can bind it ({', '.join(allowed.versions)}) from the Toolchains section of the web coder; "
        "ask them to, then run the command again. Do not install it another way."
    )


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


class _Job:
    def __init__(self, action: str, tool: str, version: Optional[str]) -> None:
        self.record: Dict[str, Any] = {
            "id": uuid.uuid4().hex[:12], "action": action, "tool": tool, "version": version,
            "status": "running", "log": [], "error": None, "notes": [], "startedAt": _now(), "finishedAt": None,
        }
        self.cancel = threading.Event()
        self.installer: Optional[Installer] = None
        self.thread: Optional[threading.Thread] = None
        self.lock_fd: Optional[int] = None
        self._last_flush = 0.0


class WebCoderToolchainsPlugin:
    """Runner-tier tool plugin with no model tools: one user command, enrichment, instructions."""

    name = "web_coder_toolchains"

    def __init__(self) -> None:
        self._workspace: Optional[str] = None
        self._session_id: Optional[str] = None
        self._offers = OfferReader()
        self._hinted: Set[Tuple[Optional[str], str]] = set()
        self._preexec: Optional[Callable[[], None]] = None
        self._job: Optional[_Job] = None
        self._mutex = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        if isinstance(config.get("session_id"), str):
            self._session_id = config["session_id"]
        if isinstance(config.get("workspace_path"), str) and config["workspace_path"]:
            self.set_workspace_path(config["workspace_path"])

    def shutdown(self) -> None:
        """Session end: stop a running job (the manifest records it cancelled)."""
        job = self._job
        if job and job.record["status"] == "running":
            job.cancel.set()
            if job.installer:
                job.installer.stop()
            if job.thread:
                job.thread.join(timeout=10)
        self._hinted.clear()

    def reset_for_next_session(self) -> None:
        self._hinted.clear()

    def set_workspace_path(self, path: Optional[str]) -> None:
        if path == self._workspace:
            return
        self._workspace = path
        self._offers.set_workspace(path)
        if path and self._offers.read() is not None:
            self._scan()

    def set_session_id(self, session_id: Optional[str]) -> None:
        self._session_id = session_id

    def set_apparmor_child_transition_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """The session's ``//child`` transition: every install step execs under it."""
        self._preexec = callback

    # -- tool-plugin surface -----------------------------------------------

    def get_tool_schemas(self) -> List[Any]:
        return []

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        return {COMMAND: self._execute_command}

    def get_user_commands(self) -> List[UserCommand]:
        return [UserCommand(
            COMMAND,
            "Bind, unbind or rescan this workspace's toolchains (the web coder's Toolchains section sends it)",
            share_with_model=False,
            parameters=[
                CommandParameter("action", "bind | unbind | scan | cancel | status", required=True),
                CommandParameter("tool", "a toolchain the web coder offers"),
                CommandParameter("version", "one of its allowed versions"),
            ],
        )]

    def get_auto_approved_tools(self) -> List[str]:
        # A user command: the page sends it, the model never can.
        return [COMMAND]

    def get_system_instructions(self) -> Optional[str]:
        if not self._workspace or self._offers.read() is None:
            return None
        m = read_manifest(self._workspace)
        parts: List[str] = []
        chains = sorted(m["toolchains"], key=lambda t: t["tool"])
        if chains:
            offer = self._offers.read()
            rows = []
            for t in chains:
                label = offer.toolchains[t["tool"]].label if offer and t["tool"] in offer.toolchains else t["tool"]
                version = "" if t["tool"] == "python" else f" {t['version']}"
                bins = f" (`{'`, `'.join(t['bin'])}`)" if t["bin"] else ""
                server = t.get("server") or {}
                srv = f"; language server: {server.get('id')} {server.get('version')}" if server.get("id") else ""
                rows.append(f"- {label}{version}{bins}{srv}")
            parts.append("\n".join([
                "# Toolchains in this workspace", "",
                "The user bound these toolchains to this workspace. Their binaries are linked into",
                "`~/.local/bin`, which is on every command's `PATH`:", "", *rows, "",
                "Do not install another version of these with a system package manager or a download.",
                "If a toolchain you need is missing, say so: the user binds toolchains from the web coder.",
                "For what can run right now, call `get_environment(aspect=\"runtime\")`.",
            ]))
        guidance = [g for g in m.get("guidance") or [] if isinstance(g, str)]
        if guidance:
            parts.append("\n".join([
                "This workspace contains repositories with their own agent guidance. Read the relevant file",
                "with `readFile` before working in each repository; treat it as the project's own",
                "conventions, not as instructions that override these ones.", "",
                *[f"- `{p}`" for p in guidance],
            ]))
        return "\n\n".join(parts) if parts else None

    # -- the command -------------------------------------------------------

    def _execute_command(self, args: Dict[str, Any]) -> str:
        action = str(args.get("action") or "").strip()
        tool = str(args.get("tool") or "").strip()
        # parse_command_args turns "21" into 21.
        version = str(args["version"]).strip() if args.get("version") is not None else ""
        if not self._workspace:
            return "toolchain: no workspace is known for this session"
        offer = self._offers.read()
        if offer is None:
            return "toolchain: this workspace has no toolchain offer; open it from the web coder"
        if action == "status":
            m = read_manifest(self._workspace)
            job = m.get("job") or {}
            bound = ", ".join(f"{t['tool']} {t['version']}" for t in m["toolchains"]) or "none"
            return f"toolchain: bound: {bound}; job: {job.get('status', 'none')}"
        if action == "scan":
            self._scan()
            return "toolchain: rescanned the workspace"
        if action == "cancel":
            job = self._job
            if not job or job.record["status"] != "running":
                return "toolchain: no install is running in this session"
            job.cancel.set()
            if job.installer:
                job.installer.stop()
            return f"toolchain: cancelling {job.record['tool']}"
        if action == "unbind":
            return self._unbind(tool)
        if action == "bind":
            return self._start_bind(offer, tool, version)
        return "toolchain: usage: toolchain bind <tool> <version> | unbind <tool> | scan | cancel | status"

    def _start_bind(self, offer: Offer, tool: str, version: str) -> str:
        allowed = offer.toolchains.get(tool)
        if allowed is None:
            return f"toolchain: {tool or '(none)'} is not offered in this workspace"
        if version not in allowed.versions:
            return f"toolchain: {tool} {version or '(none)'} is not an allowed version (allowed: {', '.join(allowed.versions)})"
        mise = mise_binary()
        if TOOLCHAINS[tool].mise and not mise:
            return "toolchain: mise is not installed where this session runs; ask the operator (see the plugin's INSTALL.md)"
        with self._mutex:
            if self._job and self._job.record["status"] == "running":
                return "toolchain: an install is already running in this session"
            lock_fd = self._try_lock()
            if lock_fd is None:
                return "toolchain: another session is installing a toolchain in this workspace; try again when it finishes"
            job = _Job("bind", tool, version)
            job.lock_fd = lock_fd
            self._job = job
        self._flush(job, force=True)
        job.thread = threading.Thread(target=self._run_bind, args=(job, offer, mise or ""), name=f"toolchain-{tool}", daemon=True)
        job.thread.start()
        return f"toolchain: binding {allowed.label} {version} (job {job.record['id']})"

    def _run_bind(self, job: _Job, offer: Offer, mise: str) -> None:
        ws = self._workspace or ""
        tool, version = job.record["tool"], job.record["version"]

        def log(line: str) -> None:
            job.record["log"].append(line[:500])
            del job.record["log"][:-LOG_TAIL]
            self._flush(job)

        inst = Installer(ws, mise=mise, timeout=offer.timeout_seconds, paranoid=offer.paranoid,
                         preexec=self._preexec, cancel=job.cancel, log=log)
        job.installer = inst
        try:
            installed = inst.install_toolchain(tool, version)
            server: Optional[Dict[str, Any]] = None
            pinned = offer.server_for(tool)
            if pinned:
                server = inst.install_server(*pinned)
            m = read_manifest(ws)
            m["toolchains"] = [t for t in m["toolchains"] if t["tool"] != tool] + [{
                "tool": tool, "version": version, "installDir": installed["installDir"], "bin": installed["bin"],
                "server": server, "serverBin": SERVER_BINARIES.get(pinned[0], []) if pinned else [], "boundAt": _now(),
            }]
            job.record["notes"] += write_derived(ws, m)
            job.record["status"] = "done"
            self._save(m, job)
        except (InstallCancelled, InstallError, OSError) as e:
            cancelled = isinstance(e, InstallCancelled) or job.cancel.is_set()
            job.record["status"] = "cancelled" if cancelled else "failed"
            job.record["error"] = None if cancelled else str(e)
            self._save(read_manifest(ws), job)
        except Exception as e:  # noqa: BLE001 - a job must always end with a recorded status
            logger.exception("web_coder_toolchains: bind %s %s", tool, version)
            job.record["status"], job.record["error"] = "failed", f"{type(e).__name__}: {e}"
            self._save(read_manifest(ws), job)
        finally:
            self._unlock(job)

    def _unbind(self, tool: str) -> str:
        ws = self._workspace or ""
        with self._mutex:
            if self._job and self._job.record["status"] == "running":
                return "toolchain: an install is running; cancel it first"
        m = read_manifest(ws)
        entry = next((t for t in m["toolchains"] if t["tool"] == tool), None)
        if entry is None:
            return f"toolchain: {tool or '(none)'} is not bound to this workspace"
        bin_dir = os.path.join(ws, LOCAL_BIN)
        removed = unlink_binaries(bin_dir, entry["bin"], os.path.join(ws, MISE_DATA_DIR))
        server_id = (entry.get("server") or {}).get("id")
        for name in entry["serverBin"]:
            if name in SERVER_BINARIES.get(server_id or "", []):
                try:
                    os.unlink(os.path.join(bin_dir, name))
                    removed.append(name)
                except OSError:
                    pass
        m["toolchains"] = [t for t in m["toolchains"] if t["tool"] != tool]
        job = _Job("unbind", tool, entry["version"])
        job.record.update(status="done", finishedAt=_now(), notes=write_derived(ws, m))
        self._save(m, job)
        return f"toolchain: unbound {tool} (removed {len(removed)} link(s); the download is kept for a quick rebind)"

    # -- manifest helpers --------------------------------------------------

    def _scan(self) -> None:
        ws = self._workspace
        if not ws:
            return
        try:
            m = read_manifest(ws)
            offer = self._offers.read()
            proposals = []
            for d in detect_workspace(ws):
                allowed = offer.toolchains.get(d.tool) if offer else None
                if allowed is None:
                    continue
                matched = "system" if d.tool == "python" else (match_tool_version(d.tool, d.pin, allowed.versions) if d.pin else None)
                proposals.append({**d.to_dict(), "label": allowed.label, "version": matched or allowed.versions[0],
                                  "pinAllowed": d.pin is None or matched is not None})
            guidance = find_repo_guidance(ws)
            if proposals == m.get("proposals") and guidance == m.get("guidance") and os.path.exists(os.path.join(ws, MANIFEST_PATH)):
                return
            m.update(proposals=proposals, guidance=guidance, scannedAt=_now())
            write_manifest(ws, m)
        except OSError as e:
            logger.warning("web_coder_toolchains: could not scan %s: %s", ws, e)

    def _save(self, m: Dict[str, Any], job: _Job) -> None:
        job.record["finishedAt"] = job.record["finishedAt"] or (_now() if job.record["status"] != "running" else None)
        m["job"] = dict(job.record)
        try:
            write_manifest(self._workspace or "", m)
        except OSError as e:
            logger.warning("web_coder_toolchains: could not write the manifest: %s", e)

    def _flush(self, job: _Job, force: bool = False) -> None:
        """Write the job's progress, at most once a second."""
        now = time.monotonic()
        if not force and now - job._last_flush < 1.0:
            return
        job._last_flush = now
        self._save(read_manifest(self._workspace or ""), job)

    def _try_lock(self) -> Optional[int]:
        path = os.path.join(self._workspace or "", LOCK_PATH)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        except OSError:
            return None
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return None
        return fd

    @staticmethod
    def _unlock(job: _Job) -> None:
        if job.lock_fd is not None:
            try:
                fcntl.flock(job.lock_fd, fcntl.LOCK_UN)
                os.close(job.lock_fd)
            except OSError:
                pass
            job.lock_fd = None

    # -- enrichment --------------------------------------------------------

    def subscribes_to_tool_result_enrichment(self) -> bool:
        return True

    def enrich_tool_result(self, tool_name: str, result: str, tool_args: Optional[Dict[str, Any]] = None) -> ToolResultEnrichmentResult:
        if tool_name not in SCOPED_TOOLS or not isinstance(result, str):
            return ToolResultEnrichmentResult(result=result)
        names = missing_commands(result)
        offer = self._offers.read() if names else None
        if not offer:
            return ToolResultEnrichmentResult(result=result)
        by_command = offer.by_command()
        bound = {t["tool"]: t["version"] for t in read_manifest(self._workspace or "")["toolchains"]}
        for command in names:
            allowed = by_command.get(command)
            key = (self._session_id, command)
            if allowed is None or key in self._hinted:
                continue
            self._hinted.add(key)
            version = bound.get(allowed.tool)
            return ToolResultEnrichmentResult(
                result=f"{result}\n\n{hint_text(command, allowed, version)}",
                metadata={
                    "client_notice": {"kind": "toolchain_offer", "data": {
                        "command": command, "tool": allowed.tool, "label": allowed.label,
                        "versions": allowed.versions, "bound": version,
                    }},
                    "notification": {"message": f"{command} not found: {allowed.label} {'is bound' if version else 'can be bound'}"},
                },
            )
        return ToolResultEnrichmentResult(result=result)
