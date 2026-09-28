"""A missing command becomes a hint for the model and a notice for the page.

The web coder's backend writes ``<workspace>/.jaato/toolchain-offer.json``:
every toolchain its operator allows, the versions offered, the commands that
suggest each one, and the version bound in this workspace (or ``null``).
This plugin reads it in the runner, and when a ``cli``, ``interactive_shell``
or ``notebook`` result shows that a command in it was not found:

- appends one line to the result for the MODEL, saying which toolchain
  provides the command and that the user binds it from the web coder, or,
  when it is already bound, that something else is wrong;
- returns a ``client_notice`` (protocol 1.31) the daemon emits as
  ``ToolResultEnrichedEvent`` ``kind="toolchain_offer"``, which the page
  turns into its Bind chip.  The page does no detection of its own.

The file sits in the workspace, where the model can write, so every field
is validated and a value that fails is dropped, never quoted: the wording
is this module's, the file only chooses among fixed sentences.  A workspace
with no file (one the web coder never touched) gets nothing.

Each command is hinted once per session.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from jaato_sdk.plugins.base import ToolResultEnrichmentResult

logger = logging.getLogger(__name__)

OFFER_PATH = os.path.join(".jaato", "toolchain-offer.json")
SCHEMA = 1
MAX_OFFER_BYTES = 64 * 1024

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
    # bash: "bash: line 1: javac: command not found", "bash: javac: command not found".
    # Not anchored to a line start: in the session's text view a result's
    # stderr arrives as "stderr: bash: line 1: ...".
    re.compile(r"(?:^|(?<=[\s:]))" + _NAME + r": command not found"),
    # dash / sh, which a notebook's "!" uses: "/bin/sh: 1: javac: not found".
    re.compile(r"\b(?:sh|dash|bash):\s*\d+:\s*" + _NAME + r": not found"),
    # "/usr/bin/env: 'node': No such file or directory"
    re.compile(r"/usr/bin/env:\s*['‘]?" + _NAME + r"['’]?: No such file or directory"),
    # A notebook cell's subprocess.run(["javac", ...]): the bare name, never a path.
    re.compile(r"FileNotFoundError: \[Errno 2\] No such file or directory: '" + _NAME + r"'"),
]

_TOOL_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .+()-]{0,47}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")
_COMMAND_RE = re.compile(r"^[A-Za-z0-9._+-]{1,64}$")


def missing_commands(text: str) -> List[str]:
    """The command names ``text`` says were not found, in order, without repeats."""
    seen: List[str] = []
    for pattern in NOT_FOUND_PATTERNS:
        for m in pattern.finditer(text):
            name = m.group(1)
            if name not in seen:
                seen.append(name)
    return seen


def parse_offer(raw: Any) -> Optional[Dict[str, Dict[str, Any]]]:
    """The offer as ``{command: toolchain}``, or ``None`` when it is not one this plugin reads.

    Every field is checked; a toolchain entry with a bad field is dropped
    whole, a bad version or command inside a good entry is dropped alone.
    """
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA or not isinstance(raw.get("toolchains"), list):
        return None
    by_command: Dict[str, Dict[str, Any]] = {}
    for entry in raw["toolchains"][:32]:
        if not isinstance(entry, dict):
            continue
        tool, label, bound = entry.get("tool"), entry.get("label"), entry.get("bound")
        if not (isinstance(tool, str) and _TOOL_RE.match(tool) and isinstance(label, str) and _LABEL_RE.match(label)):
            continue
        if bound is not None and not (isinstance(bound, str) and _VERSION_RE.match(bound)):
            continue
        versions = [v for v in (entry.get("versions") or [])[:16] if isinstance(v, str) and _VERSION_RE.match(v)]
        commands = [c for c in (entry.get("commands") or [])[:32] if isinstance(c, str) and _COMMAND_RE.match(c)]
        if not versions:
            continue
        toolchain = {"tool": tool, "label": label, "versions": versions, "bound": bound}
        for command in commands:
            by_command.setdefault(command, toolchain)
    return by_command


def hint_text(command: str, toolchain: Dict[str, Any]) -> str:
    """The line the model reads.  Built only from validated fields."""
    label = toolchain["label"]
    if toolchain["bound"]:
        version = "" if toolchain["bound"] == "system" else f" {toolchain['bound']}"
        return (
            f"[toolchain] `{command}` was not found although {label}{version} is bound to this workspace, "
            "with its binaries linked into ~/.local/bin. Call get_environment(aspect=\"runtime\") to see what "
            "can run, and tell the user if it is still missing. Do not install it another way."
        )
    offered = ", ".join(toolchain["versions"])
    return (
        f"[toolchain] `{command}` is provided by {label}, which is not bound to this workspace. "
        f"The user can bind it ({offered}) from the Toolchains section of the web coder; ask them to, "
        "then run the command again. Do not install it another way."
    )


class ToolchainOfferPlugin:
    """Enrichment-only, runner-tier.  State: the workspace, the cached offer, what was hinted."""

    name = "toolchain_offer"

    def __init__(self) -> None:
        self._workspace: Optional[str] = None
        self._session_id: Optional[str] = None
        self._cache_key: Optional[Tuple[int, int]] = None
        self._offer: Optional[Dict[str, Dict[str, Any]]] = None
        self._hinted: Set[Tuple[Optional[str], str]] = set()

    # -- lifecycle ---------------------------------------------------------

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        workspace = (config or {}).get("workspace_path")
        if isinstance(workspace, str) and workspace:
            self.set_workspace_path(workspace)
        session_id = (config or {}).get("session_id")
        if isinstance(session_id, str):
            self._session_id = session_id

    def shutdown(self) -> None:
        self._hinted.clear()

    def reset_for_next_session(self) -> None:
        self._hinted.clear()

    def set_workspace_path(self, path: str) -> None:
        if path != self._workspace:
            self._workspace = path
            self._cache_key = None
            self._offer = None

    def set_session_id(self, session_id: Optional[str]) -> None:
        self._session_id = session_id

    # -- enrichment --------------------------------------------------------

    def subscribes_to_tool_result_enrichment(self) -> bool:
        return True

    def enrich_tool_result(self, tool_name: str, result: str, tool_args: Optional[Dict[str, Any]] = None) -> ToolResultEnrichmentResult:
        if tool_name not in SCOPED_TOOLS or not isinstance(result, str):
            return ToolResultEnrichmentResult(result=result)
        names = missing_commands(result)
        if not names:
            return ToolResultEnrichmentResult(result=result)
        offer = self._read_offer()
        if not offer:
            return ToolResultEnrichmentResult(result=result)
        for command in names:
            toolchain = offer.get(command)
            key = (self._session_id, command)
            if toolchain is None or key in self._hinted:
                continue
            self._hinted.add(key)
            return ToolResultEnrichmentResult(
                result=f"{result}\n\n{hint_text(command, toolchain)}",
                metadata={
                    "client_notice": {"kind": "toolchain_offer", "data": {"command": command, **toolchain}},
                    "notification": {"message": f"{command} not found: {toolchain['label']} {'is bound' if toolchain['bound'] else 'can be bound'}"},
                },
            )
        return ToolResultEnrichmentResult(result=result)

    def _read_offer(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """The offer, re-read only when the file's size or modification time changed."""
        if not self._workspace:
            return None
        path = os.path.join(self._workspace, OFFER_PATH)
        try:
            st = os.stat(path)
        except OSError:
            self._cache_key, self._offer = None, None
            return None
        key = (st.st_mtime_ns, st.st_size)
        if key == self._cache_key:
            return self._offer
        self._cache_key, self._offer = key, None
        if st.st_size > MAX_OFFER_BYTES:
            logger.warning("toolchain_offer: %s is %d bytes (max %d); ignored", path, st.st_size, MAX_OFFER_BYTES)
            return None
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, ValueError) as e:
            logger.warning("toolchain_offer: cannot read %s: %s", path, e)
            return None
        self._offer = parse_offer(raw)
        if self._offer is None:
            logger.warning("toolchain_offer: %s is not a schema-%d offer; no hints", path, SCHEMA)
        return self._offer
