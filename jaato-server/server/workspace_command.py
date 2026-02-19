"""Workspace command handler for the ``workspace`` server command.

Provides subcommands that format WorkspaceMonitor snapshot data in various
output formats, primarily for use with ``--cmd "workspace tree"`` from
command mode but also usable from the TUI.

Subcommands
-----------
- ``tree``  – Indented tree view grouped by directory.
- ``list``  – Flat file list, one path per line with status indicator.
- ``json``  – Machine-readable JSON output.
- ``csv``   – CSV output (path, status).

The handler is invoked directly by ``SessionManager`` (not via the plugin
system) because it needs access to ``_workspace_monitors[session_id]``
which lives on the session manager, not on any plugin.
"""

import csv
import io
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .workspace_monitor import WorkspaceMonitor

# Status symbols for human-readable formats
_STATUS_SYMBOLS = {
    "created": "+",
    "modified": "~",
    "deleted": "-",
}

_VALID_SUBCOMMANDS = ("tree", "list", "json", "csv")


def handle_workspace_command(
    monitor: Optional[WorkspaceMonitor],
    args: List[str],
) -> Dict[str, Any]:
    """Handle a ``workspace`` command and return a result dict.

    The result dict follows the session manager's command result protocol:
    - ``{"result": str}`` for text output (displayed as SystemMessageEvent).
    - ``{"error": str}`` for errors.

    Args:
        monitor: The WorkspaceMonitor for the session, or None if unavailable.
        args: Subcommand and arguments, e.g. ``["tree"]``, ``["json"]``.

    Returns:
        Dict with ``"result"`` (formatted string) or ``"error"`` key.
    """
    if monitor is None:
        return {"error": "No workspace monitor active for this session"}

    subcommand = args[0].lower() if args else "tree"

    if subcommand not in _VALID_SUBCOMMANDS:
        return {
            "error": (
                f"Unknown workspace subcommand: {subcommand}\n"
                f"Available: {', '.join(_VALID_SUBCOMMANDS)}"
            )
        }

    snapshot = monitor.get_snapshot()

    if not snapshot:
        return {"result": "No workspace file changes tracked"}

    if subcommand == "tree":
        return {"result": format_tree(snapshot)}
    elif subcommand == "list":
        return {"result": format_list(snapshot)}
    elif subcommand == "json":
        return {"result": format_json(snapshot)}
    elif subcommand == "csv":
        return {"result": format_csv(snapshot)}

    # Unreachable, but defensive
    return {"error": f"Unhandled subcommand: {subcommand}"}


def format_tree(snapshot: List[Dict[str, str]]) -> str:
    """Format snapshot as an indented directory tree.

    Groups files by directory and renders a tree structure similar to
    the ``tree`` command, with status indicators on each file.

    Example output::

        Workspace changes (3 files):
        src/
        ├── main.py  [~]
        └── utils/
            └── helper.py  [+]
        README.md  [-]

    Args:
        snapshot: List of ``{"path": str, "status": str}`` dicts.

    Returns:
        Formatted tree string.
    """
    # Build a nested dict representing the directory structure.
    # Each key is a directory/file name, value is either a nested dict
    # (subdirectory) or a status string (leaf file).
    tree: Dict[str, Any] = {}
    for entry in sorted(snapshot, key=lambda e: e["path"]):
        parts = entry["path"].split("/")
        node = tree
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = entry["status"]

    active = sum(1 for e in snapshot if e["status"] != "deleted")
    deleted = sum(1 for e in snapshot if e["status"] == "deleted")
    header_parts = [f"Workspace changes ({active} file{'s' if active != 1 else ''})"]
    if deleted:
        header_parts.append(f", {deleted} deleted")
    header = "".join(header_parts)

    lines = [header]
    _render_tree_node(tree, lines, prefix="")
    return "\n".join(lines)


def _render_tree_node(
    node: Dict[str, Any],
    lines: List[str],
    prefix: str,
) -> None:
    """Recursively render a tree node into output lines.

    Args:
        node: Dict where keys are names and values are either nested dicts
              (subdirectories) or status strings (files).
        lines: Accumulator list for output lines.
        prefix: Indentation prefix for the current depth.
    """
    entries = sorted(node.items(), key=lambda kv: (not isinstance(kv[1], dict), kv[0]))
    for i, (name, value) in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        child_prefix = prefix + ("    " if is_last else "│   ")

        if isinstance(value, dict):
            lines.append(f"{prefix}{connector}{name}/")
            _render_tree_node(value, lines, child_prefix)
        else:
            symbol = _STATUS_SYMBOLS.get(value, "?")
            lines.append(f"{prefix}{connector}{name}  [{symbol}]")


def format_list(snapshot: List[Dict[str, str]]) -> str:
    """Format snapshot as a flat file list with status indicators.

    Example output::

        [+] src/main.py
        [~] src/utils/helper.py
        [-] README.md

        3 files (2 active, 1 deleted)

    Args:
        snapshot: List of ``{"path": str, "status": str}`` dicts.

    Returns:
        Formatted list string.
    """
    lines = []
    for entry in sorted(snapshot, key=lambda e: e["path"]):
        symbol = _STATUS_SYMBOLS.get(entry["status"], "?")
        lines.append(f"[{symbol}] {entry['path']}")

    active = sum(1 for e in snapshot if e["status"] != "deleted")
    deleted = sum(1 for e in snapshot if e["status"] == "deleted")
    total = len(snapshot)

    summary_parts = [f"\n{total} file{'s' if total != 1 else ''}"]
    if deleted:
        summary_parts.append(f" ({active} active, {deleted} deleted)")
    lines.append("".join(summary_parts))

    return "\n".join(lines)


def format_json(snapshot: List[Dict[str, str]]) -> str:
    """Format snapshot as JSON.

    Output structure::

        {
          "files": [
            {"path": "src/main.py", "status": "created"},
            ...
          ],
          "summary": {"total": 3, "created": 1, "modified": 1, "deleted": 1}
        }

    Args:
        snapshot: List of ``{"path": str, "status": str}`` dicts.

    Returns:
        Pretty-printed JSON string.
    """
    counts: Dict[str, int] = defaultdict(int)
    for entry in snapshot:
        counts[entry["status"]] += 1

    data = {
        "files": sorted(snapshot, key=lambda e: e["path"]),
        "summary": {
            "total": len(snapshot),
            "created": counts.get("created", 0),
            "modified": counts.get("modified", 0),
            "deleted": counts.get("deleted", 0),
        },
    }
    return json.dumps(data, indent=2)


def format_csv(snapshot: List[Dict[str, str]]) -> str:
    """Format snapshot as CSV.

    Output columns: ``path,status``

    Args:
        snapshot: List of ``{"path": str, "status": str}`` dicts.

    Returns:
        CSV string with header row.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["path", "status"])
    for entry in sorted(snapshot, key=lambda e: e["path"]):
        writer.writerow([entry["path"], entry["status"]])
    return output.getvalue().rstrip("\n")
