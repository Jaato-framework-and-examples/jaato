"""Runner-local tool executor — a thin name → callable registry.

Phase 2 ships exactly two registered tools:

- ``echo`` — RPC-overhead probe.
- ``cli_based_tool`` — the migrated cli plugin.

This is intentionally minimal: NO permission plugin, NO reliability
plugin, NO auto-background pool, NO MCP fallback.  All of that is
Phase 3 work.  The §"DO NOT migrate plugins beyond cli" rule applies.

Plugin discovery: Phase 2 hardcodes the allowlist (per §5.3 of the
plan); Phase 3 will replace this with entry-points discovery filtered
by runner-tier classification.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

from .echo_tool import execute_echo
from . import cli_runner


logger = logging.getLogger(__name__)


# Type alias for a runner-tier executor: takes ``args`` dict, returns
# ``(ok, result)``.  Mirrors the in-process plugin executor shape so
# Phase 3 can swap real plugins in without dispatcher changes.
ExecutorFn = Callable[[Dict[str, Any]], Tuple[bool, Dict[str, Any]]]


class ToolExecutor:
    """Registers Phase 2's two runner-tier executors.

    Constructor args feed configuration that the daemon side previously
    applied via plugin lifecycle hooks (``set_workspace_path``,
    ``set_runtime_limits``); for Phase 2 we receive them as plain
    function arguments at runner startup since the cli executor is the
    only consumer.
    """

    def __init__(
        self,
        *,
        workspace_root: Optional[str] = None,
        max_output_chars: int = cli_runner.DEFAULT_MAX_OUTPUT_CHARS,
        tool_timeout_seconds: Optional[float] = None,
    ) -> None:
        self._workspace_root = workspace_root
        self._max_output_chars = max_output_chars
        self._tool_timeout_seconds = tool_timeout_seconds
        self._registry: Dict[str, ExecutorFn] = {
            "echo": execute_echo,
            "cli_based_tool": self._cli_executor,
        }

    def execute(
        self, name: str, args: Dict[str, Any],
    ) -> Tuple[bool, Any]:
        """Dispatch *name* to the registered executor.

        Returns ``(False, {"error": ...})`` for unknown names — same
        shape today's in-process ToolExecutor returns when no
        executor is registered.  The dispatcher serializes this into
        the typed envelope's error payload.
        """
        fn = self._registry.get(name)
        if fn is None:
            return False, {"error": f"No executor registered for {name!r}"}
        return fn(args)

    # --------------------------- private --------------------------------

    def _cli_executor(
        self, args: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        return cli_runner.execute_cli_based_tool(
            args,
            workspace_root=self._workspace_root,
            max_output_chars=self._max_output_chars,
            tool_timeout_seconds=self._tool_timeout_seconds,
        )
