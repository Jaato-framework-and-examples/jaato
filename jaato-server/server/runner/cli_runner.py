"""Runner-side cli executor (Phase 2 task 2.5).

Migrates the body of ``shared/plugins/cli/plugin.py:_execute`` to the
runner side.  The daemon-side cli plugin is unchanged in this commit
(its ``_execute`` becomes a thin RPC stub in task 2.5); the actual
subprocess spawn happens here, inside the AppArmor-confined runner
process so the spawned child inherits the per-session profile.

Phase 2 simplifications vs. the daemon-side plugin:

- **No path-validation step.**  ``_validate_command_paths`` was a
  defense-in-depth duplicate of what AppArmor enforces at the kernel
  level; with the runner confined, kernel-side EACCES replaces the
  in-process check.  Path validation may return in Phase 3 if a soft-
  mode (no-AppArmor) deployment surfaces a need for it.
- **No background promotion.**  Auto-background is daemon-tier in
  Phase 2 (the ``BackgroundCapable`` mixin lives daemon-side); the
  runner just runs the command synchronously and returns.  Phase 3
  rewires backgrounding through the runner.

The `on_output` callback comes from the RPC dispatcher's thread-local
(installed by :class:`server.runner.rpc.RunnerRPC` before invoking the
executor); we wire it into ``shared.subprocess_runner.run_command``'s
streaming hook so each stdout line lands as a ``StreamFrame`` on the
wire.

Cancellation: the runner's per-call ``CancelToken`` (also from the
dispatcher's thread-local) is read by ``run_command`` via the existing
``get_current_cancel_token()`` contract — but ``run_command`` reads
from ``shared.ai_tool_runner._thread_local``, NOT the runner's own
``server.runner.rpc._thread_local``.  We bridge by setting the same
value on both thread-locals at entry; the dispatcher already sets
the runner-side one.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from shared.cli_path_policy import CLI_EXE_NOT_FOUND_HINT, precheck_cli_args
from shared.subprocess_runner import requires_shell, run_command, RunResult

# We bridge the runner-side cancel/output thread-local onto the
# shared.ai_tool_runner thread-local that ``run_command`` reads.  This
# is the single integration point keeping the existing run_command
# contract intact.
import shared.ai_tool_runner as _ai_tool_runner

from .rpc import (
    get_current_cancel_token,
    get_current_output_callback,
)


logger = logging.getLogger(__name__)

DEFAULT_MAX_OUTPUT_CHARS = 50000  # ~12k tokens at 4 chars/token; mirrors cli plugin.


def execute_cli_based_tool(
    args: Dict[str, Any],
    *,
    workspace_root: Optional[str] = None,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    tool_timeout_seconds: Optional[float] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Execute a cli command and return ``(ok, result_dict)``.

    Result-dict shape mirrors today's cli plugin output:
    - ``stdout`` / ``stderr`` / ``returncode`` (always present on the
      success path).
    - ``truncated`` (bool) + ``truncation_message`` (str) when the
      output cap fired.
    - ``timed_out`` (bool) + ``timeout_seconds`` (float) when the
      wall-clock cap fired.
    - ``_telemetry`` (dict) of ``jaato.cli.*`` attributes for the
      session's tool span.
    - On hard failure: ``error`` (str) + optional ``hint``.

    Args:
        args: ``{"command": str, "args": list?}`` mirroring the cli plugin's
            ToolSchema parameters.  An ``extra_paths`` key is REFUSED rather
            than honoured (#697) — PATH extension decides which binary a
            command name resolves to, and the permission decision upstream is
            keyed on the command text alone, so a caller-supplied value would
            put executable resolution outside the value that was approved.
        workspace_root: Working directory for the spawned process.
            Set by the runner's ``__main__`` from
            ``JAATO_RUNNER_WORKSPACE`` env at startup.
        max_output_chars: Cap on captured stdout + stderr.
        tool_timeout_seconds: Wall-clock timeout (None = no cap).

    Returns:
        ``(ok, result)``.  ``ok=False`` only on hard pre-spawn errors
        (missing command, unresolvable executable); a non-zero exit
        from the spawned process is reported as ``ok=True`` with
        ``returncode != 0`` so the model sees the actual exit status.
    """
    # Is this call runnable as given?  A caller-supplied ``extra_paths`` is
    # refused (#697) — this surface had no operator tier at all, so the key
    # came straight out of the model's tool-call arguments and reached
    # ``shutil.which`` while the permission decision upstream was keyed on the
    # command text alone — and an absent command is an error, as before.
    refusal = precheck_cli_args(args, surface="runner cli executor")
    if refusal is not None:
        return False, refusal

    command = args.get("command")
    arg_list = args.get("args")

    # Merge separate arg_list into the command string when it doesn't
    # need shell interpretation — matches the daemon-side plugin's
    # behavior so the runner is a drop-in replacement on the wire.
    import shlex
    if arg_list and not requires_shell(command):
        command = " ".join(
            [shlex.quote(command)] + [shlex.quote(a) for a in arg_list]
        )

    # No PATH extension here.  The only ``extra_paths`` this executor ever saw
    # was the caller-supplied one refused above; the operator's list lives on
    # the daemon-side ``CliPlugin`` and is applied there (#697).
    extra_env: Optional[Dict[str, str]] = None

    # Bridge the runner-side cancel + output to the shared.ai_tool_runner
    # thread-local that ``run_command`` reads.  Set on entry, restore
    # on exit so we don't leak state into other runner-internal code.
    rpc_token = get_current_cancel_token()
    rpc_output_cb = get_current_output_callback()

    # ``run_command``'s on_stdout_line is line-string only (no source
    # discriminator); we forward each line to the dispatcher's
    # callback as a ``stdout`` chunk.  stderr lines are captured but
    # not streamed (matching today's cli sync path — only stdout is
    # streamed live; stderr arrives in the final result).
    streaming_cb = None
    if rpc_output_cb is not None:
        def streaming_cb(line: str) -> None:  # noqa: E306
            rpc_output_cb("stdout", line + "\n", None)

    prior_token = getattr(_ai_tool_runner._thread_local, "cancel_token", None)
    _ai_tool_runner._thread_local.cancel_token = rpc_token
    try:
        try:
            r: RunResult = run_command(
                command,
                cwd=workspace_root,
                timeout=tool_timeout_seconds,
                max_output_chars=max_output_chars,
                extra_env=extra_env,
                on_stdout_line=streaming_cb,
                check_cancel=True,
                # No cgroup attach in Phase 2 — runtime-limits cgroup
                # plumbing is daemon-tier today and Phase 3 hands it
                # to the runner.
                preexec_fn=None,
            )
        except Exception as exc:  # noqa: BLE001 — boundary surface
            # Includes CancelledException from run_command's cancel
            # check; the dispatcher serializes the exception type
            # into the typed envelope's error.type.
            raise
    finally:
        _ai_tool_runner._thread_local.cancel_token = prior_token

    # Executable-not-found surfaces as a clear error dict so the model
    # sees a clean actionable message — same shape as the daemon-side
    # cli plugin.
    if r.returncode == 127 and "not found in PATH" in r.stderr:
        return False, {
            "error": f"cli_based_tool: {r.stderr}",
            "hint": CLI_EXE_NOT_FOUND_HINT,
        }

    result: Dict[str, Any] = {
        "stdout": r.stdout,
        "stderr": r.stderr,
        "returncode": r.returncode,
    }

    if r.truncated:
        result["truncated"] = True
        result["truncation_message"] = (
            f"Output truncated to {max_output_chars} chars. "
            "Consider using more specific commands (e.g., add filters, "
            "limits, or pipe to head/tail)."
        )

    if getattr(r, "timed_out", False):
        result["timed_out"] = True
        if tool_timeout_seconds is not None:
            result["timeout_seconds"] = tool_timeout_seconds

    # Plugins continue to write ``_telemetry`` into the result dict;
    # the runner's ``_emit_response`` (Phase 3 §3.15) lifts it onto
    # ``envelope.telemetry`` for the wire form.  Daemon-side
    # ``_forward_via_runner`` reinjects on the way back so existing
    # consumers (jaato_session.py's OTel forwarder) see the same
    # shape they always have.
    result["_telemetry"] = {
        "jaato.cli.command": command[:200],
        "jaato.cli.returncode": r.returncode,
        "jaato.cli.stdout_bytes": len(r.stdout),
        "jaato.cli.stderr_bytes": len(r.stderr),
        "jaato.cli.shell_mode": requires_shell(command),
        "jaato.cli.cwd": str(workspace_root or ""),
    }

    return True, result
