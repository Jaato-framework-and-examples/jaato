"""Unified completion processors — output renderers + payload validators.

Replaces (as of server 0.6.125+) the prior split between
``shared/completion_validators.py`` and the
``render_completion_artifacts`` function in
``shared/dynamic_instructions.py``.  Both surfaces had the same
plumbing — kb Python under ``.jaato/scripts/``, loaded via
``script_loader``, run after ``jsonschema.validate`` passes, block
completion on failure — so they collapsed into one
``completion_processors`` profile field with one invocation pipeline.

**KB author contract.**  Each processor is a Python module under
``.jaato/scripts/processors/`` exposing one or both of these
top-level callables — the framework probes for which symbols are
present:

- ``render(payload: dict, context: RenderContext) -> str | bytes``
  Produces output content.  When the processor's profile entry
  declares an ``output:`` path template, the returned bytes are
  written to disk (atomic ``.tmp`` + ``rename``).  When ``output:``
  is omitted, the return is logged for audit but not persisted —
  validator-as-renderer use case.

- ``validate(payload: dict, context: RenderContext) -> list[str]``
  Returns a list of error strings.  Empty list → pass.  Non-empty →
  completion blocked per the entry's ``on_error`` policy.  Use
  ``context.tool_calls`` to cross-check payload claims against the
  session's actual tool-call history.

Both can be present in one module — useful when a processor both
writes an audit record AND checks consistency.  At least one must
be present; modules with neither surface a kb authoring error to
the agent as a load error.

**Failure modes — all surfaced.**  None silent:

- Script path doesn't resolve → load error
- Module imports but exposes neither ``render`` nor ``validate``
  → load error
- ``render`` / ``validate`` raises → caught, reported as exception error
- ``validate`` returns non-list → reported as malformed return
- ``render`` write fails → write error (filesystem-level)
- ``validate`` returned non-empty list → each entry surfaced

All are bucketed per the processor's ``on_error`` policy
(``fail_completion`` vs ``warn``).  When any ``fail_completion``
error fires, the caller returns the ``validation_failed`` shape to
the model so it retries within ``max_turns``.

**Trust boundary.**  Processors run in the runner subprocess at the
same trust level as dynamic-instructions prefetch scripts and
reactor handlers.  They can read the workspace, import kb helper
modules, and observe the session's history.  They are NOT
sandboxed; treat them as part of the kb codebase.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from .script_loader import load_script_symbol, resolve_script_path

if TYPE_CHECKING:
    from .dynamic_instructions import RenderContext
    from .plugins.subagent.config import CompletionProcessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool-call ledger
# ---------------------------------------------------------------------------


def build_tool_call_ledger(history: List[Any]) -> List[Dict[str, Any]]:
    """Walk session history; pair function_calls with their responses.

    Iterates the message list, builds a chronological ledger of
    every tool call.  Pairing is by ``call_id`` (the provider's
    tool-call identifier carried on both the call Part and the
    response Part).  Calls without a matching response (e.g. the
    final turn whose tool calls are still pending) are emitted with
    ``success=False`` and ``result={"error": "no_response"}`` so a
    validator's "claimed in payload but no successful call" check
    fires deterministically.

    Ledger entry shape (stable dict contract — no framework type
    imports needed for kb authors)::

        {
            "name": str,            # e.g. "renderTemplateToFile"
            "args": dict,           # the args the agent emitted
            "result": dict,         # tool result; failed calls carry
                                    # {"error": "...", "message": "..."}
            "success": bool,        # True iff "error" not in result
            "call_id": str,         # provider call_id for cross-ref
            "turn_index": int,      # 0-based session turn
            "enrichment_metadata": Optional[dict],
                                    # Structured per-plugin metadata
                                    # from tool-result enrichment
                                    # (e.g. LSP diagnostics, artifact
                                    # tracking).  Keyed by plugin name:
                                    # ``{"lsp": {"files_with_diagnostics":
                                    # [...], "total_errors": int,
                                    # "diagnostics": {path: [...]}, ...}}``.
                                    # ``None`` when enrichment didn't
                                    # produce metadata for this call,
                                    # or when the call is in the
                                    # "no_response" pending state.
                                    # In-memory only; not persisted
                                    # across disk-restore.
        }

    Args:
        history: List of ``Message`` objects from
            ``JaatoSession.get_history()``.

    Returns:
        Chronological list of ledger dicts.  Empty when no tool
        calls fired.
    """
    # First pass: collect responses keyed by call_id.  ``Part.function_response``
    # is a ``ToolResult`` whose ``.result`` attribute carries the dict
    # the tool returned, and whose ``.enrichment_metadata`` attribute
    # carries the per-plugin structured metadata from tool-result
    # enrichment (LSP, artifact_tracker, etc.).
    responses_by_call_id: Dict[
        str, Tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]
    ] = {}
    for turn_index, message in enumerate(history):
        parts = getattr(message, "parts", None) or []
        for part in parts:
            fr = getattr(part, "function_response", None)
            if fr is None:
                continue
            call_id = getattr(fr, "call_id", None) or ""
            response_body = getattr(fr, "result", None)
            if not isinstance(response_body, dict):
                # Tool returned a non-dict (string, number, None); wrap
                # so validators see a consistent shape and can still
                # use the ``"error" not in dict`` heuristic.
                response_body = {"result": response_body}
            enrichment_metadata = getattr(fr, "enrichment_metadata", None)
            responses_by_call_id[call_id] = (
                turn_index, response_body, enrichment_metadata,
            )

    ledger: List[Dict[str, Any]] = []
    for turn_index, message in enumerate(history):
        parts = getattr(message, "parts", None) or []
        for part in parts:
            fc = getattr(part, "function_call", None)
            if fc is None:
                continue
            call_id = getattr(fc, "id", None) or ""
            name = getattr(fc, "name", "") or ""
            raw_args = getattr(fc, "args", None)
            args = raw_args if isinstance(raw_args, dict) else {"_raw": raw_args}
            paired = responses_by_call_id.get(call_id)
            if paired is None:
                result: Dict[str, Any] = {"error": "no_response"}
                success = False
                enrichment_metadata = None
            else:
                _, result, enrichment_metadata = paired
                success = "error" not in result
            ledger.append({
                "name": name,
                "args": args,
                "result": result,
                "success": success,
                "call_id": call_id,
                "turn_index": turn_index,
                "enrichment_metadata": enrichment_metadata,
            })
    return ledger


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


@dataclass
class LoadedProcessor:
    """A processor resolved + imported once per session.

    Carries the original ``CompletionProcessor`` config entry alongside
    the loaded symbols.  ``load_error`` is populated when resolution
    or import fails; either or both of ``render_fn`` / ``validate_fn``
    is set when the import succeeded.  A module with neither symbol
    counts as a load failure ("kb authoring error") so the agent
    sees the issue at signal_completion time, never silently.
    """
    processor: Any  # CompletionProcessor — Any to avoid the import cycle
    render_fn: Optional[Callable[..., Any]] = None
    validate_fn: Optional[Callable[..., Any]] = None
    load_error: Optional[str] = None


def load_processors(
    processors: List[Any],  # List[CompletionProcessor]
    workspace_path: Optional[str],
    config_root: Optional[str],
) -> List[LoadedProcessor]:
    """Resolve + import each processor module lazily.

    For every processor entry: resolves the script path via the
    standard ``script_loader`` tier (absolute →
    ``<config_root>/<path>`` → ``~/.jaato/<path>``); loads the
    module; probes for ``render`` and ``validate`` top-level
    callables.  A module with neither symbol surfaces a load error;
    otherwise the loaded callables are cached on the result for
    later invocation.

    Args:
        processors: List of ``CompletionProcessor`` entries from the
            session's profile.
        workspace_path: Session workspace for relative resolution.
        config_root: Config-root override.  Matches the prefetch +
            reactor handler convention.

    Returns:
        Parallel list of ``LoadedProcessor`` instances.  Order
        preserved so error attribution stays clear.
    """
    loaded: List[LoadedProcessor] = []
    for proc in processors:
        script_ref = getattr(proc, "script", None)
        if not script_ref:
            loaded.append(LoadedProcessor(
                processor=proc,
                load_error=f"completion_processor entry missing 'script' field: {proc!r}",
            ))
            continue
        resolved = resolve_script_path(
            script_ref,
            workspace_path=workspace_path,
            config_root=config_root,
        )
        if resolved is None:
            err = (
                f"completion_processor {script_ref!r} could not be located "
                f"(searched config_root={config_root!r}, "
                f"workspace_path={workspace_path!r}, ~/.jaato/)"
            )
            logger.warning(err)
            loaded.append(LoadedProcessor(processor=proc, load_error=err))
            continue
        render_fn = load_script_symbol(
            resolved, symbol="render",
            module_prefix="_jaato_completion_processor",
        )
        validate_fn = load_script_symbol(
            resolved, symbol="validate",
            module_prefix="_jaato_completion_processor",
        )
        if render_fn is None and validate_fn is None:
            err = (
                f"completion_processor {script_ref!r} (resolved to "
                f"{resolved}) exposes neither 'render' nor 'validate' "
                f"top-level callable"
            )
            logger.warning(err)
            loaded.append(LoadedProcessor(processor=proc, load_error=err))
            continue
        loaded.append(LoadedProcessor(
            processor=proc,
            render_fn=render_fn,
            validate_fn=validate_fn,
        ))
    return loaded


# ---------------------------------------------------------------------------
# Invocation
# ---------------------------------------------------------------------------


@dataclass
class ProcessorInvocationResult:
    """Outcome of running a session's completion_processors.

    Each processor lands in one of three buckets based on its
    ``on_error`` policy and what happened:

    Attributes:
        written: Absolute paths of files successfully written by
            ``render``-style processors.  Surfaced into the
            signal_completion result so the agent / downstream
            consumers know what landed.  An empty string in this
            list is the "validator-only render" sentinel (the
            processor ran for side-effect, no file).
        warned: List of ``(processor, message)`` tuples for
            failures whose ``on_error="warn"`` policy let them fail
            non-fatally.  Logged; the completion still succeeds.
        failed: List of ``(processor, message)`` tuples for failures
            whose ``on_error="fail_completion"`` policy forces a
            hard failure.  Caller MUST treat the agent's
            ``signal_completion`` as failed and return a
            self-correction prompt to the model.
    """
    written: List[str] = field(default_factory=list)
    warned: List[Tuple[Any, str]] = field(default_factory=list)
    failed: List[Tuple[Any, str]] = field(default_factory=list)

    @property
    def has_fatal(self) -> bool:
        return bool(self.failed)


def _resolve_output_path(
    template: str,
    payload: Dict[str, Any],
    context: "RenderContext",
) -> str:
    """Substitute ``{field}`` placeholders in an output path template.

    Lookup precedence: payload fields → agent_params → session-derived
    values (``case_id``, ``workspace_path``).  Unknown placeholders
    raise ``KeyError`` deliberately — silent miss would write files to
    weirdly-named paths.

    Relative paths resolve under ``context.workspace_path`` so a
    template like ``output/{case_id}/policy.md`` lands inside the
    session's sandbox, not the daemon's cwd.
    """
    fields: Dict[str, Any] = {}
    if context.workspace_path:
        fields["workspace_path"] = context.workspace_path
    fields.update(context.agent_params or {})
    if isinstance(payload, dict):
        fields.update(payload)
    case_id_val = (
        payload.get("case_id") if isinstance(payload, dict) else None
    ) or context.agent_params.get("case_id")
    if case_id_val:
        fields["case_id"] = case_id_val
    rendered = template.format_map(fields)
    if not os.path.isabs(rendered) and context.workspace_path:
        rendered = os.path.join(context.workspace_path, rendered)
    return rendered


def _bucket(result: ProcessorInvocationResult, proc: Any, msg: str) -> None:
    """Drop a ``(proc, msg)`` into ``failed`` or ``warned`` per policy."""
    on_error = getattr(proc, "on_error", "fail_completion")
    if on_error == "warn":
        result.warned.append((proc, msg))
    else:
        result.failed.append((proc, msg))


def invoke_processors(
    loaded: List[LoadedProcessor],
    payload: Dict[str, Any],
    context: "RenderContext",
) -> ProcessorInvocationResult:
    """Run every loaded processor; aggregate outcomes.

    For each processor: runs ``validate`` first (if present) to
    collect error strings; then runs ``render`` (if present) and
    writes output when an output path is declared.  Errors from
    either path are bucketed into ``failed`` / ``warned`` per the
    processor's ``on_error`` policy.

    **Validate return shapes** (server 0.6.160+).  The ``validate``
    callable can return EITHER:

    - **Legacy** ``list[str]``: every entry is an error, subject to
      the processor's ``on_error`` bucket policy.
    - **New** ``ProcessorResult`` TypedDict from
      ``jaato_sdk.cascade_authoring`` —
      ``{"errors": list[str], "warnings": list[str]}``.  ``errors``
      go through ``_bucket`` (subject to ``on_error``); ``warnings``
      go DIRECTLY to ``result.warned`` regardless of policy
      (advisory entries never escalate).

    Both shapes coexist indefinitely; legacy processors don't need
    to migrate.

    Errors are NEVER short-circuited: every processor runs even if
    a prior one failed.  The agent sees the full picture on retry
    rather than playing whack-a-mole one error at a time.

    Args:
        loaded: Output of :func:`load_processors` — parallel list of
            resolved + imported processor entries.
        payload: The agent's validated ``signal_completion`` payload.
        context: :class:`RenderContext` carrying workspace_path,
            config_root, agent_params, tool_calls ledger, etc.

    Returns:
        :class:`ProcessorInvocationResult` with written paths,
        warnings, and fatal failures.
    """
    result = ProcessorInvocationResult()

    for lp in loaded:
        proc = lp.processor
        script_ref = getattr(proc, "script", "?")

        if lp.load_error is not None:
            _bucket(result, proc, lp.load_error)
            continue

        # 1. Validate first (cheaper, no side-effects)
        if lp.validate_fn is not None:
            try:
                raw = lp.validate_fn(payload, context)
            except Exception as exc:
                _bucket(result, proc, (
                    f"completion_processor {script_ref!r} validate "
                    f"raised {type(exc).__name__}: {exc}"
                ))
                logger.exception(
                    "completion_processor %r validate raised", script_ref,
                )
                raw = None
            if raw is not None:
                # Server 0.6.160+: accept BOTH legacy ``list[str]`` AND
                # the new ``ProcessorResult`` TypedDict from
                # ``jaato_sdk.cascade_authoring``.  Legacy returns are
                # treated as all-errors (backwards-compat).  TypedDict
                # returns split errors → ``_bucket`` (subject to
                # ``on_error`` policy) and warnings → ``result.warned``
                # (always advisory, never escalates regardless of
                # policy).  See ``ProcessorResult`` docstring for the
                # full contract.
                errors_iter: List[str] = []
                warnings_iter: List[str] = []
                shape_ok = True
                if isinstance(raw, list):
                    errors_iter = raw  # type: ignore[assignment]
                elif isinstance(raw, dict):
                    errors_iter = raw.get("errors", []) or []
                    warnings_iter = raw.get("warnings", []) or []
                else:
                    shape_ok = False
                    _bucket(result, proc, (
                        f"completion_processor {script_ref!r} validate "
                        f"returned {type(raw).__name__}; expected "
                        f"list[str] or ProcessorResult TypedDict "
                        f"({{'errors': [...], 'warnings': [...]}})"
                    ))

                if shape_ok:
                    # Errors path: subject to on_error policy via _bucket.
                    for item in errors_iter:
                        if not isinstance(item, str):
                            _bucket(result, proc, (
                                f"completion_processor {script_ref!r} "
                                f"validate returned a non-string error "
                                f"entry: {item!r}"
                            ))
                        else:
                            _bucket(result, proc, f"[{script_ref}] {item}")
                    # Warnings path: always advisory, bypasses _bucket.
                    for item in warnings_iter:
                        if not isinstance(item, str):
                            result.warned.append((proc, (
                                f"completion_processor {script_ref!r} "
                                f"validate returned a non-string warning "
                                f"entry: {item!r}"
                            )))
                        else:
                            result.warned.append(
                                (proc, f"[{script_ref}] {item}")
                            )

        # 2. Render (and maybe write) — runs even when validate already
        #    queued failures; processors are independent surfaces.
        if lp.render_fn is not None:
            try:
                content = lp.render_fn(payload, context)
            except Exception as exc:
                _bucket(result, proc, (
                    f"completion_processor {script_ref!r} render raised "
                    f"{type(exc).__name__}: {exc}"
                ))
                logger.exception(
                    "completion_processor %r render raised", script_ref,
                )
                continue
            output_template = getattr(proc, "output", None)
            if output_template is None:
                # Validator-as-renderer: log return, don't write.
                if isinstance(content, (str, bytes, bytearray)) and content:
                    preview = (
                        content[:200] if isinstance(content, str)
                        else content[:200].decode("utf-8", errors="replace")
                    )
                    logger.info(
                        "completion_processor: render-only-no-output %r "
                        "emitted: %s", script_ref, preview,
                    )
                result.written.append("")
                continue
            try:
                output_path = _resolve_output_path(
                    output_template, payload, context,
                )
            except KeyError as exc:
                _bucket(result, proc, (
                    f"output path template references unknown field {exc} "
                    f"(template={output_template!r})"
                ))
                continue
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                tmp_path = output_path + ".tmp"
                mode = "wb" if isinstance(content, (bytes, bytearray)) else "w"
                open_kwargs: Dict[str, Any] = {"mode": mode}
                if mode == "w":
                    open_kwargs["encoding"] = "utf-8"
                with open(tmp_path, **open_kwargs) as fh:
                    fh.write(content)
                os.replace(tmp_path, output_path)
                result.written.append(output_path)
            except OSError as exc:
                logger.exception(
                    "completion_processor: write failed for %s", output_path,
                )
                _bucket(result, proc, (
                    f"write failed: {output_path}: {exc}"
                ))

    return result


def collect_failure_messages(result: ProcessorInvocationResult) -> List[str]:
    """Flatten ``result.failed`` into the string list the model sees.

    Called by :class:`shared.lifecycle_tools.LifecycleTools` when
    ``signal_completion`` must return a ``validation_failed`` retry
    prompt.  Each entry is already prefixed by ``_bucket`` (validate
    errors carry ``[<script>] <msg>``; raises / write failures
    carry just the message — script path is already in the text).
    """
    return [msg for _, msg in result.failed]
