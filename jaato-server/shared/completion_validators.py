"""Semantic validators for ``signal_completion`` payloads.

Schema validation (``jsonschema.validate``) accepts any payload whose
structure matches the profile's ``completion_payload_schema``.  That's
enough to catch most authoring mistakes — wrong types, missing keys,
extra fields when ``additionalProperties: false`` — but it can't catch
the bug class where the agent's payload is *structurally well-formed
yet semantically false*.  Concrete example (kb-enablement-2.0 v117):
the codegen agent emitted ``files[]`` with 20 entries, 6 of which
were fabricated — their ``renderTemplateToFile`` calls had actually
returned errors (``Variable shape validation failed``,
``basePackagePath substituted to empty string``), but the schema
doesn't know that and the agent rationalized success.

This module provides the hook surface kb authors use to plug in
domain-specific Python that cross-checks payload claims against the
session's actual tool-call history and the workspace filesystem.
The validator returns a list of error strings; non-empty list →
``signal_completion`` returns the same ``validation_failed`` shape
as a schema failure, forcing the agent to retry within its
``max_turns`` budget.

**Author contract.**  A validator is a Python module exposing one
top-level callable::

    def validate(
        payload: dict,                    # the schema-validated args
        tool_calls: list[dict],           # pre-computed ledger
        workspace_path: pathlib.Path,     # session workspace dir
        ctx: dict,                        # agent_params + session metadata
    ) -> list[str]:                       # empty = OK; non-empty = errors
        ...

The ``tool_calls`` ledger entries have a stable dict shape that does
not require importing framework types::

    {
        "name": str,            # e.g. "renderTemplateToFile"
        "args": dict,           # the args the agent emitted
                                # (raw — if hashed ids like tpl_xxxx,
                                # resolve via shared.tool_id_map.id_to_name)
        "result": dict,         # the tool's return; failed calls
                                # carry {"error": "...", "message": "..."}
        "success": bool,        # True iff "error" not in result
        "call_id": str,         # provider call_id for cross-referencing
        "turn_index": int,      # 0-based session turn
    }

**Path resolution.**  Validator paths in the profile YAML use the
same loader as prefetch + reactor handlers — see
:func:`shared.script_loader.resolve_script_path` for the tier order
(absolute → ``<config_root>/<path>`` → ``~/.jaato/<path>``).
Typical kb layout::

    .jaato/
      scripts/
        validators/
          codegen_files_exist.py
          codegen_render_succeeded.py
      profiles/
        codegen.json   # → completion_validators: ["scripts/validators/..."]

**Failure modes.**  All four are surfaced to the agent as
``validation_failed`` so the kb author sees a bad validator AT signal
time, never silently:

- Validator path doesn't resolve → ``"completion_validator '<path>' could not be located"``
- Validator imports but ``validate`` symbol is missing/not callable → ``"completion_validator '<path>' has no callable 'validate' symbol"``
- Validator raises → ``"completion_validator '<path>' raised <ExceptionType>: <msg>"``
- Validator returns non-list-of-str → ``"completion_validator '<path>' returned <bad shape>; expected list[str]"``

**Trust boundary.**  Validators run in the daemon process at the same
trust level as ``shared/dynamic_instructions.py`` render scripts and
reactor handlers.  They can read the filesystem, import the kb's
helper modules, and observe the session's history dict.  They are
NOT sandboxed; treat them as part of the kb codebase.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .script_loader import load_script_symbol, resolve_script_path

logger = logging.getLogger(__name__)


def build_tool_call_ledger(history: List[Any]) -> List[Dict[str, Any]]:
    """Walk session history, pair function_calls with their responses.

    Iterates the message list, builds the ledger entries documented in
    the module docstring.  Pairing is by ``call_id`` (the provider's
    tool-call identifier carried on both the call Part and the
    response Part).  Calls without a matching response (e.g. the
    final turn whose tool calls are still pending) are emitted with
    ``success=False`` and ``result={"error": "no_response"}`` so a
    validator's "claimed in payload but no successful call" check
    fires deterministically.

    Args:
        history: List of ``Message`` objects from
            ``JaatoSession.get_history()``.  Each message's ``parts``
            list may carry ``function_call`` and ``function_response``
            Part instances per the model-provider protocol.

    Returns:
        List of ledger dicts in chronological order.  Empty list when
        the history contains no tool calls.
    """
    # First pass: collect responses keyed by call_id so the second pass
    # can pair O(1).  Part.function_response is a ToolResult; its
    # ``.result`` attribute carries the dict the tool returned.
    responses_by_call_id: Dict[str, Tuple[int, Dict[str, Any]]] = {}
    for turn_index, message in enumerate(history):
        parts = getattr(message, "parts", None) or []
        for part in parts:
            fr = getattr(part, "function_response", None)
            if fr is None:
                continue
            call_id = getattr(fr, "call_id", None) or ""
            response_body = getattr(fr, "result", None)
            if not isinstance(response_body, dict):
                # Tool returned a non-dict (string, number, None);
                # wrap so validators see consistent shape and can
                # still detect failures via the ``"error" not in dict``
                # heuristic.
                response_body = {"result": response_body}
            responses_by_call_id[call_id] = (turn_index, response_body)

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
            if isinstance(raw_args, dict):
                args = raw_args
            else:
                # Some providers carry args as JSON string in older paths;
                # leave as-is rather than guessing — validator can decide.
                args = {"_raw": raw_args}
            paired = responses_by_call_id.get(call_id)
            if paired is None:
                result: Dict[str, Any] = {"error": "no_response"}
                success = False
            else:
                _, result = paired
                success = "error" not in result
            ledger.append(
                {
                    "name": name,
                    "args": args,
                    "result": result,
                    "success": success,
                    "call_id": call_id,
                    "turn_index": turn_index,
                }
            )
    return ledger


def load_validators(
    paths: List[str],
    workspace_path: Optional[str],
    config_root: Optional[str],
) -> List[Tuple[str, Optional[Callable[..., List[str]]], Optional[str]]]:
    """Resolve + import each validator path lazily.

    Returns a parallel list of ``(path, callable, load_error)``.  When
    a path resolves and imports cleanly, the callable is set and
    ``load_error`` is ``None``.  When anything fails, the callable
    is ``None`` and ``load_error`` carries a kb-actionable message
    that the invocation site surfaces back to the agent as a
    validation error.

    Args:
        paths: Validator path strings from the profile field.
        workspace_path: Session workspace for relative resolution.
        config_root: Config-root override for the
            ``<config_root>/<path>`` tier.  Matches the prefetch +
            reactor handler convention.
    """
    loaded: List[Tuple[str, Optional[Callable[..., List[str]]], Optional[str]]] = []
    for path_str in paths:
        resolved = resolve_script_path(
            path_str,
            workspace_path=workspace_path,
            config_root=config_root,
        )
        if resolved is None:
            err = (
                f"completion_validator {path_str!r} could not be located "
                f"(searched config_root={config_root!r}, "
                f"workspace_path={workspace_path!r}, ~/.jaato/)"
            )
            logger.warning(err)
            loaded.append((path_str, None, err))
            continue
        fn = load_script_symbol(
            resolved, symbol="validate", module_prefix="_jaato_completion_validator",
        )
        if fn is None:
            err = (
                f"completion_validator {path_str!r} (resolved to "
                f"{resolved}) has no callable 'validate' symbol"
            )
            logger.warning(err)
            loaded.append((path_str, None, err))
            continue
        loaded.append((path_str, fn, None))
    return loaded


def invoke_validators(
    loaded: List[Tuple[str, Optional[Callable[..., List[str]]], Optional[str]]],
    payload: Dict[str, Any],
    tool_calls: List[Dict[str, Any]],
    workspace_path: Path,
    ctx: Dict[str, Any],
) -> List[str]:
    """Run every loaded validator, aggregate all errors into one list.

    Returns the combined error list.  Empty list means every validator
    passed (or no validators were configured).  Each error string is
    prefixed with the validator's path so the agent can attribute
    the failure to a specific validator when multiple are configured.

    Errors are aggregated (not short-circuited) so the agent sees the
    full picture on its retry — playing whack-a-mole one validator at
    a time would burn turns from ``max_turns`` without converging.
    """
    errors: List[str] = []
    for path_str, fn, load_err in loaded:
        if load_err is not None:
            errors.append(load_err)
            continue
        assert fn is not None  # load_err is None ⇒ fn is set
        try:
            raw = fn(payload, tool_calls, workspace_path, ctx)
        except Exception as exc:
            errors.append(
                f"completion_validator {path_str!r} raised "
                f"{type(exc).__name__}: {exc}"
            )
            logger.exception(
                "completion_validator %r raised", path_str,
            )
            continue
        if not isinstance(raw, list):
            errors.append(
                f"completion_validator {path_str!r} returned "
                f"{type(raw).__name__}; expected list[str]"
            )
            continue
        for item in raw:
            if not isinstance(item, str):
                errors.append(
                    f"completion_validator {path_str!r} returned a "
                    f"non-string error entry: {item!r}"
                )
                continue
            errors.append(f"[{path_str}] {item}")
    return errors
