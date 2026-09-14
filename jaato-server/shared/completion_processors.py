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
  session's actual tool-call history.  A richer return —
  ``jaato_sdk.cascade_authoring.ProcessorResult``, a dict with
  ``errors`` / ``warnings`` / ``incomplete`` / ``faults`` — separates
  a wrong answer (``errors``, retryable, budgeted) from an environment
  fault (``faults``, unfixable by retrying, budget-exempt).

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

**The retry loop does not terminate on its own** (issue #768).  The
processor refuses, the agent re-claims completion, the processor
refuses again — an observed run spent seven refusals in 156 seconds
on the same two errors and ended with its whole budget gone and no
verdict.  ``max_turns`` bounds the SESSION, not this gate, and
nothing upstream bounds the gate: ``MAX_COMPLETION_NUDGES`` bounds
the opposite direction (an agent that stops WITHOUT signalling).  So
a processor entry may declare ``max_refusals:`` with an
``on_exhausted:`` policy (``allow`` / ``fail``), and the framework
counts the refusals on the per-session :class:`LoadedProcessor` —
see :func:`_record_errors`.  Only genuine wrong answers are counted:
never a ``faults[]`` entry, and never a broken gate (a load error, a
raise, a malformed return, a failed write), because retries cannot
fix those and a gate that did not run must never read as one that
passed.

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

    **Thin alias of** :func:`jaato_sdk.completion_processors.build_ledger`.
    The pairing rule lives in the SDK because that is the layer a consumer
    can import: the SDK published the ledger entry TYPE and no way to obtain
    a ledger, so anyone wanting one wrote their own pairing, and every copy
    had to re-derive that pairing is by IDENTIFIER and not by name-in-order.
    Copies of a rule rot independently unless something executes the
    comparison; there is now one rule instead of one per consumer.

    Kept as a name because completion processors and the cascade machinery
    call it, and because it takes the in-process ``Message`` carrier while
    the SDK function takes either. Behaviour is unchanged — see the SDK
    docstring for the full contract (pairing, the ``no_response`` pending
    state, the ``"error" not in result`` success rule, and the fact that
    ``enrichment_metadata`` is in-memory only).

    Args:
        history: List of ``Message`` objects from
            ``JaatoSession.get_history()``.

    Returns:
        Chronological list of ledger dicts.  Empty when no tool calls fired.
    """
    from jaato_sdk.completion_processors import build_ledger

    return build_ledger(history)


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

    **This object is also the framework's declared home for
    per-session processor state** (issue #768).  ``LifecycleTools``
    loads processors ONCE per session and caches the resulting list
    (``LifecycleTools._processors_loaded``), so the same instance is
    handed to every ``signal_completion`` / ``prepare_completion``
    call of that session — which is what lets ``refusals`` and
    ``fault_blocks_used`` accumulate across the retry loop.  That
    caching used to be an undocumented implementation detail on which
    processor authors nonetheless relied, by keeping their refusal
    counter in a module-level global; if it ever changed, the counter
    silently degraded to a no-op bound and the "ceiling" quietly
    stopped existing.  Stating the guarantee here and holding the
    state on the framework's own object is what retires that folklore
    (#765).  ``shared/tests/test_completion_processor_refusal_budget.py``
    holds the guard.

    Attributes:
        processor: The profile's ``CompletionProcessor`` entry.
        render_fn: The module's ``render`` callable, if it has one.
        validate_fn: The module's ``validate`` callable, if it has one.
        load_error: Why resolution/import failed, else ``None``.
        refusals: How many invocations of this processor have BLOCKED
            completion by returning ``errors[]``.  Compared against the
            entry's ``max_refusals``.  Counts one per invocation, not
            one per message, and counts ONLY genuine wrong-answer
            refusals — never a fault, never a broken gate (a load
            error, a raise, a malformed return, a failed write), which
            the agent's retries cannot fix and so must not pay for.
        fault_blocks_used: Whether the budget-exempt ``faults[]``
            channel has already blocked once this session.  A fault is
            unfixable by retrying, so it blocks exactly one
            round-trip — long enough for the agent to record it in its
            payload — and is advisory from then on.  Blocking
            repeatedly on a condition no retry can clear is the
            non-terminating loop the budget exists to prevent.
    """
    processor: Any  # CompletionProcessor — Any to avoid the import cycle
    render_fn: Optional[Callable[..., Any]] = None
    validate_fn: Optional[Callable[..., Any]] = None
    load_error: Optional[str] = None
    refusals: int = 0
    fault_blocks_used: int = 0


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
            Also where an ``errors[]`` entry lands once the
            processor's ``max_refusals`` ceiling is spent under
            ``on_exhausted="allow"`` — the completion is accepted as
            it stands, but the errors survive in the audit trail
            rather than vanishing.  Advisory ``warnings[]`` entries
            arrive here regardless of ``on_error``, as does a
            ``faults[]`` entry after its one blocking round-trip.
        failed: List of ``(processor, message)`` tuples for failures
            whose ``on_error="fail_completion"`` policy forces a
            hard failure.  Caller MUST treat the agent's
            ``signal_completion`` as failed and return a
            self-correction prompt to the model.
        incomplete: List of ``(processor, message)`` tuples emitted by
            ``phase: "completeness"`` processors via the
            ``ProcessorResult.incomplete[]`` channel (server 0.6.199+).
            These are NEITHER fatal NOR advisory — they are SEMANTIC
            "not done yet" signals that gate the composite
            ``is_complete`` verdict during ``prepare_completion``.
            Non-empty → ``is_complete`` stays False and the messages
            surface to the model as neutral "still needed" guidance
            (no retry penalty, no completion block).  Consumed by
            ``LifecycleTools._execute_prepare_completion``; ignored by
            the ``signal_completion`` finalization path (a completeness
            verdict is meaningless once the agent chooses to finalize).
    """
    written: List[str] = field(default_factory=list)
    warned: List[Tuple[Any, str]] = field(default_factory=list)
    failed: List[Tuple[Any, str]] = field(default_factory=list)
    incomplete: List[Tuple[Any, str]] = field(default_factory=list)

    @property
    def has_fatal(self) -> bool:
        return bool(self.failed)

    @property
    def has_incomplete(self) -> bool:
        return bool(self.incomplete)


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


#: Sentinel for "this processor's ``validate`` raised and the failure has
#: already been reported".  Distinct from ``None``, which is what a
#: ``validate`` that forgets to return produces — that is a malformed
#: return, not a pass (issue #768 rule 5).
_RAISED = object()


def _labelled(entries: List[Any], script_ref: str, channel: str) -> List[str]:
    """Prefix each entry with its script, naming any non-string entry.

    Every channel of a ``validate`` return goes through here so a
    malformed entry is REPORTED rather than coerced or dropped — a
    dropped entry is an error path that produces the same value as
    success, which is the defect class this module is most prone to
    (issue #768 rule 5).

    Args:
        entries: One channel's raw entries, as returned by the kb script.
        script_ref: The processor's script path, for attribution.
        channel: ``"error"`` / ``"warning"`` / ``"incomplete"`` / ``"fault"``
            — used only in the message describing a non-string entry.

    Returns:
        One message per entry, in order.
    """
    out: List[str] = []
    for item in entries:
        if isinstance(item, str):
            out.append(f"[{script_ref}] {item}")
        else:
            out.append(
                f"completion_processor {script_ref!r} validate returned a "
                f"non-string {channel} entry: {item!r}"
            )
    return out


def _classify_validate_return(
    raw: Any,
) -> Optional[Tuple[List[Any], List[Any], List[Any], List[Any]]]:
    """Split a ``validate`` return into its four channels.

    Accepts the legacy ``list[str]`` shape (all-errors) and the
    :class:`jaato_sdk.cascade_authoring.ProcessorResult` TypedDict.

    Args:
        raw: Whatever the kb script returned (already known non-None).

    Returns:
        ``(errors, warnings, incomplete, faults)``, or ``None`` when the
        return is neither shape — which the caller reports as a broken
        gate rather than treating as "no errors".
    """
    if isinstance(raw, list):
        return raw, [], [], []
    if isinstance(raw, dict):
        return (
            raw.get("errors", []) or [],
            raw.get("warnings", []) or [],
            raw.get("incomplete", []) or [],
            raw.get("faults", []) or [],
        )
    return None


def _budget_note(proc: Any, script_ref: str, remaining: int) -> str:
    """The remaining-attempts sentence appended to a bounded refusal.

    The return value of a processor is read by a MODEL about to try
    again, not by a human reading a log, so it is written as an
    instruction for the retry: name the attempts left, and say that
    re-sending an unchanged claim spends one (issue #768 rule 7).  The
    framework writes it rather than each author, because it is the
    framework that owns the count.

    Args:
        proc: The ``CompletionProcessor`` entry (read for ``on_exhausted``).
        script_ref: The processor's script path, for attribution.
        remaining: Attempts left AFTER the refusal being reported.

    Returns:
        One message string, already script-prefixed.
    """
    outcome = (
        "accepted as it stands and processed unfinished"
        if getattr(proc, "on_exhausted", "allow") == "allow"
        else "refused for good"
    )
    if remaining <= 0:
        return (
            f"[{script_ref}] That was your last attempt at this gate — the "
            f"next signal_completion will be {outcome}, whether or not the "
            f"errors above are fixed. Fix them now."
        )
    return (
        f"[{script_ref}] You have {remaining} further attempt(s) at this gate "
        f"before this completion is {outcome}. Re-sending the same claim "
        f"without changing anything spends one."
    )


def _record_errors(
    result: ProcessorInvocationResult, lp: LoadedProcessor,
    errors: List[Any], script_ref: str,
) -> None:
    """Bucket this invocation's ``errors[]``, applying the refusal budget.

    A processor with no ``max_refusals`` behaves exactly as it always
    did: every error blocks, forever.  With a ceiling declared, this
    counts ONE refusal per invocation (not per message) and, once the
    ceiling is spent, applies ``on_exhausted``:

    - ``"allow"`` — the errors are downgraded to warnings and the
      completion stands unfinished.  The checks still failed and
      whatever grades the run says so; a FAIL verdict carries
      information where a BLOCKED arm carries none.
    - ``"fail"`` — the errors keep blocking.

    Args:
        result: The accumulating invocation result.
        lp: The per-session loaded processor holding ``refusals``.
        errors: The raw ``errors[]`` channel.
        script_ref: The processor's script path, for attribution.
    """
    if not errors:
        return
    proc = lp.processor
    messages = _labelled(errors, script_ref, "error")
    ceiling = getattr(proc, "max_refusals", None)
    if ceiling is None:
        for msg in messages:
            _bucket(result, proc, msg)
        return
    if lp.refusals >= ceiling:
        spent = (
            f"[{script_ref}] refusal budget spent (max_refusals={ceiling}); "
            f"on_exhausted={getattr(proc, 'on_exhausted', 'allow')!r}"
        )
        if getattr(proc, "on_exhausted", "allow") == "allow":
            logger.warning(
                "completion_processor %r: %d refusal(s) spent — accepting the "
                "completion as it stands and downgrading %d error(s) to "
                "warnings", script_ref, ceiling, len(messages),
            )
            for msg in messages + [spent]:
                result.warned.append((proc, msg))
        else:
            for msg in messages + [spent]:
                _bucket(result, proc, msg)
        return
    lp.refusals += 1
    for msg in messages:
        _bucket(result, proc, msg)
    _bucket(result, proc, _budget_note(proc, script_ref, ceiling - lp.refusals))


def _record_faults(
    result: ProcessorInvocationResult, lp: LoadedProcessor,
    faults: List[Any], script_ref: str,
) -> None:
    """Bucket the budget-exempt ``faults[]`` channel.

    A fault is an ENVIRONMENT fault — a missing acceptance script, an
    absent parameter, a checks timeout — as opposed to a wrong answer.
    Nothing the agent's next attempt does can clear it, so:

    - it never consumes a refusal (a retryable message about an
      unfixable fault burns the whole budget without ever producing a
      verdict — issue #768 rule 6); and
    - it blocks exactly ONCE per session, which is the single
      round-trip the agent needs to record the fault in its payload.
      Blocking again on a condition no retry can clear is precisely the
      loop that does not terminate.

    Args:
        result: The accumulating invocation result.
        lp: The per-session loaded processor holding ``fault_blocks_used``.
        faults: The raw ``faults[]`` channel.
        script_ref: The processor's script path, for attribution.
    """
    if not faults:
        return
    proc = lp.processor
    messages = _labelled(faults, script_ref, "fault")
    if lp.fault_blocks_used:
        for msg in messages:
            result.warned.append((proc, msg))
        return
    lp.fault_blocks_used = 1
    for msg in messages:
        _bucket(result, proc, msg)


def _record_validate_outcome(
    result: ProcessorInvocationResult, lp: LoadedProcessor,
    raw: Any, script_ref: str,
) -> None:
    """Route one ``validate`` return into the result's four buckets.

    Channel semantics, all four of which a processor may use at once:

    ============  ==========================  ======================
    channel       blocks completion?          consumes a refusal?
    ============  ==========================  ======================
    ``errors``    yes, per ``on_error``       yes (once per call)
    ``faults``    once per session            never
    ``warnings``  never                       never
    ``incomplete``never (gates is_complete)   never
    ============  ==========================  ======================

    A return that is neither shape is reported as a malformed return
    and BLOCKS: a gate that did not run must never read as a gate that
    passed.

    Args:
        result: The accumulating invocation result.
        lp: The per-session loaded processor.
        raw: The ``validate`` return value (already known non-None).
        script_ref: The processor's script path, for attribution.
    """
    proc = lp.processor
    classified = _classify_validate_return(raw)
    if classified is None:
        _bucket(result, proc, (
            f"completion_processor {script_ref!r} validate "
            f"returned {type(raw).__name__}; expected "
            f"list[str] or ProcessorResult TypedDict "
            f"({{'errors': [...], 'warnings': [...], "
            f"'incomplete': [...], 'faults': [...]}})"
        ))
        return
    errors, warnings, incomplete, faults = classified
    _record_errors(result, lp, errors, script_ref)
    _record_faults(result, lp, faults, script_ref)
    for msg in _labelled(warnings, script_ref, "warning"):
        result.warned.append((proc, msg))
    for msg in _labelled(incomplete, script_ref, "incomplete"):
        result.incomplete.append((proc, msg))


def invoke_processors(
    loaded: List[LoadedProcessor],
    payload: Dict[str, Any],
    context: "RenderContext",
    phase_filter: Optional[str] = None,
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

        # Phase gate (server 0.6.199+): when a ``phase_filter`` is
        # supplied, only run processors whose declared phase matches.
        # ``prepare_completion`` passes ``"completeness"`` to run the
        # semantic-done-ness gate; ``signal_completion`` passes
        # ``"finalization"`` so completeness processors don't re-run at
        # finalize.  ``None`` (the default) runs everything — preserves
        # the pre-phase call shape for any caller that doesn't filter.
        if phase_filter is not None:
            proc_phase = getattr(proc, "phase", "finalization")
            if proc_phase != phase_filter:
                continue

        if lp.load_error is not None:
            _bucket(result, proc, lp.load_error)
            continue

        # 1. Validate first (cheaper, no side-effects)
        if lp.validate_fn is not None:
            # ``_RAISED`` rather than ``None`` as the "already reported"
            # sentinel: a ``validate`` that falls off the end RETURNS
            # ``None``, and conflating the two made that function read as
            # a pass — an error path producing the same value as success,
            # the exact class of defect issue #768 rule 5 is about.  A
            # real ``None`` return now reports a malformed return.
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
                raw = _RAISED
            if raw is not _RAISED:
                # Route the return into the four channels and apply the
                # refusal budget.  Extracted into
                # ``_record_validate_outcome`` because this function is
                # frozen in the cyclomatic-complexity baseline and new
                # logic may not grow it (see the coding policy in
                # CLAUDE.md and ``test_cyclomatic_complexity_audit``).
                _record_validate_outcome(result, lp, raw, script_ref)

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
