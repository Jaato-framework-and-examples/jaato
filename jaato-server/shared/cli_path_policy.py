"""``extra_paths`` is the operator's knob, never the model's (issue #697).

Extending ``PATH`` decides **which binary a command name resolves to**.  That
makes it a different kind of argument from ``command`` and ``args``: those say
what to run, this one says *where to find it*, and only the second is invisible
in the text a human approves.

Three facts made that dangerous, and they only mattered together:

1. **The permission signature is the command text alone.**
   ``PermissionPolicy._build_signature`` reads ``command`` and ``args`` and
   nothing else, so ``{"command": "deploy --prod"}`` and
   ``{"command": "deploy --prod", "extra_paths": ["/tmp/x/bin"]}`` produce the
   same signature, match the same whitelist pattern, and emit DECISION trace
   lines (#951/#968) that differ only in ``call_id``.
2. **Every execution site read ``extra_paths`` straight out of the model's
   tool-call arguments** — ``CliPlugin._execute`` / ``_execute_streaming``
   (where the operator's configured list was merely the *fallback*) and
   ``server.runner.cli_runner.execute_cli_based_tool`` (where there was no
   operator tier at all).
3. **It reached ``shutil.which``**, so the approved string and the executed
   binary were decided by two different inputs, and the model controlled the
   second one.

So an approval for ``deploy --prod`` was in force an approval for whatever
``deploy`` resolved to under a ``PATH`` the model supplied *in the same call*.

**The answer is (a), not (b).**  Issue #697 poses two separable questions:
whether a model-supplied path component may change binary resolution at all,
and — if it may — whether it must sit inside the value the permission decision
is keyed on.  This module answers the first with *no*, which is the issue's own
item 5, and that is what makes the second moot: a value the executor refuses
cannot vary from approval to execution, so keying the signature on it would
only prompt a human to authorize a call that is going to be refused anyway.
Nothing in the tool's purpose requires the *model* to extend ``PATH`` per call;
``plugin_configs.cli.extra_paths`` is, and remains, how a deployment adds
``/opt/toolchain/bin``.

**Refused, not silently dropped.**  Dropping the key would surface as
``executable 'x' not found in PATH`` — a message that sends the model (and a
reader of the trace) looking in entirely the wrong place.  A named refusal says
what happened and who owns the knob, and it is recorded at WARNING because a
model attempting to widen its own executable search path is worth an operator's
attention even though it failed.

**Ordering is a security property.**  Where the operator's own ``extra_paths``
are applied they are *appended* to ``PATH``, never prepended, so a configured
directory cannot shadow a system binary; see :data:`APPEND_ORDER_RATIONALE`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


#: The tool-call argument that must never be honoured from model-supplied args.
MODEL_REFUSED_ARG = "extra_paths"

#: Reason text for the refusal.  Addressed to the model, because the model is
#: what reads a tool result: it names the offending key, says the call was not
#: run, and points at the tier that owns the knob so the next attempt is a
#: request to the operator rather than a retry of the same call.
REFUSAL_ERROR = (
    "cli_based_tool: 'extra_paths' is operator-configured and cannot be set "
    "per call. Extending PATH changes which binary a command name resolves "
    "to, and that change is invisible to the permission decision made over "
    "the command text, so a caller-supplied value is refused rather than "
    "applied (issue #697). The command was NOT executed."
)

#: Actionable next step, kept separate so it rides the same ``hint`` field the
#: executable-not-found path already uses.
REFUSAL_HINT = (
    "Invoke the executable by absolute path, or ask the operator to add the "
    "directory to plugin_configs.cli.extra_paths."
)

#: The other pre-spawn refusal, unchanged in wording since before #697 — the
#: two are returned through one check so a site has a single "should this run
#: at all?" question rather than a growing ladder of guards.
MISSING_COMMAND_ERROR = "cli_based_tool: command must be provided"

#: Hint on the POST-spawn failure that shares this module's subject: the
#: command name did not resolve.  It used to read "Configure extra_paths or
#: provide full path to the executable", which is advice the model cannot
#: take — it is the tier that may not set the knob — so it invited exactly
#: the call this module refuses.  Named here so the two messages cannot drift
#: into contradicting each other.
CLI_EXE_NOT_FOUND_HINT = (
    "Provide the full path to the executable, or ask the operator to add its "
    "directory to plugin_configs.cli.extra_paths (PATH extension is "
    "operator-configured, not settable per call)."
)

#: Why the operator's own entries are appended rather than prepended.  Quoted
#: at each site that builds the ``PATH`` string so the ordering survives an
#: edit that reads it as mere formatting.
APPEND_ORDER_RATIONALE = (
    "extra_paths are APPENDED, never prepended: system locations are searched "
    "first, so a configured directory can add a command name that is absent "
    "from the base PATH but can never shadow an existing binary. Reversing "
    "this makes every whitelisted command shadowable."
)


def refuse_model_extra_paths(
    args: Dict[str, Any],
    *,
    surface: str,
) -> Optional[Dict[str, Any]]:
    """Refuse a cli tool call that carries a caller-supplied ``extra_paths``.

    Applied by every site that turns ``cli_based_tool`` arguments into a
    subprocess, so the three of them cannot drift apart the way they had
    (two honoured the key with an operator fallback beneath it, one honoured
    it with no operator tier at all).

    Presence is what is refused, not truth: ``{"extra_paths": []}`` is
    refused alongside ``{"extra_paths": ["/tmp/x"]}``.  An empty list is
    inert today only because of how the consuming code happens to be written,
    and "this argument has no effect when falsy" is precisely the kind of
    invariant a later refactor breaks silently.  The rule a reader has to hold
    is the simple one: the key never comes from the caller.

    Args:
        args: The tool-call argument dict as received from the model.
        surface: Human-readable name of the execution site, for the log line
            (e.g. ``"cli plugin"``, ``"runner cli executor"``).  It appears
            only in the operator-facing WARNING, never in the model's result.

    Returns:
        ``None`` when the call is clean and may proceed — the overwhelmingly
        common case.  Otherwise a ready-to-return result body carrying
        ``error`` and ``hint``, which each caller wraps in its own result
        shape (the plugin returns it directly; the runner executor pairs it
        with ``ok=False``).
    """
    if MODEL_REFUSED_ARG not in args:
        return None

    logger.warning(
        "cli: refusing caller-supplied '%s' at %s — PATH extension is "
        "operator-configured (issue #697); command not executed",
        MODEL_REFUSED_ARG,
        surface,
    )
    return {"error": REFUSAL_ERROR, "hint": REFUSAL_HINT}


def precheck_cli_args(
    args: Dict[str, Any],
    *,
    surface: str,
) -> Optional[Dict[str, Any]]:
    """Decide whether a ``cli_based_tool`` call may reach a subprocess at all.

    One question per execution site, asked before any environment is built:
    *is this call runnable as given?*  Two answers today, in the order that
    matters — a call carrying a caller-supplied ``extra_paths`` is refused
    (:func:`refuse_model_extra_paths`) before the command is even looked at,
    because that refusal is about the caller's authority rather than about
    whether the call is well-formed.

    Collapsing the two into one helper is not only tidiness: a site that asks
    this question once cannot answer half of it, which is how ``_execute``
    and ``_execute_streaming`` came to disagree about whether a pre-spawn
    refusal reports through the streaming callbacks.

    Args:
        args: The tool-call argument dict as received from the model.
        surface: Human-readable name of the execution site, for the
            operator-facing WARNING a refusal logs.

    Returns:
        ``None`` when the call may proceed.  Otherwise a ready-to-return
        result body carrying ``error`` and, for the ``extra_paths`` refusal,
        ``hint``; the caller wraps it in its own result shape and must not
        spawn anything.
    """
    refusal = refuse_model_extra_paths(args, surface=surface)
    if refusal is not None:
        return refusal
    if not args.get("command"):
        return {"error": MISSING_COMMAND_ERROR}
    return None
