"""The decision-model permission gate (docs/design/decision-models.md §8.2).

A gated tool's call that the policy would otherwise hand to its
``defaultPolicy`` is put to a decision model instead, as one ``noul``
question about a state the framework renders:

- probability ``>= allow_above`` → ALLOW
- probability ``<= deny_below`` → DENY, the probability named to the model
- otherwise, or on any failure → ASK, a person decides

This module is the gate's whole definition, apart from where the plugin
calls it: the configuration block (:func:`parse_decision_gate`), the state
(:func:`render_gate_state`, pinned byte for byte by a golden test), the fit
to the model's window (:func:`fit_gate_state`) and the call
(:func:`evaluate_gate`).  It never allows on an error.

The state carries two kinds of content kept apart on purpose.  Text the
MODEL wrote (the arguments, the effect, the task) is fenced as untrusted,
because any of it can carry text addressed to the gate.  Facts the
FRAMEWORK computed (path verdicts, the boundary) are not fenced: the model
cannot forge them, and they are what tells ``rm -rf build/`` inside the
workspace from the same command outside it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Tuple

from jaato_sdk.plugins.model_provider.types import wrap_untrusted_content

logger = logging.getLogger(__name__)

#: The question asked when the block names none.
DEFAULT_QUESTION = (
    "Is this action safe to run without a human approving it first?"
)

#: The id of the gate's one question on the decision wire.
GATE_QUESTION_ID = "safe"

#: ``tools: default`` gates every tool.
TOOLS_ALL = "default"

#: Keys a ``decision_gate`` block may carry.
GATE_KEYS = frozenset({"model", "tools", "question", "allow_above", "deny_below"})

#: A string value longer than this is cut to its head and tail.
VALUE_CUT_OVER = 2000
#: What the default cut keeps: the head and tail of a value.
VALUE_HEAD = 1500
VALUE_TAIL = 400
#: Fitting the window never cuts a value below this.
VALUE_FLOOR = 200

#: Characters per token, the estimate ``decide()`` itself uses.
CHARS_PER_TOKEN = 4
#: Headroom for the request's JSON envelope beyond state and question.
WINDOW_MARGIN_TOKENS = 256

#: At most this many path lines; the rest are counted.
MAX_PATHS = 32

#: An argument key that names a path: ``path``, ``file_path``, ``filePath``,
#: ``paths``, ``source``, ``destination``, ``directory``, ``target_dir``, …
_PATH_KEY = re.compile(
    r"(?:^|_|(?<=[a-z]))(?:path|paths|file|files|filename|dir|dirs|"
    r"directory|directories|source|destination|dest|target|cwd)$",
    re.IGNORECASE,
)

#: What the effect section says when the tool's plugin renders nothing.
NO_EFFECT = "(not rendered by the tool's plugin)"
NO_PATHS = "(none named)"
NO_WORKSPACE = "(no workspace boundary)"
NO_TASK = "(none)"
UNANALYZABLE = "(the command could not be analyzed)"

#: Verdict words for a path line.
INSIDE = "inside the workspace"
OUTSIDE = "outside the workspace"
ALLOWED_OUTSIDE = "allowed outside the workspace"

#: Boundary values.
BOUNDARY_UNCONFINED = "unconfined"


class DecisionGateConfigError(ValueError):
    """A ``decision_gate`` block that cannot be used; names the key."""


@dataclass(frozen=True)
class DecisionGateConfig:
    """One parsed ``plugin_configs.permission.decision_gate`` block.

    Attributes:
        model: The ``decision_models:`` entry the gate asks.
        tools: The gated tool names, or ``None`` for every tool
            (``tools: default``).
        question: The ``noul`` question asked about each call.
        allow_above: ALLOW at or above this probability.
        deny_below: DENY at or below this probability.
    """

    model: str
    tools: Optional[FrozenSet[str]]
    question: str
    allow_above: float
    deny_below: float

    def gates(self, tool_name: str) -> bool:
        """Whether a call to ``tool_name`` goes to the gate."""
        return self.tools is None or tool_name in self.tools


def _threshold(key: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionGateConfigError(
            f"decision_gate.{key} must be a number in [0, 1], got {value!r}")
    if not 0.0 <= float(value) <= 1.0:
        raise DecisionGateConfigError(
            f"decision_gate.{key} must be in [0, 1], got {value!r}")
    return float(value)


def _tools(value: Any) -> Optional[FrozenSet[str]]:
    if value == TOOLS_ALL:
        return None
    if (not isinstance(value, list) or not value
            or not all(isinstance(t, str) and t for t in value)):
        raise DecisionGateConfigError(
            "decision_gate.tools must be a non-empty list of tool names "
            f"or {TOOLS_ALL!r}, got {value!r}")
    return frozenset(value)


def parse_decision_gate(raw: Any) -> Optional[DecisionGateConfig]:
    """Normalise a ``decision_gate`` block.

    Args:
        raw: The block as written; ``None`` means no gate.

    Returns:
        The parsed gate, or ``None`` when no block is declared.

    Raises:
        DecisionGateConfigError: The block is not an object, carries an
            unknown key, misses ``model`` / ``tools`` / a threshold, or its
            thresholds are not ``0 <= deny_below < allow_above <= 1``.
            There are no default thresholds: what probability is safe
            enough is the deployment's call (§9).
    """
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise DecisionGateConfigError(
            f"decision_gate must be an object, got {type(raw).__name__}")
    unknown = sorted(set(raw) - GATE_KEYS)
    if unknown:
        raise DecisionGateConfigError(
            f"decision_gate has unknown key(s) {unknown}; it carries "
            f"{sorted(GATE_KEYS)}")
    for key in ("model", "tools", "allow_above", "deny_below"):
        if key not in raw:
            raise DecisionGateConfigError(f"decision_gate is missing {key!r}")
    model = raw["model"]
    if not isinstance(model, str) or not model.strip():
        raise DecisionGateConfigError(
            f"decision_gate.model must name a decision_models: entry, got {model!r}")
    question = raw.get("question", DEFAULT_QUESTION)
    if not isinstance(question, str) or not question.strip():
        raise DecisionGateConfigError(
            f"decision_gate.question must be non-empty text, got {question!r}")
    allow_above = _threshold("allow_above", raw["allow_above"])
    deny_below = _threshold("deny_below", raw["deny_below"])
    if not deny_below < allow_above:
        raise DecisionGateConfigError(
            f"decision_gate.deny_below ({deny_below}) must be below "
            f"allow_above ({allow_above}); the range between them asks")
    return DecisionGateConfig(
        model=model.strip(), tools=_tools(raw["tools"]),
        question=question.strip(), allow_above=allow_above,
        deny_below=deny_below)


# ---------------------------------------------------------------- the state


@dataclass(frozen=True)
class PathFact:
    """One path a call names, as the framework judged it.

    Attributes:
        path: As shown: relative to the workspace when inside it, the
            resolved absolute path otherwise.
        mode: ``read`` or ``write``.
        verdict: :data:`INSIDE`, :data:`OUTSIDE` or :data:`ALLOWED_OUTSIDE`.
    """

    path: str
    mode: str
    verdict: str


@dataclass(frozen=True)
class GateInputs:
    """Everything :func:`render_gate_state` renders, already collected.

    Attributes:
        tool_name: The tool's real name.
        args: The call's arguments.
        effect: What the call will change, as the tool's plugin renders it
            for a person (summary, warnings, details), or ``None``.
        paths: The path facts; ``None`` when the session has no workspace
            boundary.
        paths_unanalyzable: The command's paths could not be analyzed.
        boundary: What a mistake can reach (:func:`session_boundary`).
        task: The session's first user message, or ``None``.
    """

    tool_name: str
    args: Dict[str, Any]
    effect: Optional[str]
    paths: Optional[Tuple[PathFact, ...]]
    boundary: str
    task: Optional[str]
    paths_unanalyzable: bool = False


def cut_value(text: str, cap: Optional[int] = None) -> str:
    """Cut a long string to its head and tail.

    With no ``cap`` the rule is the state's default: over
    :data:`VALUE_CUT_OVER` characters, keep the first :data:`VALUE_HEAD`
    and the last :data:`VALUE_TAIL`.  With a ``cap`` (fitting the window),
    keep ``cap`` characters in the same proportion.
    """
    if cap is None:
        if len(text) <= VALUE_CUT_OVER:
            return text
        head, tail = VALUE_HEAD, VALUE_TAIL
    else:
        if len(text) <= cap:
            return text
        head = cap * VALUE_HEAD // (VALUE_HEAD + VALUE_TAIL)
        tail = cap - head
    omitted = len(text) - head - tail
    return f"{text[:head]}… [{omitted} characters omitted] …{text[-tail:]}"


def _cut(text: str, key: str, caps: Mapping[str, int]) -> str:
    return cut_value(cut_value(text), caps.get(key))


def _cut_args(obj: Any, key: str, caps: Mapping[str, int]) -> Any:
    if isinstance(obj, str):
        return _cut(obj, key, caps)
    if isinstance(obj, dict):
        return {k: _cut_args(v, f"{key}/{k}", caps) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_cut_args(v, f"{key}/{i}", caps) for i, v in enumerate(obj)]
    return obj


def _string_lengths(obj: Any, key: str, out: Dict[str, int]) -> None:
    if isinstance(obj, str):
        out[key] = len(cut_value(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _string_lengths(v, f"{key}/{k}", out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _string_lengths(v, f"{key}/{i}", out)


def _path_lines(inputs: GateInputs) -> List[str]:
    if inputs.paths is None:
        return [NO_WORKSPACE]
    lines = [f"- {json.dumps(p.path, ensure_ascii=False)} {p.mode}, {p.verdict}"
             for p in inputs.paths[:MAX_PATHS]]
    extra = len(inputs.paths) - MAX_PATHS
    if extra > 0:
        lines.append(f"- … and {extra} more")
    if inputs.paths_unanalyzable:
        lines.append(UNANALYZABLE)
    return lines or [NO_PATHS]


def render_gate_state(
    inputs: GateInputs,
    *,
    redact: Optional[Callable[[Any], Any]] = None,
    caps: Optional[Mapping[str, int]] = None,
) -> str:
    """The text the decision model reads about one call (§8.2.1).

    Six sections, always present, in this order: ``Tool:``,
    ``Arguments:``, ``Effect:``, ``Paths:``, ``Boundary:``, ``Task:``.
    Arguments, effect and task are fenced as untrusted; paths and the
    boundary are framework facts and are not.  Long strings are cut by
    :func:`cut_value`; ``caps`` cuts named values further when fitting the
    window (keys: ``args/<key>/…``, ``effect``, ``task``).

    Args:
        inputs: What to render.
        redact: Applied to the arguments, the effect, the task and each
            path before anything is cut (the secret redactor, #1215).
        caps: Per-value character caps from :func:`fit_gate_state`.
    """
    caps = caps or {}
    red = redact or (lambda obj: obj)
    args = _cut_args(red(inputs.args), "args", caps)
    arguments = json.dumps(args, indent=2, sort_keys=True,
                           ensure_ascii=False, default=str)
    effect = (_cut(red(inputs.effect), "effect", caps)
              if inputs.effect else NO_EFFECT)
    task = _cut(red(inputs.task), "task", caps) if inputs.task else NO_TASK
    redacted_paths = inputs
    if inputs.paths is not None:
        redacted_paths = GateInputs(
            **{**inputs.__dict__, "paths": tuple(
                PathFact(red(p.path), p.mode, p.verdict) for p in inputs.paths)})
    lines = [
        f"Tool: {inputs.tool_name}",
        "Arguments:",
        wrap_untrusted_content(arguments, source="tool-arguments"),
        "Effect:",
        wrap_untrusted_content(effect, source="tool-effect"),
        "Paths:",
        *_path_lines(redacted_paths),
        f"Boundary: {inputs.boundary}",
        "Task:",
        wrap_untrusted_content(task, source="session-task"),
    ]
    return "\n".join(lines) + "\n"


def estimate_tokens(text: str) -> int:
    """Tokens at :data:`CHARS_PER_TOKEN`, the estimate ``decide()`` uses."""
    return len(text) // CHARS_PER_TOKEN


def fit_gate_state(
    inputs: GateInputs,
    question: str,
    window_tokens: int,
    *,
    redact: Optional[Callable[[Any], Any]] = None,
) -> Optional[str]:
    """The rendered state, cut until it fits the window, or ``None``.

    After the per-value cuts, the longest remaining string is cut to
    three quarters of its length, repeatedly, never below
    :data:`VALUE_FLOOR`.  ``None`` means it still does not fit: the gate
    then asks without calling the model.  A ``window_tokens`` of 0 (the
    window is unknown) is not checked; ``decide()``'s own check and the
    endpoint's refusal are then the answer.
    """
    caps: Dict[str, int] = {}
    budget = window_tokens - WINDOW_MARGIN_TOKENS - estimate_tokens(question)
    red = redact or (lambda obj: obj)
    while True:
        state = render_gate_state(inputs, redact=redact, caps=caps)
        if not window_tokens or estimate_tokens(state) <= budget:
            return state
        lengths: Dict[str, int] = {}
        _string_lengths(red(inputs.args), "args", lengths)
        if inputs.effect:
            lengths["effect"] = len(cut_value(red(inputs.effect)))
        if inputs.task:
            lengths["task"] = len(cut_value(red(inputs.task)))
        current = {k: min(v, caps.get(k, v)) for k, v in lengths.items()}
        cuttable = {k: v for k, v in current.items() if v > VALUE_FLOOR}
        if not cuttable:
            return None
        key = max(sorted(cuttable), key=lambda k: cuttable[k])
        caps[key] = max(VALUE_FLOOR, cuttable[key] * 3 // 4)


def state_digest(state: str) -> str:
    """The SHA-256 the DECISION trace line records instead of the state."""
    return hashlib.sha256(state.encode("utf-8")).hexdigest()


# ------------------------------------------------- collecting the inputs


def compose_effect(display_info: Any) -> Optional[str]:
    """The effect section's text from a ``PermissionDisplayInfo``."""
    if display_info is None:
        return None
    parts: List[str] = []
    summary = getattr(display_info, "summary", None)
    if summary:
        parts.append(str(summary))
    warnings = getattr(display_info, "warnings", None)
    if warnings:
        parts.append(f"Warnings: {warnings}")
    details = getattr(display_info, "details", None)
    if details and details != summary:
        parts.append(str(details))
    return "\n".join(parts) or None


def _arg_paths(obj: Any, key: str, depth: int, out: List[Tuple[str, str]]) -> None:
    """Path strings in ``obj`` whose key names a path, one level deep."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _arg_paths(v, str(k), depth, out)
        return
    if isinstance(obj, (list, tuple)):
        if depth > 1:
            return
        for v in obj:
            _arg_paths(v, key, depth + 1, out)
        return
    if isinstance(obj, str) and obj and _PATH_KEY.search(key):
        out.append((obj, "write"))


def _gate_path_word(word: str, writes: bool) -> bool:
    """Whether a shell word names a path, for the gate's purposes.

    Looser than containment's ``path_like``, which skips a bare relative
    name because it cannot leave the workspace.  The gate needs to know
    what the command acts on, so every non-flag argument of a write
    command counts, and so does any word containing a ``/``.
    """
    from ..command_containment import path_like
    if not word or word.startswith("-") or re.match(r"^[a-zA-Z][\w+.-]*://", word):
        return False
    return path_like(word) or writes or "/" in word


def command_paths(command: str) -> List[Tuple[str, str]]:
    """The ``(path, mode)`` pairs of a shell command, per segment.

    Uses the same segmentation and write heuristics as ``cli``'s
    containment (``analyze_command``, ``classify_word_paths``), with the
    looser word rule of :func:`_gate_path_word`.  Raises
    ``UnanalyzableCommand`` when the command cannot be modelled.
    """
    from ..command_containment import (
        ALL_WRITE_CMDS, classify_word_paths, effective_command_name,
    )
    from ...command_analysis import analyze_command, executable_word_index

    out: List[Tuple[str, str]] = []
    for segment in analyze_command(command):
        name = effective_command_name(segment)
        exe = executable_word_index(segment.words)
        writes = name in ALL_WRITE_CMDS
        words = [w for i, w in enumerate(segment.words)
                 if i != exe and _gate_path_word(w, writes)]
        out.extend(classify_word_paths(name, words))
        out.extend((r.target, r.mode) for r in segment.redirects
                   if r.mode != "none" and r.target)
    return out


def named_paths(args: Mapping[str, Any]) -> Tuple[List[Tuple[str, str]], bool]:
    """The ``(path, mode)`` pairs a call names, and whether analysis failed.

    A ``command`` string argument goes through :func:`command_paths`;
    every other argument whose key names a path is taken as written to
    (the conservative mode).
    """
    from ...command_analysis import UnanalyzableCommand

    pairs: List[Tuple[str, str]] = []
    unanalyzable = False
    command = args.get("command")
    if isinstance(command, str) and command.strip():
        try:
            pairs.extend(command_paths(command))
        except UnanalyzableCommand:
            unanalyzable = True
    rest = {k: v for k, v in args.items() if k != "command"}
    _arg_paths(rest, "", 0, pairs)
    seen: Dict[str, int] = {}
    out: List[Tuple[str, str]] = []
    for path, mode in pairs:
        if path in seen:
            if mode == "write":
                out[seen[path]] = (path, "write")
            continue
        seen[path] = len(out)
        out.append((path, "write" if mode == "write" else "read"))
    return out, unanalyzable


def judge_paths(
    pairs: List[Tuple[str, str]],
    workspace_root: Optional[str],
    registry: Any = None,
) -> Optional[Tuple[PathFact, ...]]:
    """Path facts from ``pairs``; ``None`` when there is no workspace."""
    if not workspace_root:
        return None
    from ..command_containment import path_within_workspace

    root = os.path.realpath(workspace_root)
    facts: List[PathFact] = []
    for path, mode in pairs:
        expanded = os.path.expanduser(path)
        resolved = os.path.realpath(
            expanded if os.path.isabs(expanded)
            else os.path.join(root, expanded))
        if resolved == root or resolved.startswith(root + os.sep):
            shown = os.path.relpath(resolved, root)
            if path.endswith("/") and not shown.endswith("/"):
                shown += "/"
            facts.append(PathFact(shown, mode, INSIDE))
            continue
        try:
            allowed = path_within_workspace(
                path, workspace_root, registry, mode=mode)
        except Exception:  # noqa: BLE001
            allowed = False
        facts.append(PathFact(resolved, mode,
                              ALLOWED_OUTSIDE if allowed else OUTSIDE))
    return tuple(facts)


def session_boundary() -> str:
    """What a mistake in this process can reach, as the state names it.

    The runner's own AppArmor label (the calling thread's, #1023), else the
    SELinux boundary the bootstrap recorded, else ``unconfined``.  A label
    that cannot be read claims no boundary.
    """
    try:
        from ...apparmor_label import read_thread_label
        label = read_thread_label()
        if label.confined and label.mode:
            return f"apparmor ({label.mode})"
    except Exception:  # noqa: BLE001
        pass
    try:
        from ...lsm_label import selinux_session_boundary
        boundary = selinux_session_boundary()
    except Exception:  # noqa: BLE001
        boundary = None
    if boundary is not None:
        return ("selinux (enforcing)" if boundary.enforcing_attested
                else "selinux (not attested)")
    return BOUNDARY_UNCONFINED


def session_task(session: Any) -> Optional[str]:
    """The session's first user message (a subagent's spawn task)."""
    get_history = getattr(session, "get_history", None)
    if not callable(get_history):
        return None
    try:
        history = get_history()
    except Exception:  # noqa: BLE001
        return None
    for message in history or ():
        role = getattr(message, "role", None)
        if getattr(role, "value", role) != "user":
            continue
        texts = [p.text for p in getattr(message, "parts", None) or ()
                 if getattr(p, "text", None)]
        if texts:
            return "\n".join(texts)
    return None


# ---------------------------------------------------------------- the call


@dataclass(frozen=True)
class GateVerdict:
    """What the gate decided about one call.

    Attributes:
        decision: ``allow``, ``deny`` or ``ask``.
        reason: Why, in words the trace and the model read.
        model: The ``decision_models:`` entry asked.
        probability: The ``noul`` probability, ``None`` when the model was
            not asked or did not answer.
        state_sha256: The digest of the state sent, ``None`` when none was.
    """

    decision: str
    reason: str
    model: str
    probability: Optional[float] = None
    state_sha256: Optional[str] = None

    def trace_fields(self) -> Dict[str, str]:
        """The fields the DECISION trace line carries for this verdict."""
        out = {"gate_model": self.model}
        if self.probability is not None:
            out["gate_probability"] = f"{self.probability:.4f}"
        if self.state_sha256:
            out["gate_state_sha256"] = self.state_sha256
        return out


def _ask(config: DecisionGateConfig, reason: str, **kw: Any) -> GateVerdict:
    return GateVerdict("ask", f"decision_gate: {reason}", config.model, **kw)


def _record_spend(session: Any, config: DecisionGateConfig, result: Any) -> None:
    """A ``decision`` record in the session's token ledger, best effort.

    The same record type the ``decide`` tool writes, with ``via:
    decision_gate``: a gate call is spend the chat accounting never sees.
    """
    record = getattr(getattr(getattr(session, "_runtime", None), "ledger", None),
                     "_record", None)
    if not callable(record):
        return
    usage = getattr(result, "usage", None)
    reported = usage is not None and getattr(usage, "reported", True)
    try:
        record("decision", {
            "entry": config.model,
            "model": getattr(result, "model", None),
            "questions": 1,
            "via": "decision_gate",
            "usage": {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.output_tokens,
                **({"cost_usd": usage.cost_usd}
                   if getattr(usage, "cost_usd", None) is not None else {}),
            } if reported else None,
        })
    except Exception as exc:  # noqa: BLE001
        logger.debug("decision_gate: ledger record failed: %s", exc)


def evaluate_gate(
    config: DecisionGateConfig,
    session: Any,
    inputs: GateInputs,
    *,
    redact: Optional[Callable[[Any], Any]] = None,
    cancel_token: Any = None,
) -> GateVerdict:
    """Ask the gate's decision model about one call.

    Asks (never allows) when there is no session, the entry is unknown or
    cannot serve ``decide()``, the state does not fit the window, the call
    fails, or the answer is missing or not a probability.
    """
    from jaato_sdk.plugins.model_provider.decisions import DecisionQuestion

    get_provider = getattr(session, "decision_provider", None)
    if not callable(get_provider):
        return _ask(config, "no session to ask from")
    try:
        provider = get_provider(config.model)
    except Exception as exc:  # noqa: BLE001
        return _ask(config, f"decision model unavailable: {exc}")
    try:
        window = int(provider.get_context_limit() or 0)
    except Exception:  # noqa: BLE001
        window = 0
    state = fit_gate_state(inputs, config.question, window, redact=redact)
    if state is None:
        return _ask(config, "state over window")
    digest = state_digest(state)
    question = DecisionQuestion(type="noul", instructions=config.question)
    try:
        result = provider.decide(
            state, {GATE_QUESTION_ID: question}, cancel_token=cancel_token)
    except Exception as exc:  # noqa: BLE001
        logger.warning("decision_gate: %s failed: %s", config.model, exc)
        return _ask(config, f"decision call failed: {exc}",
                    state_sha256=digest)
    _record_spend(session, config, result)
    answer = (getattr(result, "answers", None) or {}).get(GATE_QUESTION_ID)
    value = getattr(answer, "value", None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return _ask(config, "no probability in the answer",
                    state_sha256=digest)
    probability = float(value)
    if probability >= config.allow_above:
        return GateVerdict(
            "allow",
            f"decision_gate: {config.model} answered {probability:.2f} "
            f"(allow at >= {config.allow_above})",
            config.model, probability, digest)
    if probability <= config.deny_below:
        return GateVerdict(
            "deny",
            f"decision_gate: {config.model} answered {probability:.2f} that "
            f"this call is safe to run unattended (deny at <= "
            f"{config.deny_below}); not run",
            config.model, probability, digest)
    return _ask(
        config,
        f"{config.model} answered {probability:.2f}, between "
        f"{config.deny_below} and {config.allow_above}",
        probability=probability, state_sha256=digest)
