"""The ``decide`` tool (docs/design/decision-models.md §8.1).

The agent hands a decision model a state (a ticket, a row, a diff) and
typed questions about it, and gets typed answers back with calibrated
probabilities.  The tool names a ``decision_models:`` entry of the CALLING
session's profile; the session builds and caches that entry's provider
(``JaatoSession.decision_provider``), so this plugin holds no provider and
no per-session state.

Per session, never per instance: the plugin instance is shared by a
parent and its in-process subagents, whose profiles may bind different
entries.  So both the visibility predicate and the ``model`` enum read the
current session at call time.

Not auto-approved: a call spends money and sends workspace content to a
third party.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any, Callable, Dict, List, Optional, Tuple

from jaato_sdk.plugins.model_provider.decisions import (
    DecisionError,
    DecisionQuestion,
)
from jaato_sdk.plugins.model_provider.types import (
    DISCOVERABILITY_DEFERRED,
    ToolSchema,
)

from ...trace import trace as _trace_write

TOOL_NAME = "decide"

#: How many questions one call may carry.  Each is answered by the model in
#: one request, and a call is meant to be one decision about one state.
MAX_QUESTIONS = 32

_QUESTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {
            "type": "string",
            "description": "Your id for the question; the answer comes back "
            "under it.",
        },
        "type": {
            "type": "string",
            "enum": ["noul", "choice", "score"],
            "description": "noul: a yes/no probability.  choice: one of "
            "`options`.  score: a level on the ordered scale `levels`.",
        },
        "instructions": {
            "type": "string",
            "description": "The question, asked about the state.",
        },
        "options": {
            "type": "array",
            "description": "choice only: the options, 1 to 255.",
            "items": {
                "type": "object",
                "properties": {
                    "option": {"type": "string"},
                    "description": {
                        "type": "string",
                        "description": "What choosing it means (optional).",
                    },
                },
                "required": ["option"],
            },
        },
        "levels": {
            "type": "array",
            "items": {"type": "string"},
            "description": "score only: 2 to 10 levels, lowest first.",
        },
        "true_means": {
            "type": "string",
            "description": "noul only, optional: what a true answer means.",
        },
        "false_means": {
            "type": "string",
            "description": "noul only, optional: what a false answer means.",
        },
    },
    "required": ["id", "type", "instructions"],
}


def _trace(msg: str) -> None:
    _trace_write("DECISION", msg)


def _current_session() -> Any:
    """The session the call is serving, or ``None`` outside one."""
    try:
        from ...session_context import get_current_session
        return get_current_session()
    except LookupError:
        return None


def _bound_names(session: Any) -> Optional[List[str]]:
    """The session's ``decision_models:`` entry names, or ``None`` when the
    session cannot say (no session, or an object without the method)."""
    names = getattr(session, "decision_model_names", None)
    if not callable(names):
        return None
    try:
        return list(names())
    except Exception:  # noqa: BLE001
        return None


def _criteria(qid: str, q: Dict[str, Any]) -> Any:
    """The wire ``criteria`` for one question, from the tool's typed fields."""
    qtype = q.get("type")
    if qtype == "choice":
        options = q.get("options")
        if not isinstance(options, list):
            raise ValueError(f"question {qid!r}: choice needs `options`")
        out: Dict[str, Any] = {}
        for item in options:
            name = item.get("option") if isinstance(item, dict) else item
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"question {qid!r}: every option needs a name")
            out[name] = item.get("description") if isinstance(item, dict) else None
        return out
    if qtype == "score":
        levels = q.get("levels")
        if not isinstance(levels, list):
            raise ValueError(f"question {qid!r}: score needs `levels`")
        return list(levels)
    true_means, false_means = q.get("true_means"), q.get("false_means")
    if true_means is None and false_means is None:
        return None
    return {"true": true_means, "false": false_means}


def build_questions(raw: Any) -> Dict[str, DecisionQuestion]:
    """The tool's ``questions`` argument as ``{id: DecisionQuestion}``.

    Raises ``ValueError`` naming the question for a missing or repeated
    id, an unknown type, or a type's field missing.  The criteria shapes
    themselves (option and level counts) are checked by the provider's
    ``build_decision_request`` before anything is sent.
    """
    if not isinstance(raw, list) or not raw:
        raise ValueError("`questions` must be a non-empty array")
    if len(raw) > MAX_QUESTIONS:
        raise ValueError(
            f"{len(raw)} questions in one call; at most {MAX_QUESTIONS}")
    out: Dict[str, DecisionQuestion] = {}
    for q in raw:
        if not isinstance(q, dict):
            raise ValueError("every question must be an object")
        qid = q.get("id")
        if not isinstance(qid, str) or not qid:
            raise ValueError("every question needs a non-empty `id`")
        if qid in out:
            raise ValueError(f"question id {qid!r} is used twice")
        if q.get("type") not in ("noul", "choice", "score"):
            raise ValueError(
                f"question {qid!r}: type must be noul, choice or score")
        out[qid] = DecisionQuestion(
            type=q["type"], instructions=q.get("instructions"),
            criteria=_criteria(qid, q))
    return out


def _answer_dict(answer: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"type": answer.type, "value": answer.value}
    if answer.probabilities is not None:
        out["probabilities"] = dict(answer.probabilities)
    if answer.confidence is not None:
        out["confidence"] = answer.confidence
    if answer.legend:
        out["legend"] = dict(answer.legend)
    return out


def _usage_dict(usage: Any) -> Optional[Dict[str, Any]]:
    """What the call cost, as reported; ``None`` when nothing was reported
    (absent is not zero, #688)."""
    if usage is None or not getattr(usage, "reported", True):
        return None
    out: Dict[str, Any] = {
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.output_tokens,
    }
    cost = getattr(usage, "cost_usd", None)
    if cost is not None:
        out["cost_usd"] = cost
    return out


class DecisionPlugin:
    """Provides ``decide``; holds no provider and no session state."""

    @property
    def name(self) -> str:
        return "decision"

    def __init__(self) -> None:
        self._agent_name: Optional[str] = None

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._agent_name = (config or {}).get("agent_name")

    def shutdown(self) -> None:
        pass

    def reset_for_next_session(self) -> None:
        """Nothing to clear: the providers belong to each session."""

    def get_tool_schemas(self) -> List[ToolSchema]:
        return [ToolSchema(
            name=TOOL_NAME,
            description=(
                "Ask a decision model typed questions about a state and get "
                "calibrated answers: noul (probability true), choice (one of "
                "your options, with a probability for each) or score (a "
                "level on your scale).  Cheap and fast; use it to classify "
                "or triage many items, one call per item.  `model` names one "
                "of the decision models this session's profile binds.  The "
                "state is sent to the decision model's provider."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "A decision model bound in this "
                        "session's profile (decision_models:).",
                    },
                    "state": {
                        "type": "string",
                        "description": "What the questions are about: text, "
                        "or JSON for structured data.",
                    },
                    "questions": {
                        "type": "array",
                        "items": _QUESTION_SCHEMA,
                        "description": f"1 to {MAX_QUESTIONS} questions.",
                    },
                },
                "required": ["model", "state", "questions"],
            },
            category="web",
            discoverability=DISCOVERABILITY_DEFERRED,
        )]

    def is_tool_visible(self, tool_name: str) -> bool:
        """``decide`` is hidden from a session that binds no decision model."""
        if tool_name != TOOL_NAME:
            return True
        names = _bound_names(_current_session())
        return names is None or bool(names)

    def narrow_tool_schema(self, schema: ToolSchema) -> ToolSchema:
        """Restrict ``model`` to the calling session's entry names."""
        if getattr(schema, "name", None) != TOOL_NAME:
            return schema
        names = _bound_names(_current_session())
        if not names:
            return schema
        params = copy.deepcopy(schema.parameters)
        params["properties"]["model"]["enum"] = sorted(names)
        return dataclasses.replace(schema, parameters=params)

    def get_executors(self) -> Dict[str, Callable[..., Any]]:
        return {TOOL_NAME: self._execute_decide}

    def get_system_instructions(self) -> Optional[str]:
        return None

    def get_auto_approved_tools(self) -> List[str]:
        return []

    def get_user_commands(self) -> List[Any]:
        return []

    def _execute_decide(self, args: Dict[str, Any]) -> Any:
        """Run one decision call; failures are explicit ``(False, payload)``."""
        session = _current_session()
        if session is None or not callable(
                getattr(session, "decision_provider", None)):
            return False, {"error": "decide runs only inside a session"}
        name = args.get("model")
        state = args.get("state")
        if not isinstance(state, (str, dict, list)) or state == "":
            return False, {"error": "`state` must be non-empty text or JSON"}
        try:
            questions = build_questions(args.get("questions"))
            # The entry's provider is its own instance, connected to the
            # entry's model, so decide() needs no model= of its own.
            provider = session.decision_provider(str(name))
            from ...ai_tool_runner import get_current_cancel_token
            result = provider.decide(
                state, questions, cancel_token=get_current_cancel_token())
        except (ValueError, LookupError, DecisionError, RuntimeError) as exc:
            _trace(f"decide: model={name!r} refused: {exc}")
            return False, {"error": str(exc), "model": name}
        payload, usage = _result_payload(name, result)
        _record_in_ledger(session, name, result, usage)
        _trace(
            f"decide: model={name!r} answered_by={result.model!r} "
            f"questions={len(questions)} usage={usage}")
        return payload


def _result_payload(name: Any, result: Any) -> Tuple[Dict[str, Any], Any]:
    usage = _usage_dict(result.usage)
    payload: Dict[str, Any] = {
        "model": name,
        "answered_by": result.model,
        "answers": {
            qid: _answer_dict(a) for qid, a in result.answers.items()},
    }
    if usage is not None:
        payload["usage"] = usage
    return payload, usage


def _record_in_ledger(session: Any, name: Any, result: Any, usage: Any) -> None:
    """A ``decision`` record in the session's token ledger, best effort.

    The ledger is where a deployment reads what a session spent; a decision
    call is spend the chat accounting never sees.
    """
    runtime = getattr(session, "_runtime", None)
    ledger = getattr(runtime, "ledger", None)
    record = getattr(ledger, "_record", None)
    if not callable(record):
        return
    try:
        record("decision", {
            "entry": name,
            "model": result.model,
            "questions": len(result.answers),
            "usage": usage,
        })
    except Exception as exc:  # noqa: BLE001
        _trace(f"decide: ledger record failed: {exc}")


def create_plugin() -> DecisionPlugin:
    return DecisionPlugin()
