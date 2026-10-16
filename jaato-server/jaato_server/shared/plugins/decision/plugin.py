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

Images (§12.5): when an entry the session binds runs on a provider that
declares ``decision_images``, the schema gains an ``images`` parameter
(each image named by a workspace ``path`` or a history ``attachment_id``,
never by bytes; see :mod:`.images`) and the ``box`` question type.  A
``box`` answer carries ``box_px``, pixel edges computed from the image's
own dimensions, and the ledger record lists each image's digest, size and
dimensions.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any, Callable, Dict, List, Optional, Tuple

from jaato_sdk.plugins.model_provider.decisions import (
    IMAGE_TOKEN_RE,
    MAX_IMAGES,
    DecisionBox,
    DecisionError,
    DecisionQuestion,
)
from jaato_sdk.plugins.model_provider.types import (
    DISCOVERABILITY_DEFERRED,
    ToolSchema,
)

from ...trace import trace as _trace_write
from .images import ImageReferenceError, ResolvedImage, image_record, resolve_images

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


#: The ``images`` parameter, offered only to a session with an entry whose
#: provider declares ``decision_images`` (see ``narrow_tool_schema``).
_IMAGES_SCHEMA: Dict[str, Any] = {
    "type": "array",
    "description": (
        f"Images the state or the questions refer to, at most {MAX_IMAGES}.  "
        "Name each image and write {{image:NAME}} in the state, an "
        "instruction or an option description where it belongs.  Give a "
        "workspace `path` or the `attachment_id` (att_...) of an image "
        "already in this conversation, never the bytes.  PNG or JPEG, "
        "280 to 1568 px a side, already upright: nothing is resized or "
        "rotated."),
    "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string",
                     "description": "The NAME in {{image:NAME}}."},
            "path": {"type": "string",
                     "description": "A workspace file (relative to the "
                     "workspace root, or absolute)."},
            "attachment_id": {"type": "string",
                              "description": "An image already in this "
                              "conversation (att_...)."},
        },
        "required": ["name"],
    },
}

#: How images and ``box`` reach a session, one rule per line.  Rendered by
#: ``jaato-scaffold explain plugin decision``, which shows only the base
#: (text-only) schema, so the shaped parts are stated here, once.
IMAGE_RULES = (
    "offered only to a session binding a decision_models: entry whose "
    "provider declares decision_images; otherwise neither `images` nor the "
    "`box` question type exists for it",
    "each image is {name, path} or {name, attachment_id}, exactly one; the "
    "model writes {{image:NAME}} in the state, an instruction or an option "
    "description; bytes never appear in the call",
    "a `path` is relative to the workspace root (or absolute) and follows "
    "readFile's rule; .env at any depth and .jaato/*_auth.json are refused",
    "an `attachment_id` (att_<16 hex>) names an image already in the "
    "session's history, a user message's or a tool result's; one evicted "
    "after its turn is refused as evicted",
    "PNG or JPEG, read from the bytes; 280-1568 px a side; never resized "
    "or rotated (EXIF orientation is not applied)",
    "a box answer is edge fractions plus `image` and `box_px` (pixels of "
    "that image): the image its instructions name, else the call's only "
    "one; with neither, no box_px",
    "a box is always returned, even when the target is absent: pair it "
    "with a noul",
    "the ledger's decision record lists each image's sha256, size and "
    "dimensions, never its bytes",
)

def images_parameter_schema() -> Dict[str, Any]:
    """A copy of the ``images`` parameter a shaped schema carries."""
    return copy.deepcopy(_IMAGES_SCHEMA)


_BOX_TYPE_TEXT = (
    "  box: where one target is in an image; the answer is edge fractions "
    "and `box_px`.  A box is ALWAYS returned, even when the target is not "
    "there, so pair it with a noul asking whether it is.")

_IMAGES_TOOL_TEXT = (
    "  With `images`, ask about pictures too (a screenshot, a photo); keep "
    "each question simple and give a box question one target.")


def _trace(msg: str) -> None:
    _trace_write("DECISION", msg)


def _current_session() -> Any:
    """The session the call is serving, or ``None`` outside one."""
    try:
        from ...session_context import get_current_session
        return get_current_session()
    except LookupError:
        return None


def _image_names(session: Any) -> List[str]:
    """The session's entries that can carry images; ``[]`` when it cannot say."""
    names = getattr(session, "decision_image_model_names", None)
    if not callable(names):
        return []
    try:
        return list(names())
    except Exception:  # noqa: BLE001
        return []


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
    if qtype == "box":
        return None
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
        if q.get("type") not in ("noul", "choice", "score", "box"):
            raise ValueError(
                f"question {qid!r}: type must be noul, choice, score or box")
        out[qid] = DecisionQuestion(
            type=q["type"], instructions=q.get("instructions"),
            criteria=_criteria(qid, q))
    return out


def _answer_dict(answer: Any) -> Dict[str, Any]:
    value = answer.value
    if isinstance(value, DecisionBox):
        value = {"top": value.top, "bottom": value.bottom,
                 "left": value.left, "right": value.right}
    out: Dict[str, Any] = {"type": answer.type, "value": value}
    if answer.probabilities is not None:
        out["probabilities"] = dict(answer.probabilities)
    if answer.confidence is not None:
        out["confidence"] = answer.confidence
    if answer.legend:
        out["legend"] = dict(answer.legend)
    return out


def _box_image(question: DecisionQuestion,
               images: List[ResolvedImage]) -> Optional[ResolvedImage]:
    """The image a ``box`` answer is relative to, or ``None`` when unclear.

    The image its instructions name with ``{{image:NAME}}`` when they name
    exactly one; else the call's only image.  Otherwise no ``box_px`` is
    given, because converting fractions against the wrong image would give
    a plausible, wrong rectangle.
    """
    named = set(IMAGE_TOKEN_RE.findall(question.instructions or ""))
    by_name = {r.image.name: r for r in images}
    if len(named) == 1:
        return by_name.get(next(iter(named)))
    if not named and len(images) == 1:
        return images[0]
    return None


def _add_box_pixels(answers: Dict[str, Dict[str, Any]], result: Any,
                    questions: Dict[str, DecisionQuestion],
                    images: List[ResolvedImage]) -> None:
    """Put ``image`` and ``box_px`` beside each ``box`` answer that has one."""
    for qid, answer in result.answers.items():
        if not isinstance(answer.value, DecisionBox):
            continue
        source = _box_image(questions[qid], images)
        if source is None:
            continue
        answers[qid]["image"] = source.image.name
        answers[qid]["box_px"] = answer.value.to_pixels(
            source.width, source.height)


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
        self._registry: Any = None

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._agent_name = (config or {}).get("agent_name")

    def shutdown(self) -> None:
        pass

    def set_plugin_registry(self, registry: Any) -> None:
        """Called during ``expose_tool()``.  Registers the decision category."""
        self._registry = registry
        registry.register_category(
            "decision",
            "Typed judgements from a decision model: classify, score, yes/no")

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
            category="decision",
            discoverability=DISCOVERABILITY_DEFERRED,
        )]

    def is_tool_visible(self, tool_name: str) -> bool:
        """``decide`` is hidden from a session that binds no decision model."""
        if tool_name != TOOL_NAME:
            return True
        names = _bound_names(_current_session())
        return names is None or bool(names)

    def narrow_tool_schema(self, schema: ToolSchema) -> ToolSchema:
        """Shape ``decide`` for the calling session.

        ``model`` becomes an enum of its entry names.  When one of them can
        carry images, ``images`` and the ``box`` question type are added
        and the description says which entries take images.
        """
        if getattr(schema, "name", None) != TOOL_NAME:
            return schema
        session = _current_session()
        names = _bound_names(session)
        if not names:
            return schema
        params = copy.deepcopy(schema.parameters)
        params["properties"]["model"]["enum"] = sorted(names)
        description = schema.description
        with_images = _image_names(session)
        if with_images:
            params["properties"]["images"] = images_parameter_schema()
            qtype = params["properties"]["questions"]["items"]["properties"]["type"]
            qtype["enum"] = ["noul", "choice", "score", "box"]
            qtype["description"] += _BOX_TYPE_TEXT
            description += _IMAGES_TOOL_TEXT + (
                "  Models that take images: " + ", ".join(with_images) + ".")
        return dataclasses.replace(
            schema, parameters=params, description=description)

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
            images = self._resolve_images(session, name, args.get("images"))
            # The entry's provider is its own instance, connected to the
            # entry's model, so decide() needs no model= of its own.
            provider = session.decision_provider(str(name))
            from ...ai_tool_runner import get_current_cancel_token
            extra = {"images": [r.image for r in images]} if images else {}
            result = provider.decide(
                state, questions, cancel_token=get_current_cancel_token(),
                **extra)
        except (ValueError, LookupError, DecisionError, RuntimeError) as exc:
            _trace(f"decide: model={name!r} refused: {exc}")
            return False, {"error": str(exc), "model": name}
        payload, usage = _result_payload(name, result)
        _add_box_pixels(payload["answers"], result, questions, images)
        _record_in_ledger(session, name, result, usage, images)
        _trace(
            f"decide: model={name!r} answered_by={result.model!r} "
            f"questions={len(questions)} images={len(images)} usage={usage}")
        return payload

    def _resolve_images(self, session: Any, name: Any,
                        raw: Any) -> List[ResolvedImage]:
        """The call's ``images``, refused before reading for a text-only entry."""
        if raw in (None, []):
            return []
        if str(name) not in _image_names(session):
            raise ImageReferenceError(
                f"decision model {name!r} does not take images; models that "
                f"do: {', '.join(_image_names(session)) or 'none'}")
        from ...session_context import get_workspace_root
        history = session.get_history() if callable(
            getattr(session, "get_history", None)) else []
        return resolve_images(raw, get_workspace_root(), list(history),
                              self._registry)


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


def _record_in_ledger(session: Any, name: Any, result: Any, usage: Any,
                      images: Optional[List[ResolvedImage]] = None) -> None:
    """A ``decision`` record in the session's token ledger, best effort.

    The ledger is where a deployment reads what a session spent; a decision
    call is spend the chat accounting never sees.
    """
    runtime = getattr(session, "_runtime", None)
    ledger = getattr(runtime, "ledger", None)
    record = getattr(ledger, "_record", None)
    if not callable(record):
        return
    entry: Dict[str, Any] = {
        "entry": name,
        "model": result.model,
        "questions": len(result.answers),
        "usage": usage,
    }
    if images:
        entry["images"] = [image_record(r) for r in images]
    try:
        record("decision", entry)
    except Exception as exc:  # noqa: BLE001
        _trace(f"decide: ledger record failed: {exc}")


def create_plugin() -> DecisionPlugin:
    return DecisionPlugin()
