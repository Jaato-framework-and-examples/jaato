"""Decision models: typed questions in, calibrated answers out.

A decision model (TypeSafe Jev, ConvAI Laya) takes a STATE and a map of
typed QUESTIONS and returns one typed answer per question, with
probabilities, in a single forward pass.  It writes no text, keeps no
history and calls no tools, so it is served by a provider's ``decide()``
call and never by ``complete()``.  Design:
``docs/design/decision-models.md``.

This module is the wire contract every adapter shares, kept pure (no I/O)
so a stand-in server, the ``openrouter`` adapter and a later ``typesafe``
or ``laya`` adapter cannot disagree about it:

* :class:`DecisionQuestion` / :class:`DecisionAnswer` /
  :class:`DecisionResult` are the types callers hold.
* :func:`build_decision_request` validates the questions and builds the
  request body.  A malformed question is a local
  :class:`DecisionQuestionError` naming the question, never a ``422``
  round trip.
* :func:`parse_decision_response` turns a response body into a
  :class:`DecisionResult`.  An answer missing for a question that was
  asked is :class:`DecisionResponseError` naming it.  It is never filled
  with a default: a decision that silently became "no" or the first
  option is the worst failure this feature could have.

The wire, as the vendor documents it (Jev 1.13):

* request ``{"state", "model", "questions": {id: {"type", "instructions",
  "criteria"}}}``;
* answers ``noul`` ``{"type", "noul"}``, ``choice`` ``{"type", "choice",
  "probabilities", "confidence"}``, ``score`` ``{"type", "score",
  "legend", "probabilities", "confidence"}``;
* ``usage`` ``{"input_tokens", "output_tokens"}``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol, Union, runtime_checkable

from .types import CancelToken, TokenUsage

#: The three question types both known decision models answer.
QUESTION_TYPES = ("noul", "choice", "score")

#: ``choice`` takes 1..255 options; ``score`` takes 2..10 ordered levels.
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10

#: A ``noul`` question's criteria, when given, name these outcomes.
NOUL_CRITERIA_KEYS = frozenset({"true", "false"})

State = Union[str, dict, list]


class DecisionError(Exception):
    """Base class for decision-call failures that are not transport errors."""


class DecisionQuestionError(DecisionError, ValueError):
    """A question (or the state) is malformed.  Raised before sending."""

    def __init__(self, question_id: Optional[str], reason: str) -> None:
        self.question_id = question_id
        self.reason = reason
        where = f"question {question_id!r}" if question_id is not None else "request"
        super().__init__(f"Decision {where}: {reason}")


class DecisionResponseError(DecisionError):
    """The response does not answer what was asked, or answers it in a
    shape this parser cannot read.  ``question_id`` names the question
    when one is to blame."""

    def __init__(self, reason: str, question_id: Optional[str] = None,
                 body: Any = None) -> None:
        self.question_id = question_id
        self.reason = reason
        self.body = body
        where = f" for question {question_id!r}" if question_id else ""
        super().__init__(f"Decision response{where}: {reason}")


class DecisionModelOnlyError(RuntimeError):
    """``complete()`` was called on a model that only answers decisions.

    A decisions-only model writes no text and calls no tools, so the
    session's agent loop cannot run on it.  Call ``decide()`` instead.
    """

    def __init__(self, model: str, provider: str = "") -> None:
        self.model = model
        self.provider = provider
        via = f" on {provider}" if provider else ""
        super().__init__(
            f"Model {model!r}{via} is a decision model (output modality "
            "'decisions' only): it answers typed questions through decide() "
            "and cannot run a chat turn.  Bind it as a decision model, not "
            "as the session's model or a tier."
        )


@dataclass(frozen=True)
class DecisionQuestion:
    """One typed question.

    ``criteria`` by type:

    * ``noul``: ``None``, or ``{"true": ..., "false": ...}`` describing
      each outcome;
    * ``choice``: ``{option: description or None}``, 1..255 options;
    * ``score``: ``[level, ...]``, 2..10 levels, lowest first.

    ``instructions`` and every criteria value may be a string, an object
    or an array, so a question can carry structured reference data.
    """

    type: str
    instructions: Union[str, dict, list]
    criteria: Union[None, dict, list] = None

    def options(self) -> Optional[list]:
        """The ``choice`` options in declared order, else ``None``."""
        if self.type == "choice" and isinstance(self.criteria, dict):
            return list(self.criteria)
        return None

    def to_wire(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.criteria is not None:
            body["criteria"] = self.criteria
        return body


@dataclass(frozen=True)
class DecisionAnswer:
    """One answer.

    ``value`` is the noul probability, the chosen option, or the
    probability-weighted score (which may land between levels).
    ``confidence`` is ``None`` when the wire sent none, which is always
    the case for ``noul``: absent is not zero.
    """

    type: str
    value: Union[float, str]
    probabilities: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None
    legend: Optional[Dict[str, str]] = None


@dataclass(frozen=True)
class DecisionResult:
    """What a decision call returned.

    ``model`` is the model that answered as the response names it (e.g.
    ``jev-1.13.0``), which may differ from the id requested.  ``usage``
    has ``reported=False`` when the wire carried no usage (#688).
    ``raw`` is the response body as received, for diagnostics.
    """

    model: str
    answers: Dict[str, DecisionAnswer]
    usage: TokenUsage
    raw: Optional[Dict[str, Any]] = None


@runtime_checkable
class DecisionCapable(Protocol):
    """A provider that serves decision models (``ProviderCapabilities.decisions``)."""

    def decide(
        self,
        state: State,
        questions: Mapping[str, DecisionQuestion],
        *,
        model: Optional[str] = None,
        cancel_token: Optional[CancelToken] = None,
    ) -> DecisionResult: ...


# ------------------------------------------------------------------ request


def _is_structured(value: Any) -> bool:
    return isinstance(value, (dict, list))


def _check_text_or_structure(qid: Optional[str], what: str, value: Any) -> None:
    if isinstance(value, str):
        if not value.strip():
            raise DecisionQuestionError(qid, f"{what} is empty")
        return
    if not _is_structured(value) or not value:
        raise DecisionQuestionError(
            qid, f"{what} must be a non-empty string, object or array"
        )


def _check_noul(qid: str, criteria: Any) -> None:
    if criteria is None:
        return
    if not isinstance(criteria, dict) or not criteria:
        raise DecisionQuestionError(
            qid, 'noul criteria must be None or {"true": ..., "false": ...}'
        )
    unknown = set(criteria) - NOUL_CRITERIA_KEYS
    if unknown:
        raise DecisionQuestionError(
            qid, f"noul criteria keys must be 'true'/'false', got {sorted(unknown)}"
        )


def _check_choice(qid: str, criteria: Any) -> None:
    if not isinstance(criteria, dict) or not criteria:
        raise DecisionQuestionError(
            qid, "choice criteria must be a non-empty {option: description} map"
        )
    if len(criteria) > MAX_CHOICE_OPTIONS:
        raise DecisionQuestionError(
            qid, f"choice takes at most {MAX_CHOICE_OPTIONS} options, got {len(criteria)}"
        )
    if any(not isinstance(k, str) or not k.strip() for k in criteria):
        raise DecisionQuestionError(qid, "every choice option must be a non-empty string")


def _check_score(qid: str, criteria: Any) -> None:
    if not isinstance(criteria, list):
        raise DecisionQuestionError(qid, "score criteria must be a list of levels")
    if not MIN_SCORE_LEVELS <= len(criteria) <= MAX_SCORE_LEVELS:
        raise DecisionQuestionError(
            qid,
            f"score takes {MIN_SCORE_LEVELS}..{MAX_SCORE_LEVELS} levels, "
            f"got {len(criteria)}",
        )


_CRITERIA_CHECKS = {"noul": _check_noul, "choice": _check_choice, "score": _check_score}


def validate_question(qid: Any, question: Any) -> None:
    """Raise :class:`DecisionQuestionError` unless ``question`` is sendable."""
    if not isinstance(qid, str) or not qid.strip():
        raise DecisionQuestionError(None, f"question id must be a non-empty string, got {qid!r}")
    if not isinstance(question, DecisionQuestion):
        raise DecisionQuestionError(qid, "must be a DecisionQuestion")
    if question.type not in QUESTION_TYPES:
        raise DecisionQuestionError(
            qid, f"type must be one of {', '.join(QUESTION_TYPES)}, got {question.type!r}"
        )
    _check_text_or_structure(qid, "instructions", question.instructions)
    _CRITERIA_CHECKS[question.type](qid, question.criteria)


def build_decision_request(
    state: State,
    questions: Mapping[str, DecisionQuestion],
    model: str,
) -> Dict[str, Any]:
    """Validate and build the request body.  Ids are kept as given: they
    are the caller's own and come back as the keys of ``answers``."""
    _check_text_or_structure(None, "state", state)
    if not isinstance(questions, Mapping) or not questions:
        raise DecisionQuestionError(None, "at least one question is required")
    if not isinstance(model, str) or not model.strip():
        raise DecisionQuestionError(None, "no model named")
    for qid, question in questions.items():
        validate_question(qid, question)
    return {
        "state": state,
        "model": model,
        "questions": {qid: q.to_wire() for qid, q in questions.items()},
    }


# ----------------------------------------------------------------- response


def _number(value: Any) -> Optional[float]:
    """A finite float, or ``None``.  ``bool`` is not a number here."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _probability(qid: str, field: str, value: Any, body: Any) -> float:
    number = _number(value)
    if number is None or not 0.0 <= number <= 1.0:
        raise DecisionResponseError(
            f"{field} must be a probability in [0, 1], got {value!r}", qid, body
        )
    return number


def _number_map(value: Any) -> Optional[Dict[str, float]]:
    """``{key: number}`` with string keys, or ``None`` when the wire sent
    something else.  Lenient on purpose: the distribution is evidence,
    the answer's ``value`` is what a caller acts on."""
    if not isinstance(value, dict):
        return None
    out: Dict[str, float] = {}
    for key, number in value.items():
        parsed = _number(number)
        if parsed is None:
            return None
        out[str(key)] = parsed
    return out


def _optional_confidence(qid: str, answer: Dict[str, Any], body: Any) -> Optional[float]:
    if answer.get("confidence") is None:
        return None
    return _probability(qid, "confidence", answer["confidence"], body)


def _parse_noul(qid: str, question: DecisionQuestion, answer: Dict[str, Any],
                body: Any) -> DecisionAnswer:
    return DecisionAnswer(type="noul", value=_probability(qid, "noul", answer.get("noul"), body))


def _parse_choice(qid: str, question: DecisionQuestion, answer: Dict[str, Any],
                  body: Any) -> DecisionAnswer:
    chosen = answer.get("choice")
    if not isinstance(chosen, str) or chosen not in (question.options() or []):
        raise DecisionResponseError(
            f"choice {chosen!r} is not one of the options asked", qid, body
        )
    return DecisionAnswer(
        type="choice",
        value=chosen,
        probabilities=_number_map(answer.get("probabilities")),
        confidence=_optional_confidence(qid, answer, body),
    )


def _parse_score(qid: str, question: DecisionQuestion, answer: Dict[str, Any],
                 body: Any) -> DecisionAnswer:
    score = _number(answer.get("score"))
    top = len(question.criteria or []) - 1
    if score is None or not 0.0 <= score <= top:
        raise DecisionResponseError(
            f"score must be a number in [0, {top}], got {answer.get('score')!r}", qid, body
        )
    legend = answer.get("legend")
    return DecisionAnswer(
        type="score",
        value=score,
        probabilities=_number_map(answer.get("probabilities")),
        confidence=_optional_confidence(qid, answer, body),
        legend={str(k): str(v) for k, v in legend.items()} if isinstance(legend, dict) else None,
    )


_ANSWER_PARSERS = {"noul": _parse_noul, "choice": _parse_choice, "score": _parse_score}


def parse_decision_usage(usage: Any) -> TokenUsage:
    """``{"input_tokens", "output_tokens"}`` to :class:`TokenUsage`.

    A body with no usage gives ``reported=False``: nothing was measured,
    which is not a measured zero (#688).  A ``cost`` the gateway adds
    (OpenRouter does on chat) is kept as ``cost_usd``.
    """
    if not isinstance(usage, dict):
        return TokenUsage(reported=False)
    prompt = _number(usage.get("input_tokens"))
    output = _number(usage.get("output_tokens"))
    if prompt is None and output is None:
        return TokenUsage(reported=False)
    prompt_i, output_i = int(prompt or 0), int(output or 0)
    return TokenUsage(
        prompt_tokens=prompt_i,
        output_tokens=output_i,
        total_tokens=prompt_i + output_i,
        cost_usd=_number(usage.get("cost")),
        reported=True,
    )


def parse_decision_response(
    body: Any,
    questions: Mapping[str, DecisionQuestion],
    requested_model: str = "",
) -> DecisionResult:
    """Read a response body against the questions that were asked.

    Every asked id must be answered, with the type it was asked as.  An
    answer for an id that was not asked is ignored.
    """
    if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
        raise DecisionResponseError("no 'answers' object in the response", body=body)
    received = body["answers"]
    missing = [qid for qid in questions if qid not in received]
    if missing:
        raise DecisionResponseError(
            f"no answer for {', '.join(repr(q) for q in missing)}",
            question_id=missing[0], body=body,
        )
    answers: Dict[str, DecisionAnswer] = {}
    for qid, question in questions.items():
        answer = received[qid]
        if not isinstance(answer, dict):
            raise DecisionResponseError("answer is not an object", qid, body)
        if answer.get("type", question.type) != question.type:
            raise DecisionResponseError(
                f"asked as {question.type!r}, answered as {answer.get('type')!r}", qid, body
            )
        answers[qid] = _ANSWER_PARSERS[question.type](qid, question, answer, body)
    model = body.get("model") if isinstance(body.get("model"), str) else requested_model
    return DecisionResult(
        model=model or requested_model,
        answers=answers,
        usage=parse_decision_usage(body.get("usage")),
        raw=body,
    )


__all__ = [
    "QUESTION_TYPES",
    "MAX_CHOICE_OPTIONS",
    "MIN_SCORE_LEVELS",
    "MAX_SCORE_LEVELS",
    "DecisionError",
    "DecisionQuestionError",
    "DecisionResponseError",
    "DecisionModelOnlyError",
    "DecisionQuestion",
    "DecisionAnswer",
    "DecisionResult",
    "DecisionCapable",
    "validate_question",
    "build_decision_request",
    "parse_decision_response",
    "parse_decision_usage",
]
