"""Serialization utilities for session persistence.

This module handles converting internal types (Message, Part) to and
from JSON-serializable dictionaries for storage.
"""

import ast
import base64
import dataclasses
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from jaato_sdk.media_identity import ATTACHMENT_ID_KEY, mint_attachment_id
from jaato_sdk.plugins.model_provider.types import (
    Message,
    Part,
    Role,
    FunctionCall,
    ToolResult,
)
from .base import SessionState, SessionInfo

logger = logging.getLogger(__name__)


#: ``Part`` fields that are plain strings and can ride BESIDE a part's
#: primary content rather than being it.  ``thought`` is the one that matters
#: (#1290): the providers that set ``replay_reasoning`` keep a thought part in
#: history on purpose, and one that is lost on persistence is replayed as
#: nothing.  Each is also a primary type of its own when it is the only thing
#: the part carries.  Order is the order a part's primary type is chosen in.
_AUXILIARY_FIELDS = ("thought", "executable_code", "code_execution_result")

#: Every ``Part`` field this module round-trips.  A guard test compares it
#: against ``dataclasses.fields(Part)``, so a field added to ``Part`` fails
#: the build here instead of being dropped on the way to disk the way
#: ``thought`` was.
PERSISTED_PART_FIELDS = frozenset(
    {"text", "function_call", "function_response", "inline_data",
     *_AUXILIARY_FIELDS}
)

#: The exact prefix the pre-#1290 text fallback wrote.  A legacy record
#: carries it on a MODEL-message text part; see
#: :func:`_repair_legacy_part`.
_LEGACY_UNRECOGNIZED_PREFIX = "[Unrecognized part: "


def _naive(dt: datetime) -> datetime:
    """Strip timezone info to ensure naive datetime for consistent comparison."""
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


def _set_field_names(part: Any) -> List[str]:
    """Names of the fields ``part`` carries a value in -- never the values.

    Used by the warnings about a part this module cannot persist or restore.
    Only NAMES are logged: a part's content can be a model's reasoning, a
    tool result or a user's attachment, and none of that belongs in a log.
    """
    try:
        if dataclasses.is_dataclass(part) and not isinstance(part, type):
            return [f.name for f in dataclasses.fields(part)
                    if getattr(part, f.name, None) is not None]
        return sorted(k for k, v in vars(part).items() if v is not None)
    except TypeError:
        return []


def serialize_part(part: Part) -> Dict[str, Any]:
    """Serialize a Part object to a dictionary.

    The dictionary is tagged with the part's PRIMARY content (``type``:
    ``text``, ``function_call``, ``function_response``, ``inline_data``,
    ``thought``, ``executable_code`` or ``code_execution_result``, chosen in
    that order).  Any auxiliary string field the part also carries --
    ``thought`` beside ``text``, for instance -- rides as an extra key of the
    same name, so a part carrying both keeps both (#1290).

    A part with none of those fields set is recorded as
    ``{'type': 'unknown', 'fields': [...]}`` naming the fields it did carry,
    and a WARNING says so.  The content is deliberately NOT written: the
    record used to hold ``repr(part)``, which :func:`deserialize_part` turned
    back into a TEXT part and so replayed a Python repr to the model as
    something the assistant had said.

    Args:
        part: A Part object.

    Returns:
        Dictionary representation of the part.
    """
    data = _serialize_primary(part)
    if data is None:
        fields = _set_field_names(part)
        logger.warning(
            "session serializer: a %s carries no field this serializer "
            "persists (non-None fields: %s); it is recorded as 'unknown' "
            "and will be dropped when the session is restored",
            type(part).__name__, ", ".join(fields) or "none",
        )
        return {'type': 'unknown', 'fields': fields}
    for name in _AUXILIARY_FIELDS:
        value = getattr(part, name, None)
        if value is not None and data['type'] != name:
            data[name] = value
    return data


def _serialize_primary(part: Part) -> Optional[Dict[str, Any]]:
    """The tagged dict for ``part``'s primary content, or ``None``.

    Auxiliary fields are added by :func:`serialize_part`; this decides only
    which content the part IS.
    """
    # Text part
    if part.text is not None:
        return {
            'type': 'text',
            'text': part.text
        }

    # Function call part
    if part.function_call is not None:
        fc = part.function_call
        return {
            'type': 'function_call',
            'id': fc.id,
            'name': fc.name,
            'args': fc.args
        }

    # Function response part
    if part.function_response is not None:
        fr = part.function_response
        return {
            'type': 'function_response',
            'call_id': fr.call_id,
            'name': fr.name,
            'result': fr.result,
            'is_error': fr.is_error,
            # The untrusted-content boundary must survive persistence.
            # Without these two keys a restored session re-sends
            # sibling-/web-/MCP-authored text to the model as ORDINARY
            # content: the provider converter wraps on the MARK, so an
            # unmarked result is never wrapped and never escaped.  The
            # security property silently weakened at exactly the moment
            # nothing looked different -- same history, same text, no
            # boundary.
            'untrusted': fr.untrusted,
            'untrusted_source': fr.untrusted_source,
        }

    # Inline data (images, audio, PDFs)
    if part.inline_data is not None:
        inline = part.inline_data
        data_bytes = inline.get('data')
        return {
            'type': 'inline_data',
            'mime_type': inline.get('mime_type'),
            'data': base64.b64encode(data_bytes).decode('utf-8') if data_bytes else None,
            # Both of these were dropped on the way to disk, so a revived
            # session lost them: the PDF ``file`` block's filename (the
            # converters read ``display_name`` back off ``inline_data``) and
            # the attachment id (#850), without which whatever replaces
            # purged bytes cannot name the archived recording.
            'display_name': inline.get('display_name'),
            ATTACHMENT_ID_KEY: inline.get(ATTACHMENT_ID_KEY),
        }

    # A part whose primary content is one of the auxiliary strings --
    # reasoning (#1290), or Gemini's code execution.  The key carries the
    # field's own name so the dict reads the same whichever role it plays.
    for name in _AUXILIARY_FIELDS:
        value = getattr(part, name, None)
        if value is not None:
            return {'type': name, name: value}

    return None


def deserialize_part(data: Dict[str, Any]) -> Optional[Part]:
    """Deserialize a dictionary to a Part object.

    Auxiliary string fields (``thought``, ``executable_code``,
    ``code_execution_result``) are restored from their keys whatever the
    part's primary type, so a part that carried text and reasoning comes
    back carrying both.

    An ``unknown`` entry returns ``None`` -- the caller drops it -- and logs
    a WARNING.  It is never turned into a text part: before #1290 it came
    back as ``Part(text="[Unrecognized part: <repr>]")``, which a revived
    session then sent to the model as assistant content.  A legacy
    ``unknown`` entry whose repr is a thought part is recovered by
    :func:`deserialize_message` before it reaches here (MODEL messages
    only).

    Args:
        data: Dictionary representation of a part.

    Returns:
        The reconstructed Part, or ``None`` for an entry that is dropped.

    Raises:
        ValueError: If the part type is not recognized.
    """
    part = _deserialize_primary(data)
    if part is None:
        return None
    for name in _AUXILIARY_FIELDS:
        value = data.get(name)
        if isinstance(value, str) and data.get('type') != name:
            setattr(part, name, value)
    return part


def _deserialize_primary(data: Dict[str, Any]) -> Optional[Part]:
    """The Part for ``data``'s primary type, before auxiliary fields.

    ``None`` means "drop this entry" (an ``unknown`` record); an
    unrecognised ``type`` still raises, because a record naming a type no
    release wrote is corruption rather than something to skip past.
    """
    part_type = data.get('type')

    if part_type == 'text':
        return Part(text=data['text'])

    if part_type in _AUXILIARY_FIELDS:
        value = data.get(part_type)
        return Part(**{part_type: value if isinstance(value, str) else ""})

    if part_type == 'function_call':
        return Part(function_call=FunctionCall(
            id=data.get('id', ''),
            name=data['name'],
            args=data.get('args', {})
        ))

    if part_type == 'function_response':
        return Part(function_response=ToolResult(
            call_id=data.get('call_id', ''),
            name=data['name'],
            result=data.get('result'),
            is_error=data.get('is_error', False),
            # ``.get`` with a safe default: transcripts written before
            # these keys existed restore as trusted, which is the
            # pre-existing behaviour rather than a new claim.  Anything
            # written since carries its real mark.
            untrusted=data.get('untrusted', False),
            untrusted_source=data.get('untrusted_source'),
        ))

    if part_type == 'inline_data':
        raw_data = None
        if data.get('data'):
            raw_data = base64.b64decode(data['data'])
        return Part(inline_data={
            'mime_type': data.get('mime_type'),
            'data': raw_data,
            # ``.get`` with a safe default, like the tool-result keys above:
            # a record written before these were persisted restores without
            # them rather than failing.  The id is RE-MINTED in that case
            # because it is a digest of the payload, so an older record
            # regains the identity it was written without -- the same value
            # ingest would have given it.
            'display_name': data.get('display_name'),
            ATTACHMENT_ID_KEY: (data.get(ATTACHMENT_ID_KEY)
                                or mint_attachment_id(raw_data)),
        })

    if part_type == 'unknown':
        # Dropped, never turned into text (#1290).  Name the fields the
        # writer saw when it recorded them; a legacy record carries a repr
        # instead, whose CONTENT is not logged -- only its length.
        fields = data.get('fields')
        if isinstance(fields, list):
            described = ", ".join(str(f) for f in fields) or "none"
        else:
            described = f"legacy repr of {len(str(data.get('repr', '')))} chars"
        logger.warning(
            "session serializer: dropping a part recorded as 'unknown' "
            "(%s); it cannot be restored and is not replayed as text",
            described,
        )
        return None

    raise ValueError(f"Unknown part type: {part_type}")


# ---------------------------------------------------------------------------
# Legacy repair (#1290)
# ---------------------------------------------------------------------------
#
# Records written before #1290 hold a reasoning part in one of two shapes:
#
#   {'type': 'unknown', 'repr': "Part(text=None, ..., thought='...', ...)"}
#       -- what serialize_part wrote for it, directly;
#   {'type': 'text', 'text': "[Unrecognized part: Part(text=None, ...)]"}
#       -- what that entry became once deserialize_part had turned it into
#          text and the history was saved again (every runner round trip
#          does this, so a revived session's next save writes this form).
#
# Both are recovered into a thought part, in MODEL messages only.  The repr
# is PARSED, never evaluated: ``ast.parse(mode="eval")`` builds a syntax
# tree without running anything, and only a ``Part(...)`` call whose every
# argument is a keyword naming a known field, with a ``None`` or a string
# literal as its value, is accepted.  Anything else -- a nested
# FunctionCall, a positional argument, a name that is not ``Part`` -- is
# refused, and the entry is dropped with a WARNING.

#: Sentinel: this entry is not a legacy shape; deserialize it normally.
_NOT_LEGACY = object()

#: Fields a repaired part may carry.  The four structured fields can only
#: be ``None`` in a repr we accept -- their values are nested objects whose
#: repr cannot be parsed back safely -- and a part that was ``unknown`` had
#: all four ``None`` by construction.
_REPR_STRING_FIELDS = frozenset(_AUXILIARY_FIELDS)
_REPR_NONE_ONLY_FIELDS = frozenset(
    {"text", "function_call", "function_response", "inline_data"}
)


def _part_from_repr(expr: str) -> Optional[Part]:
    """Restore the Part a pre-#1290 ``repr(part)`` described, or ``None``.

    Strict by design: the repr must be exactly a ``Part(...)`` call with
    keyword arguments only, each naming a ``Part`` field; the structured
    fields must be ``None`` and the string fields ``None`` or a string
    literal.  ``ast.literal_eval`` semantics, one level up -- nothing is
    ever executed.  ``None`` means the repr could not be read safely, or
    described a part with nothing in it; the caller drops the entry.
    """
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    call = tree.body
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            and call.func.id == "Part" and not call.args):
        return None
    values: Dict[str, str] = {}
    for kw in call.keywords:
        if kw.arg is None or kw.arg in values:
            return None
        ok, value = _repr_keyword_value(kw)
        if not ok:
            return None
        if value is not None:
            values[kw.arg] = value
    return Part(**values) if values else None


def _repr_keyword_value(kw: ast.keyword) -> tuple:
    """Judge one ``field=value`` of a legacy repr: ``(acceptable, value)``.

    Acceptable is a literal ``None`` for any known field, or a string
    literal for a string field.  Anything else -- a call, a name, a number,
    an unknown field -- is not, and the whole repr is refused.
    """
    if not isinstance(kw.value, ast.Constant):
        return False, None
    value = kw.value.value
    known = kw.arg in _REPR_NONE_ONLY_FIELDS or kw.arg in _REPR_STRING_FIELDS
    if value is None:
        return known, None
    if kw.arg in _REPR_STRING_FIELDS and isinstance(value, str):
        return True, value
    return False, None


def _legacy_repr(data: Dict[str, Any]) -> Optional[str]:
    """The ``Part(...)`` repr a legacy entry carries, or ``None``.

    ``None`` means the entry is not one of the two legacy shapes and must be
    deserialized normally.  The text form is recognised only when the WHOLE
    text is the fallback's shape -- a model that merely quotes it keeps its
    text -- and only when the entry carries nothing else.
    """
    kind = data.get('type')
    if kind == 'unknown':
        raw = data.get('repr')
        return raw if isinstance(raw, str) else None
    if kind != 'text' or any(name in data for name in _AUXILIARY_FIELDS):
        return None
    text = data.get('text')
    if not isinstance(text, str):
        return None
    if not (text.startswith(_LEGACY_UNRECOGNIZED_PREFIX + "Part(")
            and text.endswith(")]")):
        return None
    return text[len(_LEGACY_UNRECOGNIZED_PREFIX):-1]


def _repair_legacy_part(data: Dict[str, Any]) -> Union[Part, None, object]:
    """Recover a pre-#1290 reasoning part from a MODEL-message entry.

    Returns the restored Part, ``None`` to drop the entry (a legacy shape
    whose repr could not be read safely), or :data:`_NOT_LEGACY` when the
    entry is not a legacy shape at all.
    """
    expr = _legacy_repr(data)
    if expr is None:
        return _NOT_LEGACY
    part = _part_from_repr(expr)
    if part is None:
        logger.warning(
            "session serializer: dropping a legacy '%s' entry whose "
            "recorded repr (%d chars) could not be parsed safely into a "
            "part (#1290)", data.get('type'), len(expr),
        )
        return None
    logger.info(
        "session serializer: restored a legacy '%s' entry as a %s part "
        "(#1290)", data.get('type'),
        "+".join(_set_field_names(part)) or "empty",
    )
    return part


def serialize_message(message: Message) -> Dict[str, Any]:
    """Serialize a Message object to a dictionary.

    Includes provenance fields (model, provider) when present, enabling
    cross-provider history to round-trip through session persistence.

    Args:
        message: A Message object.

    Returns:
        Dictionary representation of the message.
    """
    result = {
        'role': message.role.value,
        'parts': [serialize_part(p) for p in (message.parts or [])]
    }
    if message.model is not None:
        result['model'] = message.model
    if message.provider is not None:
        result['provider'] = message.provider
    return result


def deserialize_message(data: Dict[str, Any]) -> Message:
    """Deserialize a dictionary to a Message object.

    Reads provenance fields (model, provider) when present. Old persisted
    sessions that lack these keys deserialize with None defaults
    (backward-compatible).

    In a MODEL message, a pre-#1290 reasoning part (recorded as an
    ``unknown`` repr, or as the ``[Unrecognized part: Part(...)]`` text that
    entry used to become) is recovered as a thought part -- see
    :func:`_repair_legacy_part`.  The repair is confined to MODEL messages
    because only the model produces reasoning, and a user who typed that
    string meant it as text.

    Entries that are dropped (``unknown``, an unreadable legacy repr) leave
    the message with fewer parts, possibly none; :func:`deserialize_history`
    removes a message that was emptied that way.

    Args:
        data: Dictionary representation of message.

    Returns:
        Reconstructed Message object.
    """
    role = Role(data['role'])
    parts: List[Part] = []
    for raw in data.get('parts', []):
        part: Any = _NOT_LEGACY
        if role == Role.MODEL and isinstance(raw, dict):
            part = _repair_legacy_part(raw)
        if part is _NOT_LEGACY:
            part = deserialize_part(raw)
        if part is not None:
            parts.append(part)
    return Message(
        role=role,
        parts=parts,
        model=data.get('model'),
        provider=data.get('provider'),
    )


def serialize_history(history: List[Any]) -> List[Dict[str, Any]]:
    """Serialize a conversation history to a list of dictionaries.

    Path E (cycle 6) idempotency contract: accepts either a list
    of :class:`Message` objects (canonical input — the historical
    contract) OR a list of already-serialized dicts (from the
    runner-RPC ``session_get_history`` wire — produced by the
    canonical ``serialize_message`` since Path E).  Dict elements
    pass through unchanged; ``Message`` elements are serialized.
    Mixed lists are tolerated.

    Pre-Path-E this function crashed with ``'dict' object has no
    attribute 'role'`` when given the wire dicts, breaking both
    ``session_manager._save_session`` (via
    ``serialize_session_state``) and the replay path (via
    ``session_replay_messages_threadsafe``).  Idempotency closes
    both crashes without requiring callers to pre-deserialize.

    Args:
        history: List of Message objects OR already-serialized dicts.

    Returns:
        List of dictionary representations.
    """
    result: List[Dict[str, Any]] = []
    for m in (history or []):
        if isinstance(m, dict):
            result.append(m)
        else:
            result.append(serialize_message(m))
    return result


def deserialize_history(data: List[Dict[str, Any]]) -> List[Message]:
    """Deserialize a list of dictionaries to conversation history.

    A message whose record HAD parts and came back with none -- every one
    of them an ``unknown`` entry or an unreadable legacy repr (#1290) -- is
    removed with a WARNING rather than restored empty.  The history
    invariant (``shared/history_invariant.py``) removes empty text blocks
    but keeps a message with zero parts, and no wire accepts an assistant
    turn with no content.  Removing it cannot orphan a tool result: such a
    message held no function call, since calls always serialize.  A
    message that was ALREADY empty in the record is left alone -- that is
    not something this module did.

    Args:
        data: List of dictionary representations.

    Returns:
        List of Message objects.
    """
    messages: List[Message] = []
    for record in (data or []):
        message = deserialize_message(record)
        if not message.parts and record.get('parts'):
            logger.warning(
                "session serializer: dropping a %s message whose %d "
                "recorded part(s) could not be restored (#1290)",
                message.role.value, len(record.get('parts') or []),
            )
            continue
        messages.append(message)
    return messages


def serialize_session_state(state: SessionState) -> Dict[str, Any]:
    """Serialize a SessionState to a JSON-compatible dictionary.

    Args:
        state: The SessionState to serialize.

    Returns:
        JSON-compatible dictionary.
    """
    return {
        # 2.8: profile_snapshot / rendered_instructions / agent_params --
        # a revived session RESTORES the recipe and the prompt it ran
        # under instead of re-deriving them from disk (issue #787).
        # 2.9: created_by -- the authenticated user the session was
        # created for, so the record is attributable without telemetry
        # (issue #859).
        # 2.10: runner_identity -- which PROCESS is (or last was) executing
        # this session, so a session an operator can see is one they can act
        # on (issue #812).
        'version': '2.10',
        'session_id': state.session_id,
        'description': state.description,
        'created_at': state.created_at.isoformat(),
        'updated_at': state.updated_at.isoformat(),
        'turn_count': state.turn_count,
        'turn_accounting': state.turn_accounting,
        'user_inputs': state.user_inputs,
        'metadata': state.metadata,
        'profile_name': state.profile_name,
        'profile_spec': state.profile_spec,  # unresolved inline recipe (2.7+)
        # 2.8+ (issue #787).  The frozen recipe and the frozen prompt: a
        # revive reads these rather than re-resolving the profile name and
        # re-running the persona's prefetch scripts.  All three are None on
        # older records, and the loader falls back to re-deriving -- which
        # is the pre-2.8 behaviour, so old records keep loading unchanged.
        'profile_snapshot': state.profile_snapshot,
        'rendered_instructions': state.rendered_instructions,
        'agent_params': state.agent_params,
        'workspace_path': state.workspace_path,
        'config_root': state.config_root,
        'sandbox_mode': state.sandbox_mode,
        'agent_name': state.agent_name,
        # 2.9+ (#859).  Fixed key list, like every field above: the
        # dataclass field alone never reaches disk.
        'created_by': state.created_by,
        'history': serialize_history(state.history),
        'budget_state': state.budget_state,
        # budget_control usage.  Enumerated explicitly like every other field
        # here -- adding it to the dataclass alone was NOT enough: this
        # serializer writes a fixed key list, so the field was silently
        # dropped and the persisted JSON carried no key at all.
        'budget_usage': state.budget_usage,
        'budget_exhausted_reason': state.budget_exhausted_reason,
        # The CEILING (distinct from budget_usage, the spend).  This
        # serializer writes a FIXED key list -- a field absent here
        # never reaches disk however well it is wired elsewhere.
        'budget_control': state.budget_control,
        'sibling_name': state.sibling_name,
        'cascade_driver_id': state.cascade_driver_id,
        # 2.10+ (#812).  Enumerated explicitly like every field above: this
        # serializer writes a FIXED key list, so a field added to the
        # dataclass alone never reaches disk.
        'runner_identity': state.runner_identity,
        'interrupted_turn': state.interrupted_turn,
        'session_state': state.session_state,
    }


def deserialize_session_state(data: Dict[str, Any]) -> SessionState:
    """Deserialize a dictionary to a SessionState.

    Args:
        data: Dictionary from JSON file.

    Returns:
        Reconstructed SessionState.

    Raises:
        ValueError: If required fields are missing or version is incompatible.
    """
    version = data.get('version', '1.0')
    # Support 1.x (legacy) + 2.x (new Message type).  2.3+ retires
    # the Google-coupled ``connection`` dict (project/location/model);
    # legacy data carrying ``connection`` is tolerated but silently
    # ignored — the profile (not state.model) is the post-multi-
    # provider source of truth for model + provider binding.
    if not (version.startswith('1.') or version.startswith('2.')):
        raise ValueError(f"Unsupported session version: {version}")

    return SessionState(
        session_id=data['session_id'],
        history=deserialize_history(data.get('history', [])),
        created_at=_naive(datetime.fromisoformat(data['created_at'])),
        updated_at=_naive(datetime.fromisoformat(data['updated_at'])),
        description=data.get('description'),
        turn_count=data.get('turn_count', 0),
        turn_accounting=data.get('turn_accounting', []),
        user_inputs=data.get('user_inputs', []),
        metadata=data.get('metadata', {}),
        profile_name=data.get('profile_name'),
        profile_spec=data.get('profile_spec'),  # None on pre-2.7 records
        profile_snapshot=data.get('profile_snapshot'),  # None pre-2.8
        rendered_instructions=data.get('rendered_instructions'),  # pre-2.8
        agent_params=data.get('agent_params'),  # None on pre-2.8 records
        workspace_path=data.get('workspace_path'),
        config_root=data.get('config_root'),
        sandbox_mode=data.get('sandbox_mode'),
        agent_name=data.get('agent_name'),
        created_by=data.get('created_by'),  # None on pre-2.9 records
        budget_state=data.get('budget_state'),
        budget_usage=data.get('budget_usage'),
        budget_exhausted_reason=data.get('budget_exhausted_reason'),
        budget_control=data.get('budget_control'),
        sibling_name=data.get('sibling_name'),
        cascade_driver_id=data.get('cascade_driver_id'),
        runner_identity=data.get('runner_identity'),  # None on pre-2.10
        interrupted_turn=data.get('interrupted_turn'),
        session_state=data.get('session_state'),
    )


def serialize_session_info(state: SessionState) -> Dict[str, Any]:
    """Extract SessionInfo-level data from a SessionState for quick listing.

    This is a subset of the full state, suitable for index files.

    Args:
        state: The SessionState to extract info from.

    Returns:
        Dictionary with just the metadata fields.
    """
    return {
        'session_id': state.session_id,
        'description': state.description,
        'created_at': state.created_at.isoformat(),
        'updated_at': state.updated_at.isoformat(),
        'turn_count': state.turn_count,
        'profile_name': state.profile_name,
        'workspace_path': state.workspace_path,
        # Needed while the session is COLD -- see SessionInfo.
        'cascade_driver_id': state.cascade_driver_id,
        'sibling_name': state.sibling_name,
        # The group predicate's other half (server.session_groups): a cold
        # session answers "do we share an owner" off the listing.
        'created_by': state.created_by,
    }


def deserialize_session_info(data: Dict[str, Any]) -> SessionInfo:
    """Deserialize a dictionary to a SessionInfo.

    Args:
        data: Dictionary with session metadata.

    Returns:
        SessionInfo object.
    """
    return SessionInfo(
        session_id=data['session_id'],
        description=data.get('description'),
        created_at=_naive(datetime.fromisoformat(data['created_at'])),
        updated_at=_naive(datetime.fromisoformat(data['updated_at'])),
        cascade_driver_id=data.get('cascade_driver_id'),
        sibling_name=data.get('sibling_name'),
        created_by=data.get('created_by'),  # None on pre-2.9 records
        turn_count=data.get('turn_count', 0),
        # Pre-2.3 sessions wrote 'model' instead of 'profile_name'.
        # Old indexes deserialize with profile_name=None; consumers
        # that need the model resolve via the profile registry.
        profile_name=data.get('profile_name'),
        workspace_path=data.get('workspace_path'),
    )
