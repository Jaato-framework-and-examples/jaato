"""Serialization utilities for session persistence.

This module handles converting internal types (Message, Part) to and
from JSON-serializable dictionaries for storage.
"""

import base64
from datetime import datetime
from typing import Any, Dict, List, Optional

from jaato_sdk.plugins.model_provider.types import (
    Message,
    Part,
    Role,
    FunctionCall,
    ToolResult,
)
from .base import SessionState, SessionInfo


def _naive(dt: datetime) -> datetime:
    """Strip timezone info to ensure naive datetime for consistent comparison."""
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


def serialize_part(part: Part) -> Dict[str, Any]:
    """Serialize a Part object to a dictionary.

    Handles text, function calls, function responses, and inline data.

    Args:
        part: A Part object.

    Returns:
        Dictionary representation of the part.
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

    # Inline data (images, etc.)
    if part.inline_data is not None:
        inline = part.inline_data
        data_bytes = inline.get('data')
        return {
            'type': 'inline_data',
            'mime_type': inline.get('mime_type'),
            'data': base64.b64encode(data_bytes).decode('utf-8') if data_bytes else None
        }

    # Unknown part type - try to capture what we can
    return {
        'type': 'unknown',
        'repr': repr(part)
    }


def deserialize_part(data: Dict[str, Any]) -> Part:
    """Deserialize a dictionary to a Part object.

    Args:
        data: Dictionary representation of a part.

    Returns:
        Reconstructed Part object.

    Raises:
        ValueError: If the part type is not recognized.
    """
    part_type = data.get('type')

    if part_type == 'text':
        return Part(text=data['text'])

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
            'data': raw_data
        })

    if part_type == 'unknown':
        # Best effort - create a text part with the repr
        return Part(text=f"[Unrecognized part: {data.get('repr', '?')}]")

    raise ValueError(f"Unknown part type: {part_type}")


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

    Args:
        data: Dictionary representation of message.

    Returns:
        Reconstructed Message object.
    """
    parts = [deserialize_part(p) for p in data.get('parts', [])]
    return Message(
        role=Role(data['role']),
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

    Args:
        data: List of dictionary representations.

    Returns:
        List of Message objects.
    """
    return [deserialize_message(d) for d in (data or [])]


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
        'version': '2.8',
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
        budget_state=data.get('budget_state'),
        budget_usage=data.get('budget_usage'),
        budget_exhausted_reason=data.get('budget_exhausted_reason'),
        budget_control=data.get('budget_control'),
        sibling_name=data.get('sibling_name'),
        cascade_driver_id=data.get('cascade_driver_id'),
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
        turn_count=data.get('turn_count', 0),
        # Pre-2.3 sessions wrote 'model' instead of 'profile_name'.
        # Old indexes deserialize with profile_name=None; consumers
        # that need the model resolve via the profile registry.
        profile_name=data.get('profile_name'),
        workspace_path=data.get('workspace_path'),
    )
