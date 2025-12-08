"""Serialization utilities for session persistence.

This module handles converting internal types (Message, Part) to and
from JSON-serializable dictionaries for storage.
"""

import base64
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..model_provider.types import (
    Message,
    Part,
    Role,
    FunctionCall,
    ToolResult,
)
from .base import SessionState, SessionInfo


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
            'is_error': fr.is_error
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
            is_error=data.get('is_error', False)
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

    Args:
        message: A Message object.

    Returns:
        Dictionary representation of the message.
    """
    return {
        'role': message.role.value,
        'parts': [serialize_part(p) for p in message.parts]
    }


def deserialize_message(data: Dict[str, Any]) -> Message:
    """Deserialize a dictionary to a Message object.

    Args:
        data: Dictionary representation of message.

    Returns:
        Reconstructed Message object.
    """
    parts = [deserialize_part(p) for p in data.get('parts', [])]
    return Message(role=Role(data['role']), parts=parts)


def serialize_history(history: List[Message]) -> List[Dict[str, Any]]:
    """Serialize a conversation history to a list of dictionaries.

    Args:
        history: List of Message objects.

    Returns:
        List of dictionary representations.
    """
    return [serialize_message(m) for m in history]


def deserialize_history(data: List[Dict[str, Any]]) -> List[Message]:
    """Deserialize a list of dictionaries to conversation history.

    Args:
        data: List of dictionary representations.

    Returns:
        List of Message objects.
    """
    return [deserialize_message(d) for d in data]


def serialize_session_state(state: SessionState) -> Dict[str, Any]:
    """Serialize a SessionState to a JSON-compatible dictionary.

    Args:
        state: The SessionState to serialize.

    Returns:
        JSON-compatible dictionary.
    """
    return {
        'version': '2.0',  # Bumped for Message type support
        'session_id': state.session_id,
        'description': state.description,
        'created_at': state.created_at.isoformat(),
        'updated_at': state.updated_at.isoformat(),
        'turn_count': state.turn_count,
        'turn_accounting': state.turn_accounting,
        'user_inputs': state.user_inputs,
        'metadata': state.metadata,
        'connection': {
            'project': state.project,
            'location': state.location,
            'model': state.model,
        },
        'history': serialize_history(state.history),
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
    # Support both 1.x (legacy) and 2.x (new Message type) versions
    if not (version.startswith('1.') or version.startswith('2.')):
        raise ValueError(f"Unsupported session version: {version}")

    connection = data.get('connection', {})

    return SessionState(
        session_id=data['session_id'],
        history=deserialize_history(data.get('history', [])),
        created_at=datetime.fromisoformat(data['created_at']),
        updated_at=datetime.fromisoformat(data['updated_at']),
        description=data.get('description'),
        turn_count=data.get('turn_count', 0),
        turn_accounting=data.get('turn_accounting', []),
        user_inputs=data.get('user_inputs', []),
        metadata=data.get('metadata', {}),
        project=connection.get('project'),
        location=connection.get('location'),
        model=connection.get('model'),
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
        'model': state.model,
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
        created_at=datetime.fromisoformat(data['created_at']),
        updated_at=datetime.fromisoformat(data['updated_at']),
        turn_count=data.get('turn_count', 0),
        model=data.get('model'),
    )
