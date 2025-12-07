"""Serialization utilities for session persistence.

This module handles converting Google genai SDK types (Content, Part) to and
from JSON-serializable dictionaries for storage.
"""

import base64
from datetime import datetime
from typing import Any, Dict, List, Optional

from google.genai import types

from .base import SessionState, SessionInfo


def serialize_part(part: types.Part) -> Dict[str, Any]:
    """Serialize a Part object to a dictionary.

    Handles text, function calls, function responses, and inline data.

    Args:
        part: A types.Part object.

    Returns:
        Dictionary representation of the part.
    """
    # Text part
    if hasattr(part, 'text') and part.text is not None:
        return {
            'type': 'text',
            'text': part.text
        }

    # Function call part
    if hasattr(part, 'function_call') and part.function_call is not None:
        fc = part.function_call
        return {
            'type': 'function_call',
            'name': fc.name,
            'args': dict(fc.args) if fc.args else {}
        }

    # Function response part
    if hasattr(part, 'function_response') and part.function_response is not None:
        fr = part.function_response
        # response can be a dict or other serializable type
        response = fr.response
        if hasattr(response, 'items'):
            response = dict(response)
        return {
            'type': 'function_response',
            'name': fr.name,
            'response': response
        }

    # Inline data (images, etc.)
    if hasattr(part, 'inline_data') and part.inline_data is not None:
        inline = part.inline_data
        return {
            'type': 'inline_data',
            'mime_type': inline.mime_type,
            'data': base64.b64encode(inline.data).decode('utf-8') if inline.data else None
        }

    # Unknown part type - try to capture what we can
    return {
        'type': 'unknown',
        'repr': repr(part)
    }


def deserialize_part(data: Dict[str, Any]) -> types.Part:
    """Deserialize a dictionary to a Part object.

    Args:
        data: Dictionary representation of a part.

    Returns:
        Reconstructed types.Part object.

    Raises:
        ValueError: If the part type is not recognized.
    """
    part_type = data.get('type')

    if part_type == 'text':
        return types.Part.from_text(text=data['text'])

    if part_type == 'function_call':
        # Function calls are typically only in model responses,
        # and we usually don't need to reconstruct them for history replay.
        # But we include them for completeness.
        return types.Part(
            function_call=types.FunctionCall(
                name=data['name'],
                args=data.get('args', {})
            )
        )

    if part_type == 'function_response':
        return types.Part.from_function_response(
            name=data['name'],
            response=data['response']
        )

    if part_type == 'inline_data':
        raw_data = None
        if data.get('data'):
            raw_data = base64.b64decode(data['data'])
        return types.Part(
            inline_data=types.Blob(
                mime_type=data['mime_type'],
                data=raw_data
            )
        )

    if part_type == 'unknown':
        # Best effort - create a text part with the repr
        return types.Part.from_text(text=f"[Unrecognized part: {data.get('repr', '?')}]")

    raise ValueError(f"Unknown part type: {part_type}")


def serialize_content(content: types.Content) -> Dict[str, Any]:
    """Serialize a Content object to a dictionary.

    Args:
        content: A types.Content object.

    Returns:
        Dictionary representation of the content.
    """
    return {
        'role': content.role,
        'parts': [serialize_part(p) for p in (content.parts or [])]
    }


def deserialize_content(data: Dict[str, Any]) -> types.Content:
    """Deserialize a dictionary to a Content object.

    Args:
        data: Dictionary representation of content.

    Returns:
        Reconstructed types.Content object.
    """
    parts = [deserialize_part(p) for p in data.get('parts', [])]
    return types.Content(role=data['role'], parts=parts)


def serialize_history(history: List[types.Content]) -> List[Dict[str, Any]]:
    """Serialize a conversation history to a list of dictionaries.

    Args:
        history: List of types.Content objects.

    Returns:
        List of dictionary representations.
    """
    return [serialize_content(c) for c in history]


def deserialize_history(data: List[Dict[str, Any]]) -> List[types.Content]:
    """Deserialize a list of dictionaries to conversation history.

    Args:
        data: List of dictionary representations.

    Returns:
        List of types.Content objects.
    """
    return [deserialize_content(c) for c in data]


def serialize_session_state(state: SessionState) -> Dict[str, Any]:
    """Serialize a SessionState to a JSON-compatible dictionary.

    Args:
        state: The SessionState to serialize.

    Returns:
        JSON-compatible dictionary.
    """
    return {
        'version': '1.0',
        'session_id': state.session_id,
        'description': state.description,
        'created_at': state.created_at.isoformat(),
        'updated_at': state.updated_at.isoformat(),
        'turn_count': state.turn_count,
        'turn_accounting': state.turn_accounting,
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
    if not version.startswith('1.'):
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
