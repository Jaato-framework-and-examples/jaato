"""Serialization utilities for subagent persistence.

This module handles converting subagent state (active sessions, profiles) to and
from JSON-serializable dictionaries for storage.

The persistence strategy mirrors the main session approach:
- Lightweight registry in SessionState.metadata["subagents"]
- Full state in dedicated files: .jaato/sessions/{session_id}/subagents/{agent_id}.json
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from ..session.serializer import serialize_history, deserialize_history


def serialize_subagent_state(session_info: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize a subagent session's full state for per-agent file storage.

    This produces the complete state needed to restore a subagent session,
    stored in .jaato/sessions/{session_id}/subagents/{agent_id}.json.

    Args:
        session_info: The session info dict from SubagentPlugin._active_sessions.
            Expected keys: session, profile, agent_id, created_at, last_activity,
            turn_count, max_turns

    Returns:
        JSON-serializable dict with full subagent state.
    """
    session = session_info.get('session')
    profile = session_info.get('profile')

    # Extract session data
    history = []
    turn_accounting = []
    if session:
        history = session.get_history()
        turn_accounting = session.get_turn_accounting()

    # Serialize profile
    profile_data = {}
    if profile:
        profile_data = {
            'name': profile.name,
            'description': profile.description,
            'plugins': profile.plugins,
            'plugin_configs': profile.plugin_configs,
            'system_instructions': profile.system_instructions,
            'model': profile.model,
            'provider': profile.provider,
            'max_turns': profile.max_turns,
            'auto_approved': profile.auto_approved,
            'icon': profile.icon,
            'icon_name': profile.icon_name,
        }
        # Serialize GC config if present
        if profile.gc:
            profile_data['gc'] = {
                'type': profile.gc.type,
                'threshold_percent': profile.gc.threshold_percent,
                'preserve_recent_turns': profile.gc.preserve_recent_turns,
                'notify_on_gc': profile.gc.notify_on_gc,
                'summarize_middle_turns': profile.gc.summarize_middle_turns,
                'max_turns': profile.gc.max_turns,
                'plugin_config': profile.gc.plugin_config,
            }

    # Convert datetimes to ISO strings
    created_at = session_info.get('created_at')
    if isinstance(created_at, datetime):
        created_at = created_at.isoformat()

    last_activity = session_info.get('last_activity')
    if isinstance(last_activity, datetime):
        last_activity = last_activity.isoformat()

    return {
        'version': '1.0',
        'agent_id': session_info.get('agent_id', ''),
        'profile': profile_data,
        'history': serialize_history(history),
        'turn_accounting': turn_accounting,
        'created_at': created_at,
        'last_activity': last_activity,
        'turn_count': session_info.get('turn_count', 0),
        'max_turns': session_info.get('max_turns', 10),
        'metadata': {},  # Reserved for future use
    }


def deserialize_subagent_state(data: Dict[str, Any]) -> Dict[str, Any]:
    """Deserialize a subagent session's full state from storage.

    Args:
        data: Dictionary from JSON file.

    Returns:
        Reconstructed session info suitable for SubagentPlugin._active_sessions.
        Note: The 'session' key will be None - actual session must be recreated
        by the caller using the profile and history.

    Raises:
        ValueError: If required fields are missing or version is incompatible.
    """
    version = data.get('version', '1.0')
    if not version.startswith('1.'):
        raise ValueError(f"Unsupported subagent state version: {version}")

    # Deserialize profile
    profile_data = data.get('profile', {})
    profile = None
    if profile_data:
        from .config import SubagentProfile, GCProfileConfig

        gc_config = None
        if profile_data.get('gc'):
            gc_data = profile_data['gc']
            gc_config = GCProfileConfig(
                type=gc_data.get('type', 'truncate'),
                threshold_percent=gc_data.get('threshold_percent', 80.0),
                preserve_recent_turns=gc_data.get('preserve_recent_turns', 5),
                notify_on_gc=gc_data.get('notify_on_gc', True),
                summarize_middle_turns=gc_data.get('summarize_middle_turns'),
                max_turns=gc_data.get('max_turns'),
                plugin_config=gc_data.get('plugin_config', {}),
            )

        profile = SubagentProfile(
            name=profile_data.get('name', ''),
            description=profile_data.get('description', ''),
            plugins=profile_data.get('plugins', []),
            plugin_configs=profile_data.get('plugin_configs', {}),
            system_instructions=profile_data.get('system_instructions'),
            model=profile_data.get('model'),
            provider=profile_data.get('provider'),
            max_turns=profile_data.get('max_turns', 10),
            auto_approved=profile_data.get('auto_approved', False),
            icon=profile_data.get('icon'),
            icon_name=profile_data.get('icon_name'),
            gc=gc_config,
        )

    # Parse datetimes
    created_at = data.get('created_at')
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)

    last_activity = data.get('last_activity')
    if isinstance(last_activity, str):
        last_activity = datetime.fromisoformat(last_activity)

    return {
        'agent_id': data.get('agent_id', ''),
        'profile': profile,
        'history': deserialize_history(data.get('history', [])),
        'turn_accounting': data.get('turn_accounting', []),
        'created_at': created_at,
        'last_activity': last_activity,
        'turn_count': data.get('turn_count', 0),
        'max_turns': data.get('max_turns', 10),
        'session': None,  # Must be recreated by caller
    }


def serialize_subagent_registry(
    active_sessions: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Serialize active subagents to a lightweight registry for SessionState.metadata.

    This produces a compact index suitable for storing in SessionState.metadata["subagents"].
    The full state for each subagent is stored separately in per-agent files.

    Args:
        active_sessions: The SubagentPlugin._active_sessions dict.

    Returns:
        JSON-serializable registry dict with version and agent list.
    """
    agents = []
    for agent_id, info in active_sessions.items():
        session = info.get('session')
        profile = info.get('profile')

        # Determine status based on session state
        status = 'idle'
        if session:
            if session.is_running:
                status = 'running'

        # Convert datetimes
        created_at = info.get('created_at')
        if isinstance(created_at, datetime):
            created_at = created_at.isoformat()

        last_activity = info.get('last_activity')
        if isinstance(last_activity, datetime):
            last_activity = last_activity.isoformat()

        agents.append({
            'agent_id': agent_id,
            'profile_name': profile.name if profile else '',
            'status': status,
            'created_at': created_at,
            'last_activity': last_activity,
            'turn_count': info.get('turn_count', 0),
            'max_turns': info.get('max_turns', 10),
        })

    return {
        'version': '1.0',
        'agents': agents,
    }


def deserialize_subagent_registry(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Deserialize subagent registry from SessionState.metadata.

    Args:
        data: Registry dict from metadata["subagents"].

    Returns:
        List of agent info dicts with parsed datetimes.
        Each dict has: agent_id, profile_name, status, created_at, last_activity,
        turn_count, max_turns.

    Raises:
        ValueError: If version is incompatible.
    """
    version = data.get('version', '1.0')
    if not version.startswith('1.'):
        raise ValueError(f"Unsupported subagent registry version: {version}")

    result = []
    for agent in data.get('agents', []):
        # Parse datetimes
        created_at = agent.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        last_activity = agent.get('last_activity')
        if isinstance(last_activity, str):
            last_activity = datetime.fromisoformat(last_activity)

        result.append({
            'agent_id': agent.get('agent_id', ''),
            'profile_name': agent.get('profile_name', ''),
            'status': agent.get('status', 'idle'),
            'created_at': created_at,
            'last_activity': last_activity,
            'turn_count': agent.get('turn_count', 0),
            'max_turns': agent.get('max_turns', 10),
        })

    return result
