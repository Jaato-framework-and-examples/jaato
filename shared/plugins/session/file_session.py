"""File-based session persistence plugin.

This plugin saves sessions to JSON files in a configurable directory,
with support for model-generated session descriptions.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from google.genai import types

from ..base import ToolPlugin, UserCommand, CommandCompletion, PromptEnrichmentResult
from .base import SessionPlugin, SessionConfig, SessionState, SessionInfo
from .serializer import (
    serialize_session_state,
    deserialize_session_state,
    serialize_session_info,
    deserialize_session_info,
)


def generate_session_id() -> str:
    """Generate a timestamp-based session ID.

    Returns:
        Session ID in format YYYYMMDD_HHMMSS.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class FileSessionPlugin:
    """File-based session persistence plugin.

    Implements both SessionPlugin (for persistence) and ToolPlugin (for user
    commands and prompt enrichment).

    Sessions are stored as JSON files named by their timestamp-based ID:
        .jaato/sessions/20251207_143022.json

    The plugin uses prompt enrichment to request a description from the model
    after a configurable number of turns.
    """

    def __init__(self):
        self._name = "session"
        self._storage_path: Path = Path(".jaato/sessions")
        self._config: Optional[Dict[str, Any]] = None

        # Current session state
        self._current_session_id: Optional[str] = None
        self._description_requested: bool = False

        # Track session for prompt enrichment
        self._turn_count: int = 0
        self._session_description: Optional[str] = None

        # Reference to JaatoClient for user command execution
        self._client = None  # Set via set_client()

    @property
    def name(self) -> str:
        return self._name

    def set_client(self, client) -> None:
        """Set the JaatoClient reference for user command execution.

        This is called by JaatoClient.set_session_plugin() to give the plugin
        access to client methods for save/resume operations.

        Args:
            client: The JaatoClient instance.
        """
        self._client = client

    # ==================== Plugin Lifecycle ====================

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the plugin with configuration.

        Args:
            config: Configuration dict. Supports:
                - storage_path: Directory for session files (default: .jaato/sessions)
        """
        self._config = config or {}
        storage = self._config.get('storage_path', '.jaato/sessions')
        self._storage_path = Path(storage)

        # Ensure storage directory exists
        self._storage_path.mkdir(parents=True, exist_ok=True)

        # Reset state
        self._current_session_id = None
        self._description_requested = False
        self._turn_count = 0
        self._session_description = None

    def shutdown(self) -> None:
        """Clean up resources."""
        pass

    # ==================== SessionPlugin: Core Persistence ====================

    def save(self, state: SessionState) -> None:
        """Save session state to a JSON file.

        Args:
            state: The complete session state to persist.
        """
        # Update with current description if we have one
        if self._session_description and not state.description:
            state.description = self._session_description

        file_path = self._storage_path / f"{state.session_id}.json"
        data = serialize_session_state(state)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self._current_session_id = state.session_id

    def load(self, session_id: str) -> SessionState:
        """Load session state from a JSON file.

        Args:
            session_id: The session ID to load.

        Returns:
            The loaded SessionState.

        Raises:
            FileNotFoundError: If the session file doesn't exist.
            ValueError: If the session data is corrupted.
        """
        file_path = self._storage_path / f"{session_id}.json"

        if not file_path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        state = deserialize_session_state(data)

        # Update internal state
        self._current_session_id = state.session_id
        self._session_description = state.description
        self._turn_count = state.turn_count
        self._description_requested = state.description is not None

        return state

    def list_sessions(self) -> List[SessionInfo]:
        """List all available sessions.

        Returns:
            List of SessionInfo objects, sorted by updated_at descending.
        """
        sessions = []

        for file_path in self._storage_path.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                info = deserialize_session_info(data)
                sessions.append(info)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # Skip corrupted files
                print(f"[SessionPlugin] Warning: skipping corrupted session file {file_path}: {e}")
                continue

        # Sort by updated_at descending (most recent first)
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def delete(self, session_id: str) -> bool:
        """Delete a session file.

        Args:
            session_id: The session ID to delete.

        Returns:
            True if deleted, False if session didn't exist.
        """
        file_path = self._storage_path / f"{session_id}.json"

        if file_path.exists():
            file_path.unlink()
            if self._current_session_id == session_id:
                self._current_session_id = None
            return True
        return False

    def get_latest(self) -> Optional[SessionInfo]:
        """Get the most recently updated session.

        Returns:
            SessionInfo for the latest session, or None if no sessions exist.
        """
        sessions = self.list_sessions()
        return sessions[0] if sessions else None

    # ==================== SessionPlugin: Lifecycle Hooks ====================

    def on_turn_complete(
        self,
        state: SessionState,
        config: SessionConfig
    ) -> None:
        """Called after each conversation turn completes.

        Handles checkpoint saves if configured.

        Args:
            state: Current session state.
            config: Session configuration.
        """
        self._turn_count = state.turn_count

        # Checkpoint save if configured
        if config.checkpoint_after_turns:
            if self._turn_count > 0 and self._turn_count % config.checkpoint_after_turns == 0:
                self.save(state)

        # Cleanup old sessions if we have too many
        self._cleanup_old_sessions(config.max_sessions)

    def on_session_start(
        self,
        config: SessionConfig
    ) -> Optional[SessionState]:
        """Called when a new client session starts.

        If auto_resume_last is enabled, returns the last session's state.

        Args:
            config: Session configuration.

        Returns:
            SessionState to restore, or None to start fresh.
        """
        # Reset state for new session
        self._current_session_id = None
        self._description_requested = False
        self._turn_count = 0
        self._session_description = None

        if config.auto_resume_last:
            latest = self.get_latest()
            if latest:
                try:
                    return self.load(latest.session_id)
                except (FileNotFoundError, ValueError):
                    pass

        return None

    def on_session_end(
        self,
        state: SessionState,
        config: SessionConfig
    ) -> None:
        """Called when the client session ends cleanly.

        Saves the session if auto_save_on_exit is enabled and there's
        actual content to save (at least one turn).

        Args:
            state: Current session state.
            config: Session configuration.
        """
        if config.auto_save_on_exit:
            # Don't save empty sessions (no turns = nothing worth saving)
            if state.turn_count == 0:
                return
            # Generate session ID if not set
            if not state.session_id:
                state.session_id = generate_session_id()
            self.save(state)

    # ==================== SessionPlugin: Description Management ====================

    def set_description(self, session_id: str, description: str) -> None:
        """Set the description for a session.

        Called when the model provides a session description via tool call.

        Args:
            session_id: The session ID to update.
            description: The model-generated description.
        """
        self._session_description = description

        # Update the file if it exists
        file_path = self._storage_path / f"{session_id}.json"
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                data['description'] = description
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            except (json.JSONDecodeError, IOError):
                pass

    def needs_description(self, state: SessionState, config: SessionConfig) -> bool:
        """Check if the session needs a description.

        Returns True if:
        - Session has no description
        - Turn count >= config.request_description_after_turns
        - Description hasn't already been requested

        Args:
            state: Current session state.
            config: Session configuration.

        Returns:
            True if description should be requested from model.
        """
        return (
            state.description is None and
            not self._description_requested and
            state.turn_count >= config.request_description_after_turns
        )

    # ==================== ToolPlugin: Function Declarations ====================

    def get_function_declarations(self) -> List[types.FunctionDeclaration]:
        """Return tool declarations for session management.

        Provides session_describe for the model to set session descriptions.
        """
        return [
            types.FunctionDeclaration(
                name="session_describe",
                description=(
                    "Set a brief description for the current conversation session. "
                    "This helps identify the session later when resuming. "
                    "Call this when prompted to describe the session."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": (
                                "A brief 3-5 word description summarizing the main topic "
                                "or goal of this conversation (e.g., 'Debugging auth refresh issue')"
                            )
                        }
                    },
                    "required": ["description"]
                }
            )
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return tool executors for both model tools and user commands."""
        return {
            "session_describe": self._execute_describe,
            # User commands
            "save": self._execute_save,
            "resume": self._execute_resume,
            "sessions": self._execute_sessions,
            "delete-session": self._execute_delete_session,
            "backtoturn": self._execute_backtoturn,
        }

    def _execute_save(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the save user command.

        Saves the current session for later resumption.

        Args:
            args: May include 'user_inputs' list for prompt history restoration.
        """
        if not self._client:
            return {"status": "error", "message": "Session plugin not properly configured"}

        try:
            user_inputs = args.get("user_inputs", [])
            session_id = self._client.save_session(user_inputs=user_inputs)
            return {
                "status": "ok",
                "message": f"Session saved: {session_id}\nUse 'resume' to restore this session later."
            }
        except Exception as e:
            return {"status": "error", "message": f"Error saving session: {e}"}

    def _resolve_session_id(self, session_id: Any) -> Optional[str]:
        """Resolve a session ID, supporting numeric indexes.

        Users can specify sessions by:
        - Full session ID (e.g., "20251207_185349")
        - Numeric index (e.g., 1, 2, 3 or "1", "2", "3") matching the `sessions` list order

        Args:
            session_id: Either a session ID string or a numeric index (1-based)

        Returns:
            The resolved session ID string, or None if not found
        """
        if session_id is None:
            return None

        # Try to parse as numeric index (handles both int and string "1", "2", etc.)
        try:
            index = int(session_id)
            if index >= 1:
                sessions = self._client.list_sessions()
                if 1 <= index <= len(sessions):
                    return sessions[index - 1].session_id
                return None  # Index out of range
        except (ValueError, TypeError):
            pass  # Not a number, treat as session_id string

        # Treat as a direct session ID string
        return str(session_id)

    def _execute_resume(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the resume user command.

        Resumes a previously saved session.

        Returns:
            Result dict including 'user_inputs' for prompt history restoration.
        """
        if not self._client:
            return {"status": "error", "message": "Session plugin not properly configured"}

        # Get session_id from standardized args list
        cmd_args = args.get("args", [])
        raw_session_id = cmd_args[0] if cmd_args else None

        try:
            if raw_session_id is not None:
                # Resolve numeric index to actual session ID
                session_id = self._resolve_session_id(raw_session_id)
                if session_id is None:
                    return {"status": "error", "message": f"Session not found: {raw_session_id}"}

                # Resume specific session
                state = self._client.resume_session(session_id)
                desc = f' - "{state.description}"' if state.description else ''
                return {
                    "status": "ok",
                    "message": f"Session resumed: {state.session_id}{desc}\nRestored {state.turn_count} turns.",
                    "user_inputs": state.user_inputs,  # For prompt history restoration
                }
            else:
                # List sessions for user to choose
                sessions = self._client.list_sessions()
                if not sessions:
                    return {"status": "info", "message": "No saved sessions found"}

                lines = ["Available sessions:"]
                for i, s in enumerate(sessions, 1):
                    desc = f' - "{s.description}"' if s.description else ' (unnamed)'
                    lines.append(f"  {i}. {s.session_id}{desc} ({s.turn_count} turns)")
                lines.append("\nUse 'resume <session_id>' to restore a specific session.")

                return {"status": "info", "message": "\n".join(lines)}

        except FileNotFoundError:
            return {"status": "error", "message": f"Session not found: {session_id}"}
        except Exception as e:
            return {"status": "error", "message": f"Error resuming session: {e}"}

    def _execute_sessions(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the sessions user command.

        Lists all available saved sessions.
        """
        if not self._client:
            return {"status": "error", "message": "Session plugin not properly configured"}

        try:
            sessions = self._client.list_sessions()

            if not sessions:
                return {"status": "info", "message": "No saved sessions found"}

            lines = ["=" * 60, "  Saved Sessions", "=" * 60]

            for i, s in enumerate(sessions, 1):
                desc = f'"{s.description}"' if s.description else '(unnamed)'
                model = f" [{s.model}]" if s.model else ""
                updated = s.updated_at.strftime("%Y-%m-%d %H:%M")
                lines.append(f"  {i}. {s.session_id}")
                lines.append(f"     {desc}{model}")
                lines.append(f"     {s.turn_count} turns, last updated {updated}")
                lines.append("")

            lines.append("=" * 60)
            lines.append("Use 'resume <session_id>' to restore a session.")
            lines.append("Use 'delete-session <session_id or index>' to delete a session.")

            return {"status": "ok", "message": "\n".join(lines)}

        except Exception as e:
            return {"status": "error", "message": f"Error listing sessions: {e}"}

    def _execute_delete_session(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the delete-session user command.

        Deletes a saved session. Supports both full session IDs and numeric indexes.
        """
        if not self._client:
            return {"status": "error", "message": "Session plugin not properly configured"}

        # Get session_id from standardized args list
        cmd_args = args.get("args", [])
        raw_session_id = cmd_args[0] if cmd_args else None
        if raw_session_id is None:
            return {"status": "error", "message": "Usage: delete-session <session_id or index>"}

        # Resolve numeric index to actual session ID
        session_id = self._resolve_session_id(raw_session_id)
        if session_id is None:
            return {"status": "error", "message": f"Session not found: {raw_session_id}"}

        try:
            if self._client.delete_session(session_id):
                return {"status": "ok", "message": f"Session deleted: {session_id}"}
            else:
                return {"status": "error", "message": f"Session not found: {session_id}"}
        except Exception as e:
            return {"status": "error", "message": f"Error deleting session: {e}"}

    def _execute_backtoturn(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the backtoturn user command.

        Reverts the conversation to a specific turn.
        """
        if not self._client:
            return {"status": "error", "message": "Session plugin not properly configured"}

        # Get turn_id from standardized args list
        cmd_args = args.get("args", [])
        turn_id = cmd_args[0] if cmd_args else None
        if turn_id is None:
            # Show current turn count and usage
            try:
                boundaries = self._client.get_turn_boundaries()
                total_turns = len(boundaries)
                return {
                    "status": "info",
                    "message": (
                        f"Current session has {total_turns} turn(s).\n"
                        "Usage: backtoturn <turn_id>\n\n"
                        "Use 'history' to see turn IDs, then specify which turn to revert to.\n"
                        "All turns after the specified turn will be removed."
                    )
                }
            except Exception as e:
                return {"status": "error", "message": f"Error: {e}"}

        try:
            turn_id = int(turn_id)
        except (ValueError, TypeError):
            return {"status": "error", "message": f"Invalid turn ID: {turn_id}. Must be a number."}

        try:
            result = self._client.revert_to_turn(turn_id)
            return {
                "status": "ok",
                "message": result.get("message", f"Reverted to turn {turn_id}")
            }
        except ValueError as e:
            return {"status": "error", "message": str(e)}
        except Exception as e:
            return {"status": "error", "message": f"Error reverting: {e}"}

    def _execute_describe(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the session_describe tool.

        When a description is first set, automatically save the session.
        This creates a natural "commitment" point for persistence.

        Args:
            args: Tool arguments with 'description' key.

        Returns:
            Status response.
        """
        description = args.get("description", "").strip()

        if not description:
            return {"status": "error", "message": "Description cannot be empty"}

        is_first_description = self._session_description is None
        self._session_description = description
        self._description_requested = True

        # Update file if we have a current session
        if self._current_session_id:
            self.set_description(self._current_session_id, description)

        # Auto-save when first description is set (natural commitment point)
        if is_first_description and self._client:
            try:
                session_id = self._client.save_session()
                self._current_session_id = session_id
            except Exception:
                pass  # Don't fail the describe call if save fails

        return {
            "status": "ok",
            "message": f"Session description set: {description}"
        }

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions about session management."""
        return None  # Instructions are injected via prompt enrichment when needed

    def get_auto_approved_tools(self) -> List[str]:
        """Return tools that should be auto-approved without permission prompts.

        Includes both the model tool (session_describe) and user commands
        (save, resume, sessions, delete-session). User commands are invoked
        directly by the user, so they should never require permission prompts.
        """
        return [
            "session_describe",
            # User commands - these are invoked directly by the user
            "save",
            "resume",
            "sessions",
            "delete-session",
        ]

    def get_user_commands(self) -> List[UserCommand]:
        """Return user-facing commands for session management."""
        return [
            UserCommand("save", "Save the current session", share_with_model=False),
            UserCommand("resume", "Resume a saved session", share_with_model=False),
            UserCommand("sessions", "List available sessions", share_with_model=False),
            UserCommand("delete-session", "Delete a saved session", share_with_model=False),
            UserCommand("backtoturn", "Revert to a specific turn (use 'history' to see turn IDs)", share_with_model=False),
        ]

    def get_command_completions(
        self, command: str, args: List[str]
    ) -> List[CommandCompletion]:
        """Return completion options for session command arguments.

        Session commands don't have subcommands - their argument completion
        (session IDs) is handled by the client's SessionIdCompleter which
        queries the plugin's list() method directly.
        """
        # Session commands take session IDs or turn IDs as arguments,
        # not subcommands. Argument completion is handled separately.
        return []

    # ==================== ToolPlugin: Prompt Enrichment ====================

    def subscribes_to_prompt_enrichment(self) -> bool:
        """Subscribe to prompt enrichment for session description requests."""
        return True

    def enrich_prompt(self, prompt: str) -> PromptEnrichmentResult:
        """Enrich prompt with session description request if needed.

        After enough turns and if no description exists, inject a hint
        for the model to provide a session description.

        Args:
            prompt: The user's original prompt.

        Returns:
            PromptEnrichmentResult with possibly enriched prompt.
        """
        # Check if we need to request a description
        # We use internal state since we don't have access to full config here
        if (
            self._session_description is None and
            not self._description_requested and
            self._turn_count >= 3  # Default threshold
        ):
            self._description_requested = True

            hint = (
                "\n\n[System: This conversation has been ongoing for a while. "
                "Please provide a brief 3-5 word description summarizing its main topic "
                "by calling the session_describe tool. This is for session management only "
                "and won't interrupt the conversation flow.]"
            )

            return PromptEnrichmentResult(
                prompt=prompt + hint,
                metadata={"description_requested": True}
            )

        return PromptEnrichmentResult(prompt=prompt)

    # ==================== Internal Helpers ====================

    def _cleanup_old_sessions(self, max_sessions: int) -> int:
        """Remove oldest sessions if we exceed the limit.

        Args:
            max_sessions: Maximum number of sessions to keep.

        Returns:
            Number of sessions deleted.
        """
        sessions = self.list_sessions()

        if len(sessions) <= max_sessions:
            return 0

        # Delete oldest sessions (they're sorted by updated_at desc)
        to_delete = sessions[max_sessions:]
        deleted = 0

        for session in to_delete:
            if self.delete(session.session_id):
                deleted += 1

        return deleted

    def get_current_session_id(self) -> Optional[str]:
        """Get the current session ID.

        Returns:
            Current session ID or None if no session is active.
        """
        return self._current_session_id

    def set_current_session_id(self, session_id: str) -> None:
        """Set the current session ID.

        Args:
            session_id: The session ID to set as current.
        """
        self._current_session_id = session_id

    def increment_turn_count(self) -> None:
        """Increment the internal turn count for prompt enrichment tracking."""
        self._turn_count += 1

    def set_turn_count(self, count: int) -> None:
        """Set the internal turn count.

        Used when reverting to a previous turn to keep the count in sync.

        Args:
            count: The new turn count.
        """
        self._turn_count = count
