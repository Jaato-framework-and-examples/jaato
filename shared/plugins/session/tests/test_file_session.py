"""Tests for FileSessionPlugin."""

import json
import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from google.genai import types

from ..file_session import FileSessionPlugin, generate_session_id
from ..base import SessionConfig, SessionState


class TestGenerateSessionId:
    """Tests for session ID generation."""

    def test_format(self):
        """Test that session ID has correct format."""
        session_id = generate_session_id()
        # Should be YYYYMMDD_HHMMSS format
        assert len(session_id) == 15
        assert session_id[8] == '_'
        # Should be parseable
        datetime.strptime(session_id, "%Y%m%d_%H%M%S")


class TestFileSessionPlugin:
    """Tests for FileSessionPlugin."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for session storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def plugin(self, temp_dir):
        """Create a configured FileSessionPlugin."""
        plugin = FileSessionPlugin()
        plugin.initialize({"storage_path": temp_dir})
        return plugin

    def test_initialization(self, plugin, temp_dir):
        """Test plugin initialization."""
        assert plugin.name == "session"
        assert plugin._storage_path == Path(temp_dir)

    def test_save_and_load(self, plugin):
        """Test saving and loading a session."""
        history = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text="Hello")]
            )
        ]

        state = SessionState(
            session_id="test_session",
            history=history,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            description="Test session",
            turn_count=1,
        )

        # Save
        plugin.save(state)

        # Load
        loaded = plugin.load("test_session")

        assert loaded.session_id == "test_session"
        assert loaded.description == "Test session"
        assert loaded.turn_count == 1
        assert len(loaded.history) == 1

    def test_load_nonexistent(self, plugin):
        """Test loading a nonexistent session raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            plugin.load("nonexistent")

    def test_list_sessions_empty(self, plugin):
        """Test listing sessions when none exist."""
        sessions = plugin.list_sessions()
        assert sessions == []

    def test_list_sessions(self, plugin):
        """Test listing multiple sessions."""
        for i in range(3):
            state = SessionState(
                session_id=f"session_{i}",
                history=[],
                created_at=datetime.now(),
                updated_at=datetime.now(),
                turn_count=i,
            )
            plugin.save(state)

        sessions = plugin.list_sessions()

        assert len(sessions) == 3
        # Should be sorted by updated_at descending
        session_ids = [s.session_id for s in sessions]
        assert "session_0" in session_ids
        assert "session_1" in session_ids
        assert "session_2" in session_ids

    def test_delete_session(self, plugin):
        """Test deleting a session."""
        state = SessionState(
            session_id="to_delete",
            history=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        plugin.save(state)

        # Verify it exists
        assert len(plugin.list_sessions()) == 1

        # Delete
        result = plugin.delete("to_delete")
        assert result is True

        # Verify it's gone
        assert len(plugin.list_sessions()) == 0

    def test_delete_nonexistent(self, plugin):
        """Test deleting a nonexistent session returns False."""
        result = plugin.delete("nonexistent")
        assert result is False

    def test_get_latest(self, plugin):
        """Test getting the most recent session."""
        # No sessions yet
        assert plugin.get_latest() is None

        # Add sessions
        for i in range(3):
            state = SessionState(
                session_id=f"session_{i}",
                history=[],
                created_at=datetime.now(),
                updated_at=datetime.now(),
                turn_count=i,
            )
            plugin.save(state)

        latest = plugin.get_latest()
        assert latest is not None
        # The last saved should be "latest" (most recent updated_at)
        assert latest.session_id == "session_2"

    def test_set_description(self, plugin):
        """Test setting session description."""
        state = SessionState(
            session_id="desc_test",
            history=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        plugin.save(state)

        plugin.set_description("desc_test", "New description")

        # Reload and verify
        loaded = plugin.load("desc_test")
        assert loaded.description == "New description"

    def test_get_user_commands(self, plugin):
        """Test that plugin provides user commands."""
        commands = plugin.get_user_commands()

        command_names = [cmd.name for cmd in commands]
        assert "save" in command_names
        assert "resume" in command_names
        assert "sessions" in command_names
        assert "delete-session" in command_names
        assert "backtoturn" in command_names

    def test_get_function_declarations(self, plugin):
        """Test that plugin provides function declarations."""
        declarations = plugin.get_function_declarations()

        assert len(declarations) == 1
        assert declarations[0].name == "session_describe"

    def test_get_executors(self, plugin):
        """Test that plugin provides executors."""
        executors = plugin.get_executors()

        assert "session_describe" in executors
        assert "save" in executors
        assert "resume" in executors
        assert "sessions" in executors
        assert "delete-session" in executors
        assert "backtoturn" in executors

    def test_set_turn_count(self, plugin):
        """Test setting the turn count."""
        plugin._turn_count = 5
        plugin.set_turn_count(3)
        assert plugin._turn_count == 3

    def test_subscribes_to_prompt_enrichment(self, plugin):
        """Test that plugin subscribes to prompt enrichment."""
        assert plugin.subscribes_to_prompt_enrichment() is True

    def test_enrich_prompt_no_description_needed(self, plugin):
        """Test prompt enrichment when description not needed."""
        plugin._turn_count = 1  # Below threshold
        result = plugin.enrich_prompt("Hello")

        assert result.prompt == "Hello"
        assert "description_requested" not in result.metadata

    def test_enrich_prompt_requests_description(self, plugin):
        """Test prompt enrichment requests description after threshold."""
        plugin._turn_count = 3  # At threshold
        plugin._session_description = None
        plugin._description_requested = False

        result = plugin.enrich_prompt("Hello")

        assert "session_describe" in result.prompt
        assert result.metadata.get("description_requested") is True

    def test_enrich_prompt_already_has_description(self, plugin):
        """Test prompt enrichment skips if already has description."""
        plugin._turn_count = 10
        plugin._session_description = "Already named"

        result = plugin.enrich_prompt("Hello")

        assert result.prompt == "Hello"


class TestFileSessionPluginWithClient:
    """Tests for FileSessionPlugin with mock client."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for session storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def plugin_with_client(self, temp_dir):
        """Create a plugin with a mock client."""
        plugin = FileSessionPlugin()
        plugin.initialize({"storage_path": temp_dir})

        # Create mock client
        mock_client = MagicMock()
        mock_client.save_session.return_value = "20251207_143022"
        mock_client.list_sessions.return_value = []

        plugin.set_client(mock_client)
        return plugin, mock_client

    def test_execute_save(self, plugin_with_client):
        """Test executing save command."""
        plugin, mock_client = plugin_with_client

        result = plugin._execute_save({})

        assert result["status"] == "ok"
        assert "20251207_143022" in result["message"]
        mock_client.save_session.assert_called_once()

    def test_execute_sessions_empty(self, plugin_with_client):
        """Test executing sessions command when empty."""
        plugin, mock_client = plugin_with_client

        result = plugin._execute_sessions({})

        assert result["status"] == "info"
        assert "No saved sessions" in result["message"]

    def test_execute_delete_session_missing_id(self, plugin_with_client):
        """Test delete-session without session ID."""
        plugin, _ = plugin_with_client

        result = plugin._execute_delete_session({})

        assert result["status"] == "error"
        assert "Usage" in result["message"]

    def test_execute_backtoturn_no_turn_id(self, plugin_with_client):
        """Test backtoturn without turn ID shows info."""
        plugin, mock_client = plugin_with_client
        mock_client.get_turn_boundaries.return_value = [0, 5, 10]  # 3 turns

        result = plugin._execute_backtoturn({})

        assert result["status"] == "info"
        assert "3 turn(s)" in result["message"]
        assert "Usage" in result["message"]

    def test_execute_backtoturn_invalid_turn_id(self, plugin_with_client):
        """Test backtoturn with non-numeric turn ID."""
        plugin, _ = plugin_with_client

        result = plugin._execute_backtoturn({"turn_id": "abc"})

        assert result["status"] == "error"
        assert "Invalid turn ID" in result["message"]

    def test_execute_backtoturn_success(self, plugin_with_client):
        """Test successful backtoturn execution."""
        plugin, mock_client = plugin_with_client
        mock_client.revert_to_turn.return_value = {
            "success": True,
            "turns_removed": 2,
            "new_turn_count": 3,
            "message": "Reverted to turn 3 (removed 2 turn(s))."
        }

        result = plugin._execute_backtoturn({"turn_id": "3"})

        assert result["status"] == "ok"
        assert "Reverted to turn 3" in result["message"]
        mock_client.revert_to_turn.assert_called_once_with(3)

    def test_execute_backtoturn_invalid_turn(self, plugin_with_client):
        """Test backtoturn with invalid turn ID raises error."""
        plugin, mock_client = plugin_with_client
        mock_client.revert_to_turn.side_effect = ValueError("Turn 10 does not exist.")

        result = plugin._execute_backtoturn({"turn_id": "10"})

        assert result["status"] == "error"
        assert "Turn 10 does not exist" in result["message"]


class TestSessionLifecycle:
    """Tests for session lifecycle hooks."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for session storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def plugin(self, temp_dir):
        """Create a configured FileSessionPlugin."""
        plugin = FileSessionPlugin()
        plugin.initialize({"storage_path": temp_dir})
        return plugin

    def test_on_session_end_skips_empty_session(self, plugin, temp_dir):
        """Test that on_session_end doesn't save empty sessions."""
        # Create an empty session state (0 turns)
        state = SessionState(
            session_id="empty_session",
            history=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            turn_count=0,  # Empty session
        )

        config = SessionConfig(auto_save_on_exit=True)

        # Call on_session_end
        plugin.on_session_end(state, config)

        # Verify no file was created
        session_file = Path(temp_dir) / "empty_session.json"
        assert not session_file.exists()

    def test_on_session_end_saves_non_empty_session(self, plugin, temp_dir):
        """Test that on_session_end saves sessions with content."""
        # Create a session state with content
        state = SessionState(
            session_id="real_session",
            history=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="Hello")]
                )
            ],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            turn_count=1,  # Has content
        )

        config = SessionConfig(auto_save_on_exit=True)

        # Call on_session_end
        plugin.on_session_end(state, config)

        # Verify file was created
        session_file = Path(temp_dir) / "real_session.json"
        assert session_file.exists()

        # Verify content
        with open(session_file) as f:
            data = json.load(f)
        assert data["session_id"] == "real_session"
        assert data["turn_count"] == 1

    def test_on_session_end_respects_auto_save_config(self, plugin, temp_dir):
        """Test that on_session_end respects auto_save_on_exit=False."""
        state = SessionState(
            session_id="no_save_session",
            history=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="Hello")]
                )
            ],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            turn_count=1,
        )

        config = SessionConfig(auto_save_on_exit=False)

        # Call on_session_end
        plugin.on_session_end(state, config)

        # Verify no file was created (auto_save_on_exit=False)
        session_file = Path(temp_dir) / "no_save_session.json"
        assert not session_file.exists()
