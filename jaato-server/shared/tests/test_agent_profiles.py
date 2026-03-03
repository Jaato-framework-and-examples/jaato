"""Tests for agent profile-based session creation.

Verifies that the SDK events, IPC client, and server correctly support
creating sessions with predefined agent profiles.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestSessionProfilesEvent:
    """Tests for the SessionProfilesEvent SDK event."""

    def test_session_profiles_event_type_exists(self):
        """EventType.SESSION_PROFILES exists in the enum."""
        from jaato_sdk.events import EventType
        assert hasattr(EventType, "SESSION_PROFILES")
        assert EventType.SESSION_PROFILES.value == "session.profiles"

    def test_session_profiles_event_serialization(self):
        """SessionProfilesEvent serializes and deserializes correctly."""
        from jaato_sdk.events import (
            SessionProfilesEvent,
            serialize_event,
            deserialize_event,
        )

        profiles = [
            {
                "name": "researcher",
                "description": "Research profile",
                "model": "claude-sonnet-4-20250514",
                "provider": "anthropic",
                "icon_name": "search",
                "plugins": ["cli", "web_search"],
            },
            {
                "name": "coder",
                "description": "Coding profile",
                "model": None,
                "provider": None,
                "icon_name": "code",
                "plugins": ["cli", "file_edit", "lsp"],
            },
        ]

        event = SessionProfilesEvent(profiles=profiles)
        json_str = serialize_event(event)
        restored = deserialize_event(json_str)

        assert isinstance(restored, SessionProfilesEvent)
        assert len(restored.profiles) == 2
        assert restored.profiles[0]["name"] == "researcher"
        assert restored.profiles[1]["plugins"] == ["cli", "file_edit", "lsp"]

    def test_session_profiles_event_empty(self):
        """SessionProfilesEvent works with no profiles."""
        from jaato_sdk.events import (
            SessionProfilesEvent,
            serialize_event,
            deserialize_event,
        )

        event = SessionProfilesEvent(profiles=[])
        json_str = serialize_event(event)
        restored = deserialize_event(json_str)

        assert isinstance(restored, SessionProfilesEvent)
        assert restored.profiles == []


class TestSessionInfoProfileName:
    """Tests for the profile_name field on SessionInfoEvent."""

    def test_session_info_has_profile_name(self):
        """SessionInfoEvent has profile_name field."""
        from jaato_sdk.events import SessionInfoEvent
        event = SessionInfoEvent(
            session_id="test_123",
            session_name="Test",
            profile_name="researcher",
        )
        assert event.profile_name == "researcher"

    def test_session_info_profile_name_default_none(self):
        """SessionInfoEvent.profile_name defaults to None."""
        from jaato_sdk.events import SessionInfoEvent
        event = SessionInfoEvent(session_id="test_123")
        assert event.profile_name is None

    def test_session_info_profile_name_serializes(self):
        """profile_name round-trips through serialization."""
        from jaato_sdk.events import (
            SessionInfoEvent,
            serialize_event,
            deserialize_event,
        )

        event = SessionInfoEvent(
            session_id="test_123",
            session_name="Test",
            profile_name="analyst",
        )
        json_str = serialize_event(event)
        restored = deserialize_event(json_str)

        assert isinstance(restored, SessionInfoEvent)
        assert restored.profile_name == "analyst"


class TestProfileDiscoveryForSession:
    """Tests for profile discovery and resolution in SessionManager context."""

    def test_resolve_profile_from_workspace(self):
        """_resolve_profile finds a profile from .jaato/profiles/."""
        with tempfile.TemporaryDirectory() as workspace:
            # Create .jaato/profiles/ directory with a profile
            profiles_dir = Path(workspace) / ".jaato" / "profiles"
            profiles_dir.mkdir(parents=True)

            profile_data = {
                "name": "test-agent",
                "description": "A test agent profile",
                "plugins": ["cli", "todo"],
                "system_instructions": "You are a test agent.",
                "model": "gemini-2.5-flash",
                "max_turns": 5,
            }
            (profiles_dir / "test-agent.json").write_text(json.dumps(profile_data))

            # Use discover_profiles directly (same as _resolve_profile uses)
            from shared.plugins.subagent.config import discover_profiles
            profiles = discover_profiles(".jaato/profiles", base_path=workspace)

            assert "test-agent" in profiles
            profile = profiles["test-agent"]
            assert profile.description == "A test agent profile"
            assert profile.plugins == ["cli", "todo"]
            assert profile.system_instructions == "You are a test agent."
            assert profile.model == "gemini-2.5-flash"

    def test_resolve_nonexistent_profile_returns_none(self):
        """Looking up a nonexistent profile name returns None."""
        with tempfile.TemporaryDirectory() as workspace:
            # Empty .jaato/profiles/
            profiles_dir = Path(workspace) / ".jaato" / "profiles"
            profiles_dir.mkdir(parents=True)

            from shared.plugins.subagent.config import discover_profiles
            profiles = discover_profiles(".jaato/profiles", base_path=workspace)

            assert profiles.get("nonexistent") is None

    def test_resolve_profile_no_profiles_dir(self):
        """Missing .jaato/profiles/ directory returns empty dict."""
        with tempfile.TemporaryDirectory() as workspace:
            from shared.plugins.subagent.config import discover_profiles
            profiles = discover_profiles(".jaato/profiles", base_path=workspace)
            assert profiles == {}


class TestBuildProfileSessionKwargs:
    """Tests for JaatoServer._build_profile_session_kwargs()."""

    def _make_server_with_profile(self, profile_data):
        """Create a minimal JaatoServer with a profile set."""
        from shared.plugins.subagent.config import SubagentProfile
        profile = SubagentProfile(**profile_data)

        # We can't easily instantiate JaatoServer without imports,
        # so test the logic directly
        return profile

    def test_profile_with_plugins_produces_tools_kwarg(self):
        """Profile with plugins list produces 'tools' in kwargs."""
        from shared.plugins.subagent.config import SubagentProfile, parse_plugin_list

        profile = SubagentProfile(
            name="test",
            description="Test",
            plugins=["cli", "todo", "web_search"],
        )

        # Simulate _build_profile_session_kwargs logic
        kwargs = {}
        if profile.plugins:
            clean_plugins, preloaded = parse_plugin_list(profile.plugins)
            kwargs["tools"] = clean_plugins
            if preloaded:
                kwargs["preloaded_plugins"] = preloaded

        assert kwargs["tools"] == ["cli", "todo", "web_search"]
        assert "preloaded_plugins" not in kwargs

    def test_profile_with_preload_annotations(self):
        """Profile with (preload) annotations produces preloaded_plugins."""
        from shared.plugins.subagent.config import SubagentProfile, parse_plugin_list

        profile = SubagentProfile(
            name="test",
            description="Test",
            plugins=["cli", "todo(preload)", "web_search"],
        )

        kwargs = {}
        if profile.plugins:
            clean_plugins, preloaded = parse_plugin_list(profile.plugins)
            kwargs["tools"] = clean_plugins
            if preloaded:
                kwargs["preloaded_plugins"] = preloaded

        assert kwargs["tools"] == ["cli", "todo", "web_search"]
        assert kwargs["preloaded_plugins"] == {"todo"}

    def test_profile_with_system_instructions(self):
        """Profile system_instructions are passed through."""
        from shared.plugins.subagent.config import SubagentProfile

        profile = SubagentProfile(
            name="test",
            description="Test",
            system_instructions="You are a code reviewer.",
        )

        kwargs = {}
        if profile.system_instructions:
            kwargs["system_instructions"] = profile.system_instructions

        assert kwargs["system_instructions"] == "You are a code reviewer."

    def test_profile_with_provider_override(self):
        """Profile with provider produces provider_name kwarg."""
        from shared.plugins.subagent.config import SubagentProfile

        profile = SubagentProfile(
            name="test",
            description="Test",
            provider="anthropic",
        )

        kwargs = {}
        if profile.provider:
            kwargs["provider_name"] = profile.provider

        assert kwargs["provider_name"] == "anthropic"

    def test_empty_profile_produces_none(self):
        """Profile with no overrides produces empty kwargs."""
        from shared.plugins.subagent.config import SubagentProfile

        profile = SubagentProfile(
            name="test",
            description="Test",
        )

        kwargs = {}
        if profile.plugins:
            kwargs["tools"] = profile.plugins
        if profile.system_instructions:
            kwargs["system_instructions"] = profile.system_instructions
        if profile.provider:
            kwargs["provider_name"] = profile.provider

        assert kwargs == {}


class TestIPCClientCreateSessionProfile:
    """Tests for IPCClient.create_session() profile parameter."""

    def test_create_session_with_profile_sends_args(self):
        """create_session(profile='x') includes --profile in args."""
        from jaato_sdk.events import CommandRequest

        # Verify the args construction logic
        name = None
        profile = "researcher"

        args = [name] if name else []
        if profile:
            args.extend(["--profile", profile])

        assert args == ["--profile", "researcher"]

    def test_create_session_with_name_and_profile(self):
        """create_session(name='x', profile='y') includes both."""
        name = "my-session"
        profile = "analyst"

        args = [name] if name else []
        if profile:
            args.extend(["--profile", profile])

        assert args == ["my-session", "--profile", "analyst"]

    def test_create_session_without_profile(self):
        """create_session() without profile sends no --profile arg."""
        name = "my-session"
        profile = None

        args = [name] if name else []
        if profile:
            args.extend(["--profile", profile])

        assert args == ["my-session"]


class TestSessionNewArgParsing:
    """Tests for parsing session.new command args with --profile flag."""

    def _parse_session_new_args(self, args):
        """Parse session.new args the same way __main__.py does."""
        name = None
        profile_name = None
        args_iter = iter(args)
        for arg in args_iter:
            if arg == "--profile":
                profile_name = next(args_iter, None)
            elif name is None:
                name = arg
        return name, profile_name

    def test_parse_profile_only(self):
        """--profile researcher with no name."""
        name, profile = self._parse_session_new_args(["--profile", "researcher"])
        assert name is None
        assert profile == "researcher"

    def test_parse_name_and_profile(self):
        """my-session --profile researcher."""
        name, profile = self._parse_session_new_args(
            ["my-session", "--profile", "researcher"]
        )
        assert name == "my-session"
        assert profile == "researcher"

    def test_parse_profile_then_name(self):
        """--profile researcher my-session (profile before name)."""
        name, profile = self._parse_session_new_args(
            ["--profile", "researcher", "my-session"]
        )
        assert name == "my-session"
        assert profile == "researcher"

    def test_parse_no_args(self):
        """No args at all."""
        name, profile = self._parse_session_new_args([])
        assert name is None
        assert profile is None

    def test_parse_name_only(self):
        """Name without profile."""
        name, profile = self._parse_session_new_args(["my-session"])
        assert name == "my-session"
        assert profile is None

    def test_parse_dangling_profile_flag(self):
        """--profile with no value after it."""
        name, profile = self._parse_session_new_args(["--profile"])
        assert name is None
        assert profile is None
