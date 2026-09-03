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
            ProfileSummary,
            SessionProfilesEvent,
            serialize_event,
            deserialize_event,
        )

        profiles = [
            ProfileSummary(
                name="researcher",
                description="Research profile",
                model="claude-sonnet-4-20250514",
                provider="anthropic",
                plugins=["cli", "web_search"],
            ),
            ProfileSummary(
                name="coder",
                description="Coding profile",
                plugins=["cli", "file_edit", "lsp"],
            ),
        ]

        event = SessionProfilesEvent(profiles=profiles)
        json_str = serialize_event(event)
        restored = deserialize_event(json_str)

        assert isinstance(restored, SessionProfilesEvent)
        assert len(restored.profiles) == 2
        assert restored.profiles[0].name == "researcher"
        assert restored.profiles[0].model == "claude-sonnet-4-20250514"
        assert restored.profiles[1].plugins == ["cli", "file_edit", "lsp"]
        # New: parse_errors is its own field, defaults empty
        assert restored.parse_errors == []

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
        assert restored.parse_errors == []


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
            result = discover_profiles(".jaato/profiles", base_path=workspace)

            assert "test-agent" in result.profiles
            profile = result.profiles["test-agent"]
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
            result = discover_profiles(".jaato/profiles", base_path=workspace)

            assert result.profiles.get("nonexistent") is None

    def test_resolve_profile_no_profiles_dir(self, monkeypatch):
        """A workspace with no profiles dir contributes no profiles.

        Asserted per tier, because ``discover_profiles`` reads three of
        them: the workspace, ``~/.jaato/profiles/``, and profiles
        registered through ``jaato.premium`` entry points.  "The
        workspace has no profiles dir" has never implied "there are no
        profiles" — it implies "none from tier 1" — and the flat
        ``== {}`` this replaces measured all three at once.  On a
        developer box with jaato-premium installed it saw 21 profiles
        and read as though ``base_path`` were being ignored (#734).

        The home tier is empty because ``jaato-server/conftest.py``
        points ``HOME`` at an empty directory; the premium tier is
        stubbed here, since an installed package is not something an
        env var can isolate.
        """
        from shared.plugins.subagent import config as config_module

        monkeypatch.setattr(
            config_module, "_discover_premium_profiles", lambda: {},
        )
        with tempfile.TemporaryDirectory() as workspace:
            result = config_module.discover_profiles(
                ".jaato/profiles", base_path=workspace,
            )
            assert result.profiles == {}

    def test_user_tier_is_scanned_when_the_workspace_has_none(
        self, tmp_path, monkeypatch,
    ):
        """``~/.jaato/profiles/`` answers when the workspace tier is empty.

        The companion to the test above: an empty result only says
        "tier 1 is empty" if the other tiers would otherwise have
        spoken.  This one proves the user tier is read at all, so the
        two together pin the precedence the docstring on
        ``discover_profiles`` claims.
        """
        from shared.plugins.subagent import config as config_module

        monkeypatch.setattr(
            config_module, "_discover_premium_profiles", lambda: {},
        )
        home = tmp_path / "home"
        user_profiles = home / ".jaato" / "profiles"
        user_profiles.mkdir(parents=True)
        (user_profiles / "user_tier.yaml").write_text(
            "name: user_tier\ndescription: from the user tier\nplugins: []\n"
        )
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("USERPROFILE", str(home))

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        result = config_module.discover_profiles(
            ".jaato/profiles", base_path=str(workspace),
        )
        assert "user_tier" in result.profiles


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
            clean_plugins, preloaded, tool_scopes = parse_plugin_list(profile.plugins)
            kwargs["tools"] = clean_plugins
            if preloaded:
                kwargs["preloaded_plugins"] = preloaded
            if tool_scopes:
                kwargs["tool_scopes"] = tool_scopes

        assert kwargs["tools"] == ["cli", "todo", "web_search"]
        assert "preloaded_plugins" not in kwargs
        assert "tool_scopes" not in kwargs

    def test_profile_with_preload_annotations(self):
        """Profile with (preload) annotations produces preloaded_plugins."""
        from shared.plugins.subagent.config import SubagentProfile

        # Simulate real flow: profile loaded from JSON has clean plugins
        # and preloaded_plugins already separated
        profile = SubagentProfile(
            name="test",
            description="Test",
            plugins=["cli", "todo", "web_search"],
            preloaded_plugins={"todo"},
        )

        kwargs = {}
        if profile.plugins:
            kwargs["tools"] = profile.plugins
            if profile.preloaded_plugins:
                kwargs["preloaded_plugins"] = profile.preloaded_plugins

        assert kwargs["tools"] == ["cli", "todo", "web_search"]
        assert kwargs["preloaded_plugins"] == {"todo"}

    def test_profile_with_space_before_preload(self):
        """Profile with space before (preload) annotation is parsed correctly."""
        from shared.plugins.subagent.config import SubagentProfile

        # Simulate real flow: profile loaded from JSON has clean plugins
        # and preloaded_plugins already separated
        profile = SubagentProfile(
            name="test",
            description="Test",
            plugins=["cli", "ast_search", "todo", "prompt_library"],
            preloaded_plugins={"ast_search", "prompt_library"},
        )

        kwargs = {}
        if profile.plugins:
            kwargs["tools"] = profile.plugins
            if profile.preloaded_plugins:
                kwargs["preloaded_plugins"] = profile.preloaded_plugins

        assert kwargs["tools"] == ["cli", "ast_search", "todo", "prompt_library"]
        assert kwargs["preloaded_plugins"] == {"ast_search", "prompt_library"}

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


class TestProfilePermissionConfig:
    """Tests for profile permission config being applied to the middleware."""

    def test_permission_init_config_merges_profile_policy(self):
        """Profile's plugin_configs.permission overrides the default policy."""
        # Simulate the logic in JaatoServer.initialize() that merges
        # the profile's permission config into the middleware init config.
        from shared.plugins.subagent.config import SubagentProfile

        profile = SubagentProfile(
            name="test-agent",
            description="Test agent with allow policy",
            plugin_configs={
                "permission": {
                    "policy": {
                        "defaultPolicy": "allow",
                        "blacklist": {
                            "tools": [],
                            "patterns": [],
                            "arguments": {
                                "cli_based_tool": {
                                    "command": ["rm -rf", "sudo"]
                                }
                            }
                        }
                    }
                }
            },
        )

        # Default middleware config (hardcoded in core.py)
        permission_init_config = {
            "channel_type": "queue",
            "channel_config": {"use_colors": False},
            "policy": {
                "defaultPolicy": "ask",
                "whitelist": {"tools": [], "patterns": []},
                "blacklist": {"tools": [], "patterns": []},
            }
        }

        # Apply profile overrides (same logic as core.py)
        if profile.plugin_configs:
            profile_perm_config = profile.plugin_configs.get("permission")
            if profile_perm_config:
                permission_init_config.update(profile_perm_config)

        # The profile's policy should override the default "ask"
        assert permission_init_config["policy"]["defaultPolicy"] == "allow"
        # Channel config should be preserved (not in profile override)
        assert permission_init_config["channel_type"] == "queue"
        # Blacklist from profile should be present
        assert "cli_based_tool" in permission_init_config["policy"]["blacklist"]["arguments"]

    def test_permission_init_config_unchanged_without_profile(self):
        """Without a profile, default 'ask' policy is used."""
        permission_init_config = {
            "channel_type": "queue",
            "channel_config": {"use_colors": False},
            "policy": {
                "defaultPolicy": "ask",
                "whitelist": {"tools": [], "patterns": []},
                "blacklist": {"tools": [], "patterns": []},
            }
        }

        profile = None
        if profile and hasattr(profile, 'plugin_configs') and profile.plugin_configs:
            profile_perm_config = profile.plugin_configs.get("permission")
            if profile_perm_config:
                permission_init_config.update(profile_perm_config)

        assert permission_init_config["policy"]["defaultPolicy"] == "ask"


class TestParsePluginEntryToolScope:
    """Tests for the ``plugin(mode:..., tools:[...])`` modifier grammar.

    Covers the implicit (positional, by token-shape) form, the explicit
    (tagged ``key:value``) form, free mixing of the two, order
    independence, legacy ``(preload)`` back-compat, and error handling.
    """

    def test_bare_name(self):
        from shared.plugins.subagent.config import parse_plugin_entry
        assert parse_plugin_entry("cli") == ("cli", False, None)

    def test_legacy_preload_flag_still_parses(self):
        from shared.plugins.subagent.config import parse_plugin_entry
        assert parse_plugin_entry("file_edit(preload)") == ("file_edit", True, None)

    def test_legacy_preload_with_space(self):
        from shared.plugins.subagent.config import parse_plugin_entry
        assert parse_plugin_entry("file_edit (preload)") == ("file_edit", True, None)

    def test_explicit_mode_preload(self):
        from shared.plugins.subagent.config import parse_plugin_entry
        assert parse_plugin_entry("file_edit(mode:preload)") == ("file_edit", True, None)

    def test_explicit_mode_discover_is_default(self):
        from shared.plugins.subagent.config import parse_plugin_entry
        assert parse_plugin_entry("file_edit(mode:discover)") == ("file_edit", False, None)

    def test_implicit_tools_allowlist(self):
        from shared.plugins.subagent.config import parse_plugin_entry
        assert parse_plugin_entry("file_edit([readFile])") == (
            "file_edit", False, ["readFile"]
        )

    def test_explicit_tools_allowlist(self):
        from shared.plugins.subagent.config import parse_plugin_entry
        assert parse_plugin_entry("file_edit(tools:[readFile,writeFile])") == (
            "file_edit", False, ["readFile", "writeFile"]
        )

    def test_explicit_tools_bare_single_value(self):
        from shared.plugins.subagent.config import parse_plugin_entry
        assert parse_plugin_entry("file_edit(tools:readFile)") == (
            "file_edit", False, ["readFile"]
        )

    def test_combined_mode_and_tools_tagged(self):
        from shared.plugins.subagent.config import parse_plugin_entry
        assert parse_plugin_entry(
            "file_edit(mode:preload, tools:[readFile,writeFile])"
        ) == ("file_edit", True, ["readFile", "writeFile"])

    def test_combined_implicit(self):
        from shared.plugins.subagent.config import parse_plugin_entry
        assert parse_plugin_entry("file_edit(preload, [readFile])") == (
            "file_edit", True, ["readFile"]
        )

    def test_order_independent(self):
        from shared.plugins.subagent.config import parse_plugin_entry
        # tools-then-mode parses identically to mode-then-tools
        assert parse_plugin_entry("file_edit([readFile], preload)") == (
            "file_edit", True, ["readFile"]
        )

    def test_whitespace_in_tool_list(self):
        from shared.plugins.subagent.config import parse_plugin_entry
        assert parse_plugin_entry("file_edit(tools:[readFile, writeFile])") == (
            "file_edit", False, ["readFile", "writeFile"]
        )

    def test_invalid_mode_raises(self):
        from shared.plugins.subagent.config import parse_plugin_entry
        with pytest.raises(ValueError):
            parse_plugin_entry("file_edit(mode:eager)")

    def test_unknown_key_raises(self):
        from shared.plugins.subagent.config import parse_plugin_entry
        with pytest.raises(ValueError):
            parse_plugin_entry("file_edit(foo:bar)")

    def test_unrecognised_bareword_raises(self):
        from shared.plugins.subagent.config import parse_plugin_entry
        with pytest.raises(ValueError):
            parse_plugin_entry("file_edit(bogus)")

    def test_parse_plugin_list_aggregates_scopes(self):
        from shared.plugins.subagent.config import parse_plugin_list
        names, preloaded, scopes = parse_plugin_list(
            ["cli", "file_edit(preload, [readFile])", "todo(mode:discover)"]
        )
        assert names == ["cli", "file_edit", "todo"]
        assert preloaded == {"file_edit"}
        assert scopes == {"file_edit": ["readFile"]}

    def test_parse_plugin_list_no_scopes(self):
        from shared.plugins.subagent.config import parse_plugin_list
        names, preloaded, scopes = parse_plugin_list(["cli", "todo"])
        assert names == ["cli", "todo"]
        assert preloaded == set()
        assert scopes == {}

    def test_from_dict_carries_tool_scopes(self):
        from shared.plugins.subagent.config import build_inline_profile
        profile = build_inline_profile(
            {"plugins": ["file_edit([readFile])", "todo"]},
            name="scoped",
            description="Scoped",
        )
        assert profile.tool_scopes == {"file_edit": ["readFile"]}
        assert profile.plugins == ["file_edit", "todo"]
