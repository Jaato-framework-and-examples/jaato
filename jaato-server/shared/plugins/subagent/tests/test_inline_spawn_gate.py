"""The inline-spawn gate: ``allow_inline`` finally decides something (#944).

``spawn_subagent`` let the model omit ``profile``, and the subagent then
inherited the parent's ENTIRE plugin set with no system instructions —
returning ``success: true``.  A voice agent told to delegate document
writing to a ``documentalista`` profile spawned an inline subagent with no
``file_edit`` and no persona; nothing was written, and the tool result was
indistinguishable from a correct delegation.

The knob against that, ``allow_inline``, was declared, documented,
advertised to the model by ``list_subagent_profiles`` — and read nowhere.
These tests pin the four surfaces that now agree about it:

* the tool schema's ``required`` array (the contract the provider enforces),
* the tool + parameter descriptions (what the model reads),
* the executor's gate (what actually runs, on the local AND remote paths),
* ``inline_allowed_plugins``, which used to bind only a caller that opted
  into being restricted.

Plus the profile-side half: ``default_agent``, so a profile that knows
which persona belongs to it can say so instead of every caller repeating
the pair.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ..config import SubagentConfig, SubagentProfile
from ..plugin import SubagentPlugin


def _plugin(*, allow_inline=None, profiles=None, inline_allowed_plugins=None,
            parent_plugins=("cli", "memory")) -> SubagentPlugin:
    """A plugin initialized just far enough to reach the gate."""
    plugin = SubagentPlugin()
    plugin._initialized = True
    cfg = SubagentConfig(project="", location="")
    if allow_inline is not None:
        cfg.allow_inline = allow_inline
    if profiles:
        for prof in profiles:
            cfg.add_profile(prof)
    if inline_allowed_plugins is not None:
        cfg.inline_allowed_plugins = list(inline_allowed_plugins)
    plugin._config = cfg
    plugin._parent_plugins = list(parent_plugins)
    return plugin


def _spawn_schema(plugin: SubagentPlugin):
    for schema in plugin.get_tool_schemas():
        if schema.name == "spawn_subagent":
            return schema
    raise AssertionError("spawn_subagent schema missing")


class TestDefaults:
    """The runtime default and the advertised default agree now."""

    def test_config_default_is_false(self):
        assert SubagentConfig(project="", location="").allow_inline is False

    def test_from_dict_default_is_false(self):
        cfg = SubagentConfig.from_dict({"project": "p", "location": "l"})
        assert cfg.allow_inline is False

    def test_from_dict_honours_explicit_true(self):
        cfg = SubagentConfig.from_dict(
            {"project": "p", "location": "l", "allow_inline": True})
        assert cfg.allow_inline is True

    def test_config_schema_default_matches_the_dataclass(self):
        """Defect 2: the schema said False while the runtime said True."""
        schema = SubagentPlugin().get_config_schema()
        advertised = schema["properties"]["allow_inline"]["default"]
        assert advertised is SubagentConfig(project="", location="").allow_inline

    def test_no_config_at_all_denies_inline(self):
        """A plugin with no config is not a plugin with a permissive one."""
        plugin = SubagentPlugin()
        assert plugin._inline_allowed() is False


class TestToolSchemaFollowsTheKnob:

    def test_profile_is_required_when_inline_is_disallowed(self):
        schema = _spawn_schema(_plugin(allow_inline=False))
        assert schema.parameters["required"] == ["task", "profile"]

    def test_profile_is_optional_when_inline_is_allowed(self):
        schema = _spawn_schema(_plugin(allow_inline=True))
        assert schema.parameters["required"] == ["task"]

    def test_inline_config_absent_when_inline_is_disallowed(self):
        """Advertising a parameter the executor rejects wastes a turn."""
        schema = _spawn_schema(_plugin(allow_inline=False))
        assert "inline_config" not in schema.parameters["properties"]

    def test_inline_config_present_when_inline_is_allowed(self):
        schema = _spawn_schema(_plugin(allow_inline=True))
        assert "inline_config" in schema.parameters["properties"]

    def test_description_stops_offering_the_inline_alternative(self):
        """Models read descriptions; leaving 'EITHER ... OR ...' in place
        while ``required`` says otherwise reproduces the defect at the
        prompt layer."""
        denied = _spawn_schema(_plugin(allow_inline=False))
        allowed = _spawn_schema(_plugin(allow_inline=True))
        assert "EITHER a profile name" in allowed.description
        assert "EITHER a profile name" not in denied.description
        assert "REQUIRED" in denied.description
        assert "REQUIRED" in denied.parameters["properties"]["profile"]["description"]

    def test_schema_is_rebuilt_per_exposure_not_memoised(self):
        """Subagents share the parent's registry, so a cached schema would
        leak one agent's knob into another's tool list."""
        plugin = _plugin(allow_inline=True)
        assert _spawn_schema(plugin).parameters["required"] == ["task"]
        plugin._config.allow_inline = False
        assert _spawn_schema(plugin).parameters["required"] == ["task", "profile"]

    def test_system_instructions_follow_the_knob(self):
        denied = _plugin(allow_inline=False).get_system_instructions()
        allowed = _plugin(allow_inline=True).get_system_instructions()
        assert "Only spawn inline (no profile)" in allowed
        assert "Only spawn inline (no profile)" not in denied
        assert "'profile' is a REQUIRED argument" in denied


class TestExecutorGate:

    def test_omitted_profile_is_refused(self):
        plugin = _plugin(allow_inline=False, profiles=[
            SubagentProfile(name="documentalista", description="writes docs",
                            plugins=["file_edit"]),
        ])
        result = plugin._execute_spawn_subagent({"task": "write a doc"})
        assert result["success"] is False
        assert "requires a 'profile'" in result["error"]
        # Matches the wording a WRONG profile name has always produced.
        assert "documentalista" in result["error"]

    def test_refusal_says_so_when_no_profiles_exist_at_all(self):
        plugin = _plugin(allow_inline=False)
        result = plugin._execute_spawn_subagent({"task": "write a doc"})
        assert result["success"] is False
        assert "No profiles are configured" in result["error"]

    def test_inline_config_is_rejected(self):
        plugin = _plugin(allow_inline=False, profiles=[
            SubagentProfile(name="writer", description="", plugins=["file_edit"]),
        ])
        result = plugin._execute_spawn_subagent({
            "task": "write a doc", "profile": "writer",
            "inline_config": {"plugins": ["cli"]},
        })
        assert result["success"] is False
        assert "inline_config" in result["error"]

    def test_a_named_profile_still_passes_the_gate(self):
        """The gate refuses inline spawns, not spawns."""
        plugin = _plugin(allow_inline=False, profiles=[
            SubagentProfile(name="writer", description="", plugins=["file_edit"]),
        ])
        assert plugin._inline_spawn_denial("writer", None) is None

    def test_wrong_profile_name_still_reports_not_found(self):
        """The gate must not swallow the pre-existing not-found error."""
        plugin = _plugin(allow_inline=False, profiles=[
            SubagentProfile(name="writer", description="", plugins=["file_edit"]),
        ])
        result = plugin._execute_spawn_subagent({
            "task": "t", "profile": "typo"})
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_inline_spawn_survives_when_allowed(self):
        plugin = _plugin(allow_inline=True)
        assert plugin._inline_spawn_denial(None, None) is None
        assert plugin._inline_spawn_denial(None, {"plugins": ["cli"]}) is None

    def test_own_profile_refusal_stops_advertising_inline_config(self):
        plugin = _plugin(allow_inline=False, profiles=[
            SubagentProfile(name="writer", description="", plugins=["cli"]),
        ])
        plugin._self_profile_name = "writer"
        result = plugin._execute_spawn_subagent({"task": "t", "profile": "writer"})
        assert result["success"] is False
        assert "inline_config" not in result["error"]

    def test_own_profile_is_not_offered_as_an_alternative(self):
        plugin = _plugin(allow_inline=False, profiles=[
            SubagentProfile(name="writer", description="", plugins=["cli"]),
        ])
        plugin._self_profile_name = "writer"
        denial = plugin._inline_spawn_denial(None, None)
        assert "writer" not in denial


class TestInlineAllowedPlugins:
    """Defect 3: the allow-list bound only a caller that opted in."""

    def test_inherited_set_is_validated(self):
        plugin = _plugin(allow_inline=True,
                         inline_allowed_plugins=["cli", "todo"],
                         parent_plugins=["cli", "file_edit", "memory"])
        denial = plugin._disallowed_inline_plugins(plugin._parent_plugins)
        assert denial is not None
        assert "file_edit" in denial and "memory" in denial

    def test_explicit_plugins_are_validated(self):
        plugin = _plugin(allow_inline=True, inline_allowed_plugins=["cli"])
        assert plugin._disallowed_inline_plugins(["cli", "mcp"]) is not None

    def test_subset_passes(self):
        plugin = _plugin(allow_inline=True, inline_allowed_plugins=["cli", "todo"])
        assert plugin._disallowed_inline_plugins(["cli"]) is None

    def test_no_allow_list_means_no_restriction(self):
        plugin = _plugin(allow_inline=True, inline_allowed_plugins=[])
        assert plugin._disallowed_inline_plugins(["anything"]) is None

    def test_spawn_refuses_an_inherited_set_outside_the_allow_list(self):
        """End to end: the spawn that omitted ``inline_config`` entirely."""
        plugin = _plugin(allow_inline=True,
                         inline_allowed_plugins=["cli", "todo"],
                         parent_plugins=["cli", "file_edit"])
        result = plugin._execute_spawn_subagent({"task": "do it"})
        assert result["success"] is False
        assert "not allowed for inline creation" in result["error"]
        assert "file_edit" in result["error"]


class TestListProfilesReportsTheTruth:
    """``inline_allowed`` was the knob's ONLY reader, and it lied."""

    def test_inline_allowed_is_reported_false_by_default(self):
        plugin = _plugin(allow_inline=False, profiles=[
            SubagentProfile(name="writer", description="", plugins=["cli"]),
        ])
        result = plugin._execute_list_profiles({})
        assert result["inline_allowed"] is False
        assert result["profile_required"] is True

    def test_inline_allowed_is_reported_true_when_opted_in(self):
        plugin = _plugin(allow_inline=True, profiles=[
            SubagentProfile(name="writer", description="", plugins=["cli"]),
        ])
        result = plugin._execute_list_profiles({})
        assert result["inline_allowed"] is True
        assert "profile_required" not in result

    def test_empty_profile_list_does_not_invite_an_inline_spawn(self):
        plugin = _plugin(allow_inline=False)
        result = plugin._execute_list_profiles({})
        assert result["profiles"] == []
        assert "just call spawn_subagent with a task" not in result["message"]
        assert result["inline_allowed"] is False

    def test_default_agent_is_surfaced(self):
        plugin = _plugin(allow_inline=False, profiles=[
            SubagentProfile(name="writer", description="", plugins=["file_edit"],
                            default_agent="documentalista"),
            SubagentProfile(name="plain", description="", plugins=["cli"]),
        ])
        entries = {p["name"]: p for p in plugin._execute_list_profiles({})["profiles"]}
        assert entries["writer"]["default_agent"] == "documentalista"
        assert "default_agent" not in entries["plain"]


class TestDefaultAgent:
    """A profile supplies plugins; an agent supplies the persona.  The
    binding belongs in the profile that already knows which persona is
    its own, not in every caller."""

    def test_field_defaults_to_none(self):
        assert SubagentProfile(name="x", description="").default_agent is None

    def test_parsed_from_a_profile_dict(self):
        from ..config import build_inline_profile
        p = build_inline_profile({"plugins": [], "default_agent": "documentalista"})
        assert p.default_agent == "documentalista"

    def test_survives_a_snapshot_round_trip(self):
        from ..config import profile_from_snapshot, profile_to_snapshot
        original = SubagentProfile(name="writer", description="d",
                                   plugins=["file_edit"],
                                   default_agent="documentalista")
        revived = profile_from_snapshot(profile_to_snapshot(original))
        assert revived.default_agent == "documentalista"

    def test_spawn_resolves_the_profile_persona(self, monkeypatch):
        plugin = _plugin(allow_inline=False, profiles=[
            SubagentProfile(name="documentalista", description="",
                            plugins=["file_edit"],
                            default_agent="documentalista_persona"),
        ])
        plugin._executor = MagicMock()
        seen = {}

        def _resolve_agent(name, params, cwd, config_root=None):
            seen["name"] = name
            return {"system_instructions": "You write documents."}

        import server.session_manager as sm
        monkeypatch.setattr(sm.SessionManager, "_resolve_agent",
                            staticmethod(_resolve_agent))
        plugin._execute_spawn_subagent({"task": "write it",
                                        "profile": "documentalista"})
        assert seen["name"] == "documentalista_persona"

    def test_explicit_agent_wins_over_default_agent(self, monkeypatch):
        plugin = _plugin(allow_inline=False, profiles=[
            SubagentProfile(name="documentalista", description="",
                            plugins=["file_edit"],
                            default_agent="documentalista_persona"),
        ])
        plugin._executor = MagicMock()
        seen = {}

        def _resolve_agent(name, params, cwd, config_root=None):
            seen["name"] = name
            return {"system_instructions": "..."}

        import server.session_manager as sm
        monkeypatch.setattr(sm.SessionManager, "_resolve_agent",
                            staticmethod(_resolve_agent))
        plugin._execute_spawn_subagent({"task": "t", "profile": "documentalista",
                                        "agent": "reviewer"})
        assert seen["name"] == "reviewer"

    def test_missing_default_agent_blames_the_profile_not_the_caller(
            self, monkeypatch):
        plugin = _plugin(allow_inline=False, profiles=[
            SubagentProfile(name="documentalista", description="",
                            plugins=["file_edit"],
                            default_agent="gone"),
        ])
        plugin._executor = MagicMock()

        import server.session_manager as sm
        monkeypatch.setattr(sm.SessionManager, "_resolve_agent",
                            staticmethod(lambda *a, **k: None))
        result = plugin._execute_spawn_subagent({"task": "t",
                                                 "profile": "documentalista"})
        assert result["success"] is False
        assert "declares default_agent 'gone'" in result["error"]
        assert "configuration error" in result["error"]

    def test_inheritance_is_scalar_override(self):
        from ..config import _merge_profiles
        parent = SubagentProfile(name="base", description="",
                                 default_agent="base_persona")
        child = SubagentProfile(name="leaf", description="")
        errors = {}
        merged = _merge_profiles("leaf", [parent], child, errors)
        assert errors == {}
        assert merged.default_agent == "base_persona"

        child2 = SubagentProfile(name="leaf2", description="",
                                 default_agent="own_persona")
        merged2 = _merge_profiles("leaf2", [parent], child2, {})
        assert merged2.default_agent == "own_persona"


class TestFindAgentFile:
    """The lookup half of ``resolve_agent``, split out so ``validate`` can
    ask whether a persona EXISTS without rendering it."""

    def test_finds_a_workspace_agent(self, tmp_path):
        from ..config import find_agent_file
        agents = tmp_path / ".jaato" / "agents"
        agents.mkdir(parents=True)
        (agents / "writer.md").write_text("hi", encoding="utf-8")
        found = find_agent_file("writer", None, str(tmp_path / ".jaato"))
        assert found is not None and found.name == "writer.md"

    def test_finds_a_directory_persona(self, tmp_path):
        from ..config import find_agent_file
        d = tmp_path / ".jaato" / "agents" / "writer"
        d.mkdir(parents=True)
        (d / "PROMPT.md").write_text("hi", encoding="utf-8")
        assert find_agent_file("writer", None, str(tmp_path / ".jaato")) is not None

    def test_returns_none_for_an_unknown_name(self, tmp_path):
        from ..config import find_agent_file
        (tmp_path / ".jaato" / "agents").mkdir(parents=True)
        assert find_agent_file("gone", None, str(tmp_path / ".jaato")) is None

    def test_resolve_agent_still_resolves_through_it(self, tmp_path):
        """The extraction must not change what ``resolve_agent`` finds."""
        from ..config import resolve_agent
        agents = tmp_path / ".jaato" / "agents"
        agents.mkdir(parents=True)
        (agents / "writer.md").write_text("You write.", encoding="utf-8")
        result = resolve_agent("writer", {}, None, str(tmp_path / ".jaato"))
        assert result is not None
        assert "You write." in result["system_instructions"]
        assert resolve_agent("gone", {}, None, str(tmp_path / ".jaato")) is None
