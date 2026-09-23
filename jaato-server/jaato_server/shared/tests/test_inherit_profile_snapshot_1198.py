"""``spawn_subagent(profile="inherit")`` — a frozen parent snapshot (#1198).

``inherit`` is one reserved value added to the ``profile`` enum.  It does
NOT re-open inline profiles (``allow_inline`` stays ``False``); it spawns a
subagent from a frozen snapshot of THIS session's profile at the moment of
the call — the parent's plugin set and the parent's system instructions —
so the common "spawn a child like me" pattern is expressible without the
opt-in that #944 deliberately switched off.

The one design constraint the triage settled, implemented here as the
smaller, local mitigation rather than a general spawn-depth bound (#680):
an ``inherit`` snapshot includes the persona that makes the parent
delegate, so an ``inherit`` child is primed to spawn ``inherit`` itself.
The child therefore receives the parent's plugins **minus the subagent
plugin**, so it can neither spawn nor message siblings.  A caller that
needs a spawning child names a real profile.

These tests pin the acceptance criteria that can regress silently.  The
guard lives here (``shared/tests``) rather than beside the plugin because
the reversion meta-suite walks only ``shared/tests`` and ``server/tests``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

from jaato_server.shared.plugins.subagent.config import (
    INHERIT_PROFILE_NAME,
    SubagentConfig,
    SubagentProfile,
    _scan_profiles_dir,
)
from jaato_server.shared.plugins.subagent.plugin import SubagentPlugin
from jaato_server.shared.tests.reversion import Reversion
from jaato_server.shared.tool_result_builder import split_executor_result as _split

_PLUGIN = "jaato-server/jaato_server/shared/plugins/subagent/plugin.py"
_CONFIG = "jaato-server/jaato_server/shared/plugins/subagent/config.py"

_PARENT_INSTRUCTION = "PARENT FRAMING: you are the coordinator.\n\nDo the thing."


def _plugin(
    *,
    allow_inline: bool = False,
    parent_plugins=("cli", "subagent", "todo"),
    parent_instruction: str = _PARENT_INSTRUCTION,
    profiles=None,
) -> SubagentPlugin:
    """A plugin wired just far enough to resolve an ``inherit`` spawn.

    ``parent_plugins`` deliberately INCLUDES ``subagent`` so the strip is a
    load-bearing step the guard can observe, even though the runtime that
    populates ``_parent_plugins`` already excludes it upstream.
    """
    plugin = SubagentPlugin()
    plugin._initialized = True
    cfg = SubagentConfig(project="", location="")
    cfg.allow_inline = allow_inline
    for prof in (profiles or []):
        cfg.add_profile(prof)
    plugin._config = cfg
    plugin._parent_plugins = list(parent_plugins)
    # A stand-in parent session: _build_inherit_profile only reads
    # get_system_instruction, and the executor only needs id()/truthiness.
    plugin._parent_session = SimpleNamespace(
        get_system_instruction=lambda: parent_instruction,
    )
    return plugin


class _RecordingExecutor:
    """Captures submit() args instead of running the subagent thread."""

    def __init__(self) -> None:
        self.calls = []

    def submit(self, fn, *args, **kwargs):
        self.calls.append((fn, args, kwargs))
        return None


def _spawn_schema(plugin: SubagentPlugin):
    for schema in plugin.get_tool_schemas():
        if schema.name == "spawn_subagent":
            return schema
    raise AssertionError("spawn_subagent schema missing")


def _plugin_names(profile: SubagentProfile):
    from jaato_server.shared.plugins.subagent.config import parse_plugin_entry
    return [parse_plugin_entry(entry)[0] for entry in (profile.plugins or [])]


# ── (a) inherit succeeds under allow_inline: false ─────────────────────────

def test_inherit_passes_the_inline_gate_under_allow_inline_false():
    """The whole point: no ``allow_inline`` needed.  ``inherit`` names a
    profile, so the #944 gate that refuses a profile-less spawn lets it
    through even with inline spawning off."""
    plugin = _plugin(allow_inline=False)
    assert plugin._inline_spawn_denial(INHERIT_PROFILE_NAME, None) is None


def test_inherit_resolves_under_allow_inline_false():
    """Driven end to end, ``profile="inherit"`` reaches session creation
    (not the not-found refusal a discovered-profile miss produces)."""
    plugin = _plugin(allow_inline=False)
    rec = _RecordingExecutor()
    plugin._executor = rec
    result = plugin._execute_spawn_subagent({"profile": "inherit", "task": "go"})
    ok, _payload = _split(result)
    assert ok is True, result
    assert rec.calls, "spawn was not submitted"
    # submit(fn, agent_id, profile, prompt, cwd, owner, display, params, override)
    _fn, args, _kw = rec.calls[0]
    submitted_profile = args[1]
    assert isinstance(submitted_profile, SubagentProfile)
    assert "subagent" not in _plugin_names(submitted_profile)
    assert args[7] == _PARENT_INSTRUCTION  # override threaded to the session


# ── (b) child plugins == parent minus the subagent plugin ──────────────────

def test_inherit_child_cannot_spawn():
    """The self-replication guard: the inherit child does not receive the
    subagent plugin, so it can neither spawn nor message siblings."""
    plugin = _plugin(parent_plugins=["cli", "subagent", "todo"])
    profile, _override, err = plugin._build_inherit_profile("", None)
    assert err is None, err
    names = _plugin_names(profile)
    assert "subagent" not in names
    assert names == ["cli", "todo"]


def test_inherit_strips_subagent_even_with_a_tools_modifier():
    """A ``subagent(tools:[...])`` entry is stripped by plugin name, not by
    exact string, so the guard holds however the entry is spelled."""
    plugin = _plugin(parent_plugins=["cli", "subagent(tools:[list_siblings])"])
    profile, _override, err = plugin._build_inherit_profile("", None)
    assert err is None, err
    assert "subagent" not in _plugin_names(profile)


# ── (c) child instructions match the parent at spawn time ──────────────────

def test_inherit_captures_parent_instructions():
    plugin = _plugin(parent_instruction="EXACT PARENT PROMPT")
    _profile, override, err = plugin._build_inherit_profile("", None)
    assert err is None, err
    assert override == "EXACT PARENT PROMPT"


# ── (d) post-spawn parent mutation does not affect the child ───────────────

def test_inherit_is_frozen_against_later_parent_mutation():
    """Frozen = deep copy.  Mutating the parent's plugin list after the
    snapshot must not reach the already-built child."""
    parent_plugins = ["cli", "subagent", "todo"]
    plugin = _plugin(parent_plugins=parent_plugins)
    profile, _override, err = plugin._build_inherit_profile("", None)
    assert err is None, err
    before = list(profile.plugins)
    # Mutate the SAME list object the plugin holds, and its own field.
    plugin._parent_plugins.append("web_search")
    plugin._parent_plugins.clear()
    assert list(profile.plugins) == before


# ── (e) inherit.yaml is refused at discovery ───────────────────────────────

def test_inherit_yaml_is_refused_at_discovery():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "inherit.yaml").write_text(
            "description: shadow\nplugins: [cli]\n", encoding="utf-8")
        (Path(d) / "helper.yaml").write_text(
            "description: fine\nplugins: [cli]\n", encoding="utf-8")
        profiles: dict = {}
        errors: dict = {}
        _scan_profiles_dir(Path(d), profiles, errors)
    assert "inherit" not in profiles, "reserved name was loaded from disk"
    assert "inherit" in errors
    assert "reserved" in errors["inherit"].lower()
    # Control: a normally-named sibling still loads.
    assert "helper" in profiles


def test_inherit_name_via_the_name_key_is_also_refused():
    """The reservation is on the resolved name, so ``name: inherit`` in an
    ordinarily-named file is refused too — not only the filename."""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "sneaky.yaml").write_text(
            "name: inherit\nplugins: [cli]\n", encoding="utf-8")
        profiles: dict = {}
        errors: dict = {}
        _scan_profiles_dir(Path(d), profiles, errors)
    assert "inherit" not in profiles


# ── (f) the profile enum stays sorted and offers inherit ───────────────────

def test_inherit_is_in_the_sorted_profile_enum():
    plugin = _plugin(
        allow_inline=False,
        profiles=[
            SubagentProfile(name="zebra", description="z", plugins=["cli"]),
            SubagentProfile(name="alpha", description="a", plugins=["cli"]),
        ],
    )
    schema = _spawn_schema(plugin)
    enum = schema.parameters["properties"]["profile"]["enum"]
    assert "inherit" in enum
    assert enum == sorted(enum)
    assert enum == ["alpha", "inherit", "zebra"]


# ── remote/isolated refusals (each a distinct wrong outcome) ───────────────

def test_inherit_is_refused_on_the_remote_path():
    plugin = _plugin()
    ok, payload = _split(plugin._execute_spawn_subagent(
        {"profile": "inherit", "task": "go", "server": "peer-1"}))
    assert ok is False
    assert "inherit" in str(payload).lower()


def test_inherit_needs_a_parent_session():
    plugin = _plugin()
    plugin._parent_session = None
    _profile, _override, err = plugin._build_inherit_profile("", None)
    assert err is not None
    assert "parent session" in err.lower()


REVERSIONS = [
    Reversion(
        target=_PLUGIN,
        find=(
            "        stripped = [\n"
            "            entry for entry in (self._parent_plugins or [])\n"
            "            if parse_plugin_entry(entry)[0] != 'subagent'\n"
            "        ]"
        ),
        replace="        stripped = list(self._parent_plugins or [])",
        because=(
            "the subagent plugin is no longer stripped, so an inherit child "
            "keeps spawn_subagent and can self-replicate unbounded"
        ),
        test="test_inherit_child_cannot_spawn",
    ),
    Reversion(
        target=_PLUGIN,
        find="        override = self._parent_session.get_system_instruction()",
        replace="        override = None",
        because="the parent's system instructions are no longer captured",
        test="test_inherit_captures_parent_instructions",
    ),
    Reversion(
        target=_PLUGIN,
        find="        return {\"enum\": sorted([*names, INHERIT_PROFILE_NAME])}",
        replace="        return {\"enum\": sorted(names)}",
        because="inherit is no longer offered in the profile enum",
        test="test_inherit_is_in_the_sorted_profile_enum",
    ),
    Reversion(
        target=_PLUGIN,
        find=(
            "        if profile_name == INHERIT_PROFILE_NAME:\n"
            "            profile, inherited_override, inherit_error = (\n"
            "                self._build_inherit_profile(custom_name, agent_params_arg)\n"
            "            )"
        ),
        replace=(
            "        if False and profile_name == INHERIT_PROFILE_NAME:\n"
            "            profile, inherited_override, inherit_error = (\n"
            "                self._build_inherit_profile(custom_name, agent_params_arg)\n"
            "            )"
        ),
        because=(
            "inherit is no longer resolved to the parent snapshot, so it "
            "falls through to the discovered-profile lookup and is refused "
            "as 'not found'"
        ),
        test="test_inherit_resolves_under_allow_inline_false",
    ),
    Reversion(
        target=_CONFIG,
        find="        if name in RESERVED_PROFILE_NAMES:",
        replace="        if name in RESERVED_PROFILE_NAMES and False:",
        because="a file named inherit.yaml would be loaded as a real profile",
        test="test_inherit_yaml_is_refused_at_discovery",
    ),
]
