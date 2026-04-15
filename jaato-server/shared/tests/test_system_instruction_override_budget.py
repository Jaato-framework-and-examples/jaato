"""Tests for override-aware budget + lazy base instructions.

Two related concerns converge here:

1. ``system_instruction_override`` replaces the wire-level system message
   entirely (empty-string for "send nothing", non-empty for "send exactly
   this").  The instruction budget must reflect that — not the assembly
   pipeline's would-be output — or the user sees a budget display that
   lies about what's on the wire.  Regression for the 2026-04-15 trace
   where ``--no-instructions`` sent empty to the model while the budget
   still showed thousands of tokens of premium base instructions.

2. Base system instructions (``.jaato/instructions/*.md`` plus any
   premium-provided layer) are loaded **lazily** on first request —
   sessions that use an override never pay the disk-I/O cost.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from shared.instruction_budget import (
    InstructionSource,
    SystemChildType,
)
from shared.instruction_token_cache import InstructionTokenCache
from shared.jaato_runtime import JaatoRuntime
from shared.jaato_session import JaatoSession


# ──────────────────────────────────────────────────────────────────
# Lazy base-instruction loading on the runtime
# ──────────────────────────────────────────────────────────────────


class TestLazyBaseInstructions:
    """Loading must not happen during ``JaatoRuntime.__init__`` and must
    be cached after the first request."""

    def test_not_loaded_at_construction(self):
        runtime = JaatoRuntime(provider_name="lmstudio")
        assert runtime._base_loaded is False
        # Internal field stays None until first request — even though a
        # real premium install would make the content readable, we did
        # not read it yet.
        assert runtime._base_system_instructions is None

    def test_first_call_loads_and_caches(self):
        runtime = JaatoRuntime(provider_name="lmstudio")
        first = runtime.get_base_system_instructions()
        assert runtime._base_loaded is True
        # Second call returns the cached value without re-reading.
        with patch.object(runtime, "_load_base_system_instructions") as reloader:
            second = runtime.get_base_system_instructions()
            reloader.assert_not_called()
        assert first == second

    def test_caches_across_repeated_lookups(self):
        """The cache is the core guarantee — once loaded, subsequent
        lookups must never re-read from disk, regardless of whether the
        loader found files or returned None."""
        runtime = JaatoRuntime(provider_name="lmstudio")
        runtime.get_base_system_instructions()
        assert runtime._base_loaded is True
        with patch.object(runtime, "_load_base_system_instructions") as reloader:
            for _ in range(3):
                runtime.get_base_system_instructions()
            reloader.assert_not_called()


# ──────────────────────────────────────────────────────────────────
# Override-aware budget via _collect_instruction_texts
# ──────────────────────────────────────────────────────────────────


def _session_for_collect(base_instructions: str = "BASE CONTENT") -> JaatoSession:
    """Build a minimal session suitable for exercising _collect_instruction_texts."""
    runtime = MagicMock()
    runtime.provider_name = "lmstudio"
    runtime.instruction_token_cache = InstructionTokenCache()
    runtime.get_base_system_instructions.return_value = base_instructions
    runtime._formatter_pipeline = None
    runtime.telemetry = MagicMock()
    runtime.telemetry.enabled = False
    runtime.registry = None

    session = JaatoSession.__new__(JaatoSession)
    session._runtime = runtime
    session._agent_id = "t"
    session._agent_type = "main"
    session._agent_name = None
    session._deferred_plugin_instructions = set()
    session._preloaded_plugins = set()
    session._system_instruction_override = None
    session._provider_name_override = None
    session._tool_plugins = None
    return session


class TestOverrideAwareCollect:

    def test_no_override_walks_assembly(self):
        """Baseline: no override → SYSTEM children (BASE, FRAMEWORK) appear."""
        session = _session_for_collect(base_instructions="BASE CONTENT")
        reqs = session._collect_instruction_texts(
            session_instructions=None,
            system_instruction_override=None,
        )
        child_keys = {r.child_key for r in reqs}
        assert SystemChildType.BASE.value in child_keys
        assert SystemChildType.FRAMEWORK.value in child_keys

    def test_empty_override_yields_no_entries(self):
        """Wire message is empty → budget shows nothing.  The whole
        point of --no-instructions is to free tokens in small-context
        sessions; the budget display must confirm that freedom."""
        session = _session_for_collect()
        reqs = session._collect_instruction_texts(
            session_instructions=None,
            system_instruction_override="",
        )
        assert reqs == []

    def test_non_empty_override_yields_single_override_entry(self):
        session = _session_for_collect()
        override_text = "You are terse.  Use tools."
        reqs = session._collect_instruction_texts(
            session_instructions=None,
            system_instruction_override=override_text,
        )
        assert len(reqs) == 1
        req = reqs[0]
        assert req.text == override_text
        assert req.source == InstructionSource.SYSTEM
        assert req.child_key == SystemChildType.OVERRIDE.value
        assert "Override" in req.label

    def test_override_suppresses_client_instructions(self):
        """When override is set, session_instructions (agent .md etc.)
        is also discarded — it would have been concatenated into the
        assembly the override now replaces."""
        session = _session_for_collect()
        reqs = session._collect_instruction_texts(
            session_instructions="19 KB of agent markdown",
            system_instruction_override="",
        )
        assert reqs == []

    def test_override_does_not_trigger_base_load(self):
        """Key lazy-loading claim: a session with an override never
        calls get_base_system_instructions, so no disk I/O fires."""
        session = _session_for_collect()
        session._collect_instruction_texts(
            session_instructions=None,
            system_instruction_override="",
        )
        session._runtime.get_base_system_instructions.assert_not_called()

    def test_no_override_does_call_base_loader(self):
        """Symmetric check: without an override, the base loader IS called."""
        session = _session_for_collect()
        session._collect_instruction_texts(
            session_instructions=None,
            system_instruction_override=None,
        )
        session._runtime.get_base_system_instructions.assert_called_once()


# ──────────────────────────────────────────────────────────────────
# Override + plugin-level entries
# ──────────────────────────────────────────────────────────────────


class TestSuppressBase:
    """``suppress_base_instructions`` is the partial-suppression knob:
    drops the framework baseline layer (``.jaato/instructions/*.md`` +
    premium) while keeping the agent prompt, plugin instructions, and
    framework constants that the session actually needs.
    """

    def test_suppress_skips_base_entry_only(self):
        """BASE entry drops out of the budget; CLIENT and FRAMEWORK stay."""
        session = _session_for_collect(base_instructions="FRAMEWORK BASELINE")
        reqs = session._collect_instruction_texts(
            session_instructions="agent prompt",
            suppress_base=True,
        )
        child_keys = {r.child_key for r in reqs}
        assert SystemChildType.BASE.value not in child_keys
        # CLIENT (agent prompt) and FRAMEWORK (task completion, parallel
        # tool guidance, turn summary) must still reach the model.
        assert SystemChildType.CLIENT.value in child_keys
        assert SystemChildType.FRAMEWORK.value in child_keys

    def test_suppress_does_not_trigger_base_load(self):
        """No disk I/O when the base layer is suppressed — the whole
        point of the lazy getter + suppression combo is that sessions
        which drop base never pay for it."""
        session = _session_for_collect(base_instructions="FRAMEWORK BASELINE")
        session._collect_instruction_texts(
            session_instructions="agent prompt",
            suppress_base=True,
        )
        session._runtime.get_base_system_instructions.assert_not_called()

    def test_default_includes_base(self):
        """Without the flag, BASE is fetched + included — default
        behaviour unchanged so sessions not using the flag see the full
        assembled prompt."""
        session = _session_for_collect(base_instructions="FRAMEWORK BASELINE")
        reqs = session._collect_instruction_texts(
            session_instructions="agent prompt",
        )
        child_keys = {r.child_key for r in reqs}
        assert SystemChildType.BASE.value in child_keys
        session._runtime.get_base_system_instructions.assert_called_once()

    def test_override_wins_over_suppress(self):
        """If both knobs are set, override takes precedence — the wire
        message is just the override string (or empty), ignoring the
        partial-suppression request entirely."""
        session = _session_for_collect()
        reqs = session._collect_instruction_texts(
            session_instructions="agent prompt",
            system_instruction_override="custom",
            suppress_base=True,
        )
        assert len(reqs) == 1
        assert reqs[0].child_key == SystemChildType.OVERRIDE.value
        assert reqs[0].text == "custom"

    def test_suppress_preserves_plugin_instructions(self):
        """The headline claim: plugin tool instructions survive the
        partial suppression.  The agent still knows how to use cli,
        file_edit, etc. — only the framework baseline is dropped."""
        session = _session_for_collect()
        registry = MagicMock()
        plugin = MagicMock()
        plugin.get_system_instructions.return_value = "PLUGIN TOOL HINTS"
        registry._exposed = ["cli"]
        registry.get_plugin = lambda name: plugin if name == "cli" else None
        registry.plugin_has_core_tools = lambda name: True
        session._runtime.registry = registry

        reqs = session._collect_instruction_texts(
            session_instructions="agent prompt",
            suppress_base=True,
        )
        plugin_entries = [r for r in reqs if r.source == InstructionSource.PLUGIN]
        assert len(plugin_entries) >= 1
        assert any(r.text == "PLUGIN TOOL HINTS" for r in plugin_entries)


class TestOverrideSuppressesPluginEntries:
    """Plugins contribute text that gets concatenated into the assembled
    system message — so if that whole assembly is replaced by an
    override, plugin entries must not appear in the budget either."""

    def _session_with_plugin(self) -> JaatoSession:
        session = _session_for_collect()

        registry = MagicMock()
        plugin = MagicMock()
        plugin.get_system_instructions.return_value = "PLUGIN INSTRUCTIONS"
        registry._exposed = ["cli"]
        registry.get_plugin = lambda name: plugin if name == "cli" else None
        registry.plugin_has_core_tools = lambda name: True
        session._runtime.registry = registry
        return session

    def test_no_override_includes_plugin_entries(self):
        session = self._session_with_plugin()
        reqs = session._collect_instruction_texts(
            session_instructions=None,
            system_instruction_override=None,
        )
        plugin_entries = [r for r in reqs if r.source == InstructionSource.PLUGIN]
        assert len(plugin_entries) >= 1

    def test_override_excludes_plugin_entries(self):
        session = self._session_with_plugin()
        reqs = session._collect_instruction_texts(
            session_instructions=None,
            system_instruction_override="just this",
        )
        plugin_entries = [r for r in reqs if r.source == InstructionSource.PLUGIN]
        assert plugin_entries == []
        # The single OVERRIDE entry is the only thing that reaches the wire.
        assert len(reqs) == 1
        assert reqs[0].child_key == SystemChildType.OVERRIDE.value
