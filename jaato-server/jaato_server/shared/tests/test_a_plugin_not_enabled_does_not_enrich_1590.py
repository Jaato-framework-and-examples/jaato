"""A plugin a profile does not enable does not enrich that session (#1590).

A stage whose profile listed only ``permission`` and
``file_edit(mode:preload, tools:[readFile])`` still received the
``references`` plugin's prompt enrichment: a 16k-character
"**Reference sources available** — use `selectReferences`" block naming
139 ids and a tool the session could not call.  The model read the ids as
pages, guessed paths, failed, and ended the stage with an error.

Two facts made it possible.  The registry is shared and initializes every
discovered plugin whether or not the profile lists it (#950, #1563), and
its enrichment subscriber lists asked every exposed plugin.  And #1491's
guard (``references`` asks ``tool_in_session_surface("selectReferences")``)
read ``JaatoSession.tool_in_surface``, which answered ``True`` for any
tool whose plugin the profile did not SCOPE, so a plugin the profile did
not enable at all counted as in the surface.

Now ``JaatoSession.plugin_enabled`` is the session's enabled set (its
``plugins:`` list, plus ``_ALWAYS_INITIALIZE_PLUGINS`` and enrichment-only
plugins; everything when there is no list), ``tool_in_surface`` is False
for a tool of a plugin outside it, and the registry's three enrichment
subscriber lists skip such a plugin for the calling session.

The tests drive a real ``PluginRegistry`` holding the real ``references``
plugin, shared by two real ``JaatoSession`` objects, one enabling
``references`` and one not.  A synthetic enricher that checks nothing
proves the registry gate on its own (``references`` would also be stopped
by its own #1491 guard, which would make a registry-only reversion
decorative), and calling ``references`` directly proves the guard on its
own.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from jaato_sdk.plugins.base import (
    PromptEnrichmentResult,
    SystemInstructionEnrichmentResult,
    ToolResultEnrichmentResult,
)
from jaato_sdk.plugins.model_provider.types import ToolSchema
from jaato_server.shared.jaato_runtime import JaatoRuntime
from jaato_server.shared.jaato_session import JaatoSession
from jaato_server.shared.plugins.introspection import plugin as _introspection_module
from jaato_server.shared.plugins.introspection.plugin import IntrospectionPlugin
from jaato_server.shared.plugins.references.plugin import ReferencesPlugin
from jaato_server.shared.plugins.registry import PluginRegistry
from jaato_server.shared.session_context import (
    isolated_current_session, set_current_session,
)
from jaato_server.shared.tests.reversion import Reversion

_SESSION = "jaato-server/jaato_server/shared/jaato_session.py"
_REGISTRY = "jaato-server/jaato_server/shared/plugins/registry.py"


def _gate(kind: str) -> str:
    return (
        "                if not plugin_enabled_for_session(name):\n"
        "                    continue\n"
        f"                if (hasattr(plugin, 'subscribes_to_{kind}') and\n"
    )


REVERSIONS = [
    Reversion(
        target=_REGISTRY,
        find=_gate("prompt_enrichment"),
        replace="                if (hasattr(plugin, 'subscribes_to_prompt_enrichment') and\n",
        test="test_a_plugin_not_enabled_adds_nothing_to_the_prompt",
        because="the prompt-enrichment loop asks every exposed plugin, so a "
                "plugin the profile does not enable still enriches the session",
    ),
    Reversion(
        target=_REGISTRY,
        find=_gate("tool_result_enrichment"),
        replace="                if (hasattr(plugin, 'subscribes_to_tool_result_enrichment') and\n",
        test="test_a_plugin_not_enabled_adds_nothing_to_a_tool_result",
        because="tool results are enriched by plugins the session cannot reach",
    ),
    Reversion(
        target=_REGISTRY,
        find=_gate("system_instruction_enrichment"),
        replace="                if (hasattr(plugin, 'subscribes_to_system_instruction_enrichment') and\n",
        test="test_a_plugin_not_enabled_adds_nothing_to_the_system_instructions",
        because="system instructions are enriched by plugins the session "
                "cannot reach",
    ),
    Reversion(
        target=_SESSION,
        find=("        if not self.plugin_enabled(plugin.name):\n"
              "            return (\n"),
        replace=("        if False:\n"
                 "            return (\n"),
        test="test_tool_in_surface_is_false_for_a_plugin_not_enabled",
        because="a plugin the profile does not enable has no scope, so its "
                "tools count as in the surface and #1491's guard admits them",
    ),
    Reversion(
        target=_SESSION,
        find=("        if plugin_name in PluginRegistry._ALWAYS_INITIALIZE_PLUGINS:\n"
              "            return True\n"),
        replace="",
        test="test_core_tools_stay_in_the_surface",
        because="list_tools / get_tool_schemas leave the surface of a session "
                "whose plugins: list does not name introspection",
    ),
    Reversion(
        target=_SESSION,
        find="                return is_enrichment_only(plugin_name) is True\n",
        replace="                return False\n",
        test="test_an_enrichment_only_plugin_still_enriches_every_session",
        because="an enrichment-only plugin is never named in plugins:, so it "
                "would stop enriching every profiled session",
    ),
    Reversion(
        target=_SESSION,
        find="        if self._tool_plugins is None or plugin_name in self._tool_plugins:\n",
        replace="        if self._tool_plugins is None:\n",
        test="test_the_session_that_enables_references_gets_the_hint",
        because="a plugin the profile DOES enable is treated as disabled",
    ),
]


SOURCES = [{
    "id": "java-guide",
    "name": "Java Guide",
    "description": "How we write Java services",
    "type": "inline",
    "mode": "selectable",
    "content": "Use records.",
    "tags": ["java"],
}]

PROMPT = "we write this service in java"
MARK = "<<probe-enricher>>"


class _ProbeEnricher:
    """A tool plugin whose enrichment checks nothing: whether it runs for a
    session is decided by the registry alone."""

    def __init__(self, name: str = "probe") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def get_tool_schemas(self) -> List[ToolSchema]:
        return [ToolSchema(
            name=f"{self._name}_tool", description="probe",
            parameters={"type": "object", "properties": {}},
        )]

    def get_executors(self) -> Dict[str, Any]:
        return {f"{self._name}_tool": lambda args: {"ok": True}}

    def get_system_instructions(self) -> Optional[str]:
        return None

    def get_auto_approved_tools(self) -> List[str]:
        return []

    def get_user_commands(self) -> list:
        return []

    def subscribes_to_prompt_enrichment(self) -> bool:
        return True

    def enrich_prompt(self, prompt: str) -> PromptEnrichmentResult:
        return PromptEnrichmentResult(prompt=prompt + MARK)

    def subscribes_to_tool_result_enrichment(self) -> bool:
        return True

    def enrich_tool_result(self, tool_name, result, tool_args=None):
        return ToolResultEnrichmentResult(result=result + MARK)

    def subscribes_to_system_instruction_enrichment(self) -> bool:
        return True

    def enrich_system_instructions(self, instructions: str):
        return SystemInstructionEnrichmentResult(instructions=instructions + MARK)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(
        _introspection_module._thread_local, "session", None, raising=False,
    )
    with isolated_current_session():
        yield


@pytest.fixture
def rt(tmp_path: Path):
    """One runtime, one registry, ONE references instance and ONE probe
    that every session shares."""
    ws = tmp_path / "ws"
    ws.mkdir()
    runtime = JaatoRuntime(provider_name="echo", workspace_path=ws)
    reg = PluginRegistry()
    reg.set_workspace_path(str(ws))
    reg.register_plugin(ReferencesPlugin(), expose=True, config={
        "sources": SOURCES,
        "lookup_strategy": "tags_only",
        "refresh_catalog": False,
        "workspace_path": str(ws),
    })
    intro = IntrospectionPlugin()
    reg.register_plugin(intro, expose=True)
    intro.set_plugin_registry(reg)
    reg.register_plugin(_ProbeEnricher(), expose=True)
    runtime.configure_plugins(reg)
    return runtime


def _session(runtime, plugins: Optional[List[str]], agent_id: str) -> JaatoSession:
    session = JaatoSession(runtime, "test-model", agent_id=agent_id)
    session.configure(skip_provider=True, plugins=plugins)
    session._provider = SimpleNamespace(uses_external_tools=lambda: True)
    return session


def _without(runtime) -> JaatoSession:
    """The incident's shape: neither references nor the probe enabled."""
    return _session(runtime, ["permission", "file_edit"], "stage")


def _with(runtime) -> JaatoSession:
    return _session(runtime, ["references", "probe", "introspection"], "writer")


# ------------------------------------------------------------- the surface


def test_tool_in_surface_is_false_for_a_plugin_not_enabled(rt):
    s = _without(rt)
    assert not s.tool_in_surface("selectReferences")
    assert not s.tool_in_surface("probe_tool")
    assert ("the profile does not enable plugin `references`"
            in s.tool_scope_refusal("selectReferences"))


def test_core_tools_stay_in_the_surface(rt):
    s = _without(rt)
    for name in ("list_tools", "get_tool_schemas", "signal_completion",
                 "askPermission"):
        assert s.tool_in_surface(name), name


def test_a_session_with_no_plugin_list_has_every_tool(rt):
    s = _session(rt, None, "embedded")
    assert s.tool_in_surface("selectReferences")
    assert s.plugin_enabled("references")


def test_the_references_guard_holds_on_its_own(rt):
    """#1491's guard, asked directly: no hint for the disabled session."""
    s = _without(rt)
    set_current_session(s)
    refs = rt.registry.get_plugin("references")
    enriched = refs.enrich_prompt(PROMPT).prompt
    assert "selectReferences" not in enriched
    assert "java-guide" not in enriched


# ------------------------------------------------------------ enrichment


def test_a_plugin_not_enabled_adds_nothing_to_the_prompt(rt):
    s = _without(rt)
    enriched = s._enrich_and_clean_prompt(PROMPT)
    assert MARK not in enriched
    assert "Reference sources available" not in enriched
    assert "selectReferences" not in enriched


def test_a_plugin_not_enabled_adds_nothing_to_a_tool_result(rt):
    s = _without(rt)
    set_current_session(s)
    out = rt.registry.enrich_tool_result("readFile", "we use java here").result
    assert MARK not in out
    assert "selectReferences" not in out


def test_a_plugin_not_enabled_adds_nothing_to_the_system_instructions(rt):
    s = _without(rt)
    assert "selectReferences" not in (s._system_instruction or "")
    set_current_session(s)
    out = rt.registry.enrich_system_instructions("base").instructions
    assert MARK not in out


def test_the_session_that_enables_references_gets_the_hint(rt):
    """Per session: the sibling sharing the registry still gets both.

    The disabled session goes first, because references dedups a hint it
    has surfaced on the shared instance."""
    disabled = _without(rt)
    assert MARK not in disabled._enrich_and_clean_prompt(PROMPT)
    enabled = _with(rt)
    assert enabled.tool_in_surface("selectReferences")
    enriched = enabled._enrich_and_clean_prompt(PROMPT)
    assert MARK in enriched
    assert "selectReferences" in enriched and "java-guide" in enriched
    set_current_session(enabled)
    assert MARK in rt.registry.enrich_tool_result("readFile", "x").result


def test_an_enrichment_only_plugin_still_enriches_every_session(rt):
    rt.registry.register_plugin(
        _ProbeEnricher("marker_only"), enrichment_only=True)
    s = _without(rt)
    assert s.plugin_enabled("marker_only")
    assert MARK in s._enrich_and_clean_prompt(PROMPT)
