"""A tool a profile scopes out does not exist for that session (#1513, #1491).

``references(preload, tools:[proposeReference])`` is the profile's statement
of which of the plugin's tools a session has.  Before #1513 only the initial
wire schema honoured it: ``list_tools`` listed the other references tools,
``get_tool_schemas`` returned their schemas, and the executor ran them —
``listReferences`` is housekeeping, so ``auto_allow_housekeeping`` approved
it, and a 228k-token result overflowed a session whose author had removed
the tool precisely to stop that.  And the references plugin's own hints
(#1491) told such a session to call ``selectReferences``, checking only the
undeclared, instance-wide ``exclude_tools``.

One per-session predicate, ``JaatoSession.tool_in_surface``, now answers at
all four points: the wire, discovery (``filter_visible_tool_schemas``), the
executor (before the permission gate) and the hints.

These tests drive a real ``PluginRegistry`` holding the real ``references``
and ``introspection`` plugins, shared by two real ``JaatoSession`` objects —
one scoped, one not — because a shared registry is where the per-session
property can fail.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Set

import pytest

from jaato_server.shared.jaato_runtime import JaatoRuntime
from jaato_server.shared.jaato_session import JaatoSession
from jaato_server.shared.plugins.introspection import plugin as _introspection_module
from jaato_server.shared.plugins.introspection.plugin import IntrospectionPlugin
from jaato_server.shared.plugins.permission.plugin import PermissionPlugin
from jaato_server.shared.plugins.references.plugin import ReferencesPlugin
from jaato_server.shared.plugins.registry import PluginRegistry
from jaato_server.shared.session_context import (
    isolated_current_session, set_current_session,
)
from jaato_server.shared.tests.reversion import Reversion
from jaato_server.shared.tool_id_map import name_to_id

_SESSION = "jaato-server/jaato_server/shared/jaato_session.py"
_RUNNER = "jaato-server/jaato_server/shared/ai_tool_runner.py"
_VIS = "jaato-server/jaato_server/shared/tool_visibility.py"
_REFS = "jaato-server/jaato_server/shared/plugins/references/plugin.py"

REVERSIONS = [
    Reversion(
        target=_VIS,
        find="    schemas = _scoped_to_session(schemas, _session_or_current(session))\n",
        replace="",
        test="test_list_tools_does_not_name_a_scoped_out_tool",
        because="discovery walks a registry-wide view, so the catalog lists "
                "tools the profile took away",
    ),
    Reversion(
        target=_VIS,
        find="    schemas = _scoped_to_session(schemas, _session_or_current(session))\n",
        replace="",
        test="test_get_tool_schemas_returns_no_schema_for_a_scoped_out_tool",
        because="get_tool_schemas hands back the schema of a scoped-out tool",
    ),
    Reversion(
        target=_RUNNER,
        find=("        scope_refusal = self._scope_refusal(name, args, call_id)\n"
              "        if scope_refusal is not None:\n"
              "            return scope_refusal\n"),
        replace="",
        test="test_calling_a_scoped_out_tool_is_refused_before_the_permission_gate",
        because="the executor runs a scoped-out tool a housekeeping rule approves",
    ),
    Reversion(
        target=_SESSION,
        find=("        scope = self._tool_scope_of(tool_name)\n"
              "        return scope is None or tool_name in scope[1]\n"),
        replace="        return True\n",
        test="test_calling_a_scoped_out_tool_is_refused_before_the_permission_gate",
        because="the one predicate admits everything, so no point holds the scope",
    ),
    Reversion(
        target=_SESSION,
        find="        self._tool_scopes = dict(tool_scopes) if tool_scopes else {}\n",
        replace=("        _shared = JaatoSession.__dict__.get('_shared_scopes')\n"
                 "        if _shared is None:\n"
                 "            _shared = {}\n"
                 "            JaatoSession._shared_scopes = _shared\n"
                 "        _shared.update(tool_scopes or {})\n"
                 "        self._tool_scopes = _shared\n"),
        test="test_a_sibling_session_without_the_scope_still_has_the_tools",
        because="a scope kept on state the sessions share (the #944 shape) "
                "takes the tools away from every session on the registry",
    ),
    Reversion(
        target=_REFS,
        find=("        if tool_name in self._exclude_tools:\n"
              "            return False\n"
              "        return tool_in_session_surface(tool_name)\n"),
        replace="        return tool_name not in self._exclude_tools\n",
        test="test_no_reference_hint_when_select_references_is_scoped_out",
        because="the hints consult only exclude_tools and point the session "
                "at a tool it cannot call (#1491)",
    ),
    Reversion(
        target=_REFS,
        find=("        if mentioned_ids and not self._tool_available(\"selectReferences\"):\n"),
        replace="        if False:\n",
        test="test_no_mention_expansion_when_select_references_is_scoped_out",
        because="pass 1 expands an @mention into instructions for a session "
                "that cannot select the reference",
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

SCOPE = {"references": ["proposeReference"]}
OUT_OF_SCOPE = ("listReferences", "selectReferences")


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """No session leaks through the ContextVar or introspection's
    thread-local, and no developer HOME config leaks in."""
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
    """One runtime, one registry, ONE references and ONE introspection
    instance that every session shares."""
    ws = tmp_path / "ws"
    ws.mkdir()
    runtime = JaatoRuntime(provider_name="echo", workspace_path=ws)
    reg = PluginRegistry()
    reg.set_workspace_path(str(ws))
    refs = ReferencesPlugin()
    reg.register_plugin(refs, expose=True, config={
        "sources": SOURCES,
        "lookup_strategy": "tags_only",
        "refresh_catalog": False,
        "workspace_path": str(ws),
    })
    intro = IntrospectionPlugin()
    reg.register_plugin(intro, expose=True)
    intro.set_plugin_registry(reg)
    runtime.configure_plugins(reg)
    return runtime


def _session(runtime, *, scoped: bool, agent_id: str) -> JaatoSession:
    session = JaatoSession(runtime, "test-model", agent_id=agent_id)
    session.configure(
        skip_provider=True,
        plugins=["references", "introspection"],
        tool_scopes=SCOPE if scoped else None,
        preloaded_plugins={"references"},  # references(preload, ...)
    )
    session._provider = SimpleNamespace(uses_external_tools=lambda: True)
    return session


def _intro(runtime) -> IntrospectionPlugin:
    return runtime.registry.get_plugin("introspection")


def _as(session: JaatoSession) -> None:
    """Serve ``session`` on this thread, as ``_execute_single_tool`` does."""
    set_current_session(session)
    _intro(session._runtime).set_session(session)


def _listed(session: JaatoSession) -> Set[str]:
    """Every tool name ``list_tools`` names for ``session``."""
    _as(session)
    intro = _intro(session._runtime)
    names: Set[str] = set()
    for cat in intro._execute_list_tools({})["categories"]:
        detail = intro._execute_list_tools({"category_id": cat["id"]})
        names.update(t["name"] for t in detail.get("tools", []))
    return names


def _permission(spy: list) -> PermissionPlugin:
    """A real permission plugin that WOULD approve listReferences, twice
    over: whitelisted, and housekeeping under auto_allow_housekeeping."""
    perm = PermissionPlugin()
    perm.initialize({
        "auto_allow_housekeeping": True,
        "policy": {
            "defaultPolicy": "deny",
            "whitelist": {"tools": list(OUT_OF_SCOPE)},
        },
    })
    original = perm.check_permission

    def _spy(name, *a, **kw):
        spy.append(name)
        return original(name, *a, **kw)

    perm.check_permission = _spy
    return perm


# ------------------------------------------------------------- the wire


def test_the_wire_omits_the_scoped_out_tools(rt):
    s = _session(rt, scoped=True, agent_id="writer")
    wire = {t.name for t in s._get_tools_for_provider()}
    assert "proposeReference" in wire
    assert not (wire & set(OUT_OF_SCOPE))


# ------------------------------------------------------------ discovery


def test_list_tools_does_not_name_a_scoped_out_tool(rt):
    s = _session(rt, scoped=True, agent_id="writer")
    listed = _listed(s)
    assert "list_tools" in listed  # core tools are not subject to the scope
    assert not (listed & set(OUT_OF_SCOPE)), sorted(listed & set(OUT_OF_SCOPE))


def test_get_tool_schemas_returns_no_schema_for_a_scoped_out_tool(rt):
    s = _session(rt, scoped=True, agent_id="writer")
    _as(s)
    result = _intro(rt)._execute_get_tool_schemas(
        {"tool_ids": [name_to_id("listReferences")]}
    )
    assert not result.get("schemas"), result
    [note] = result["not_available"]
    assert "references" in note["reason"] and "proposeReference" in note["reason"]
    assert "listReferences" not in {t.name for t in s._tools}


# ------------------------------------------------------------ execution


def test_calling_a_scoped_out_tool_is_refused_before_the_permission_gate(rt):
    s = _session(rt, scoped=True, agent_id="writer")
    asked: list = []
    s._executor.set_permission_plugin(_permission(asked))
    _as(s)
    ok, result = s._executor.execute("listReferences", {}, call_id="c1")
    assert ok is False
    assert result["_permission"]["method"] == "tool_scope"
    assert ("`listReferences` is not available in this session: the profile "
            "scopes plugin `references` to [proposeReference]") in result["error"]
    assert "listReferences" not in asked, "permission gate consulted"


def test_a_sibling_session_without_the_scope_still_has_the_tools(rt):
    """Per session: the registry and plugin instances are shared."""
    _session(rt, scoped=True, agent_id="writer")
    sibling = _session(rt, scoped=False, agent_id="reader")

    assert set(OUT_OF_SCOPE) <= _listed(sibling)

    asked: list = []
    sibling._executor.set_permission_plugin(_permission(asked))
    _as(sibling)
    ok, result = sibling._executor.execute("listReferences", {}, call_id="c2")
    assert ok is True, result
    assert asked == ["listReferences"]


# ---------------------------------------------------------------- hints


def test_no_reference_hint_when_select_references_is_scoped_out(rt):
    s = _session(rt, scoped=True, agent_id="writer")
    enriched = s._enrich_and_clean_prompt("we write this service in java")
    assert "selectReferences" not in enriched
    assert "java-guide" not in enriched


def test_the_hint_is_injected_when_select_references_is_in_scope(rt):
    _session(rt, scoped=True, agent_id="writer")
    s = _session(rt, scoped=False, agent_id="reader")
    enriched = s._enrich_and_clean_prompt("we write this service in java")
    assert "selectReferences" in enriched
    assert "java-guide" in enriched


def test_no_mention_expansion_when_select_references_is_scoped_out(rt):
    s = _session(rt, scoped=True, agent_id="writer")
    enriched = s._enrich_and_clean_prompt("follow @java-guide please")
    assert "Referenced Sources" not in enriched
    # The mention was not resolved, so its @ is kept (#1429).
    assert "@java-guide" in enriched


def test_the_system_instructions_do_not_point_at_select_references(rt):
    s = _session(rt, scoped=True, agent_id="writer")
    set_current_session(s)
    text = rt.registry.get_plugin("references").get_system_instructions() or ""
    assert "selectReferences" not in text
    sibling = _session(rt, scoped=False, agent_id="reader")
    set_current_session(sibling)
    text = rt.registry.get_plugin("references").get_system_instructions() or ""
    assert "selectReferences" in text
