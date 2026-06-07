"""Session-side integration tests for the per-turn visibility filter
added to ``JaatoSession._get_tools_for_provider`` 2026-06-07.

Pins:
    1. Plugins WITHOUT ``is_tool_visible`` are unaffected — every tool
       flows through.
    2. Plugins WITH ``is_tool_visible`` returning ``True`` for every
       name are unaffected.
    3. Plugins WITH ``is_tool_visible`` returning ``False`` for a
       specific name → that tool is stripped from the array.
    4. Two plugins with predicates: hidden if EITHER hides it
       (intersection of visible sets).
    5. A predicate that raises is fail-open (treats as visible,
       logs the error).
    6. Empty ``_tools`` short-circuits without touching the registry.
    7. ``uses_external_tools() is False`` returns ``[]`` regardless.

Companion to
``shared/plugins/todo/tests/test_plan_required_visibility.py``,
which pins the plugin-side contract for ``is_tool_visible`` itself.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from shared.jaato_session import JaatoSession
from jaato_sdk.plugins.model_provider.types import ToolSchema


def _make_session_stub(
    tools,
    registry=None,
    uses_external_tools=True,
):
    """Build the minimal SimpleNamespace `self` shape that
    ``_get_tools_for_provider`` reads from.  The method is invoked
    as a bound function — no full session construction needed.
    """
    provider = SimpleNamespace(uses_external_tools=lambda: uses_external_tools)
    runtime = SimpleNamespace(registry=registry)
    return SimpleNamespace(
        _tools=tools,
        _provider=provider,
        _runtime=runtime,
        _trace=lambda msg: None,
    )


def _make_tool(name: str) -> ToolSchema:
    return ToolSchema(name=name, description=f"desc of {name}", parameters={})


class _FakePlugin:
    """Plugin stub with configurable ``is_tool_visible``."""

    def __init__(self, name: str, predicate=None):
        self._name = name
        self._predicate = predicate

    @property
    def name(self) -> str:
        return self._name

    def is_tool_visible(self, tool_name: str) -> bool:
        return self._predicate(tool_name) if self._predicate else True


class _PluginNoFilter:
    """Plugin stub WITHOUT ``is_tool_visible``."""

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name


def _make_registry(plugins_by_name):
    """Build a registry stub exposing the given plugins."""
    return SimpleNamespace(
        list_exposed=lambda: list(plugins_by_name.keys()),
        get_plugin=lambda name: plugins_by_name.get(name),
    )


# ============================================================
# Filter behavior
# ============================================================


class TestVisibilityFilter:

    def test_no_plugins_with_filter_passes_through(self):
        """Plugins without ``is_tool_visible`` don't affect the array."""
        tools = [_make_tool("a"), _make_tool("b")]
        registry = _make_registry({"x": _PluginNoFilter("x")})
        session = _make_session_stub(tools, registry=registry)

        result = JaatoSession._get_tools_for_provider(session)

        assert [t.name for t in result] == ["a", "b"]

    def test_filter_returning_true_for_everything_passes_through(self):
        tools = [_make_tool("a"), _make_tool("b")]
        plugin = _FakePlugin("p", predicate=lambda name: True)
        registry = _make_registry({"p": plugin})
        session = _make_session_stub(tools, registry=registry)

        result = JaatoSession._get_tools_for_provider(session)

        assert [t.name for t in result] == ["a", "b"]

    def test_filter_hides_specific_tool(self):
        tools = [_make_tool("a"), _make_tool("b"), _make_tool("c")]
        plugin = _FakePlugin("p", predicate=lambda name: name != "b")
        registry = _make_registry({"p": plugin})
        session = _make_session_stub(tools, registry=registry)

        result = JaatoSession._get_tools_for_provider(session)

        assert [t.name for t in result] == ["a", "c"]

    def test_two_filters_union_their_hides(self):
        """Tool is hidden if EITHER predicate hides it
        (intersection of visible sets)."""
        tools = [_make_tool("a"), _make_tool("b"), _make_tool("c")]
        p1 = _FakePlugin("p1", predicate=lambda name: name != "a")
        p2 = _FakePlugin("p2", predicate=lambda name: name != "b")
        registry = _make_registry({"p1": p1, "p2": p2})
        session = _make_session_stub(tools, registry=registry)

        result = JaatoSession._get_tools_for_provider(session)

        assert [t.name for t in result] == ["c"]

    def test_predicate_raising_fails_open(self):
        """A buggy ``is_tool_visible`` must not break the turn.
        Fail-open (treat tool as visible)."""
        def boom(tool_name):
            raise RuntimeError("predicate exploded")

        tools = [_make_tool("a"), _make_tool("b")]
        plugin = _FakePlugin("p", predicate=boom)
        registry = _make_registry({"p": plugin})
        session = _make_session_stub(tools, registry=registry)

        # No exception should propagate.
        result = JaatoSession._get_tools_for_provider(session)

        assert [t.name for t in result] == ["a", "b"]

    def test_one_good_one_bad_predicate_good_still_applies(self):
        """The good predicate still hides its target even when a
        sibling predicate raises."""
        def boom(tool_name):
            raise RuntimeError("predicate exploded")

        tools = [_make_tool("a"), _make_tool("b"), _make_tool("c")]
        bad = _FakePlugin("bad", predicate=boom)
        good = _FakePlugin("good", predicate=lambda name: name != "b")
        registry = _make_registry({"bad": bad, "good": good})
        session = _make_session_stub(tools, registry=registry)

        result = JaatoSession._get_tools_for_provider(session)

        assert [t.name for t in result] == ["a", "c"]


# ============================================================
# Fast-path short-circuits
# ============================================================


class TestFastPaths:

    def test_empty_tools_returns_empty(self):
        session = _make_session_stub(tools=[], registry=_make_registry({}))
        assert JaatoSession._get_tools_for_provider(session) == []

    def test_none_tools_returns_none(self):
        session = _make_session_stub(tools=None, registry=_make_registry({}))
        assert JaatoSession._get_tools_for_provider(session) is None

    def test_no_registry_passes_through(self):
        tools = [_make_tool("a")]
        # _runtime.registry is None
        session = _make_session_stub(tools, registry=None)
        result = JaatoSession._get_tools_for_provider(session)
        assert [t.name for t in result] == ["a"]

    def test_registry_list_exposed_raises_passes_through(self):
        """If the registry itself blows up enumerating plugins, we
        return the unfiltered tools rather than fail the turn."""
        tools = [_make_tool("a")]
        registry = SimpleNamespace(
            list_exposed=MagicMock(side_effect=RuntimeError("registry broke")),
            get_plugin=lambda name: None,
        )
        session = _make_session_stub(tools, registry=registry)
        result = JaatoSession._get_tools_for_provider(session)
        assert [t.name for t in result] == ["a"]

    def test_uses_external_tools_false_returns_empty(self):
        """Claude CLI delegated mode short-circuits with []
        BEFORE any filtering happens."""
        tools = [_make_tool("a")]
        plugin = _FakePlugin("p", predicate=lambda name: False)
        registry = _make_registry({"p": plugin})
        session = _make_session_stub(
            tools, registry=registry, uses_external_tools=False,
        )
        assert JaatoSession._get_tools_for_provider(session) == []
