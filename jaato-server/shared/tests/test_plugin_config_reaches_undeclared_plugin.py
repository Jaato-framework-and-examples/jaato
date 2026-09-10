"""A profile's ``plugin_configs`` reach the plugin it names (#950).

``JaatoSession.configure`` used to apply a config block only when the plugin
was ALSO named in ``plugins:``::

    if plugins is None or plugin_name in plugins:

which quietly discarded the rest.  For a top-level session it never showed:
its configs reach the plugins through ``expose_all(plugin_configs)`` at
bootstrap, and ``permission`` gets an explicit merge on both the daemon
(``server/core.py``) and runner (``server/runner/session.py`` Step 8) paths.
A **subagent** reuses the parent's already-bootstrapped registry, so this loop
was the only place its own profile's configs could land — and a
``documentalista`` profile whose ``plugin_configs.permission`` whitelisted
``writeNewFile`` produced 55 permission ASKs on that pre-approved tool, with
no diagnostic anywhere.  Adding ``permission`` to ``plugins:`` — a plugin that
exposes no tools at all — was the whole difference.

Three properties are asserted here, each attached to a way the fix could go
wrong:

1. an undeclared plugin's config IS applied (the defect);
2. a **provider** section (``openrouter``, ``anthropic``, …) is skipped rather
   than fed to ``expose_tool``, which knows no such plugin — the old gate
   filtered those out only as a side effect of them never being in
   ``plugins``, so removing it without this would log a warning per provider
   per session;
3. the model's tool surface is untouched — ``configure`` records ``plugins``
   as ``_tool_plugins`` and that is what gates the wire, so configuring a
   plugin does not expose it.
"""

from unittest.mock import MagicMock

from ..jaato_session import JaatoSession


def _Registry(names, expose=None):
    """A ``PluginRegistry`` double that records ``expose_tool`` calls.

    A ``MagicMock`` rather than a hand-written class: ``configure`` calls a
    good deal more of the registry than this test cares about
    (``register_core_tool``, ``get_exposed_tool_schemas``, …), and stubbing
    each one would tie the test to code it is not about.  Only the two methods
    under test carry real behaviour.
    """
    reg = MagicMock()
    reg.exposed = []
    reg.list_available.return_value = list(names)

    def _expose(name, config=None):
        if name not in names:
            raise ValueError(f"Plugin '{name}' not found")
        reg.exposed.append((name, config))
        return True

    reg.expose_tool.side_effect = expose or _expose
    return reg


def _session(registry):
    runtime = MagicMock()
    runtime.registry = registry
    return JaatoSession(runtime, "test-model")


def _configure(session, **kwargs):
    session.configure(skip_provider=True, **kwargs)


def _configured(registry):
    return {name for name, _ in registry.exposed}


def test_config_for_a_plugin_absent_from_plugins_is_applied():
    """The #950 defect: the block was dropped, so the policy never applied."""
    reg = _Registry(["cli", "permission", "file_edit"])
    _configure(_session(reg), plugins=["cli", "file_edit"], plugin_configs={
        "permission": {"policy": {"whitelist": {"tools": ["writeNewFile"]}}},
    })

    assert "permission" in _configured(reg)
    (_, cfg), = [e for e in reg.exposed if e[0] == "permission"]
    assert cfg["policy"]["whitelist"]["tools"] == ["writeNewFile"]


def test_declared_plugins_are_still_configured():
    reg = _Registry(["cli", "todo"])
    _configure(_session(reg), plugins=["cli", "todo"],
               plugin_configs={"todo": {"storage_type": "file"}})

    assert _configured(reg) == {"todo"}


def test_a_provider_section_is_skipped_not_attempted():
    """``plugin_configs`` also carries provider knobs; they are not plugins."""
    reg = _Registry(["cli", "permission"])
    _configure(_session(reg), plugins=["cli"], plugin_configs={
        "openrouter": {"api_key": "sk-or-x"},
        "permission": {"policy": {"defaultPolicy": "allow"}},
    })

    assert _configured(reg) == {"permission"}


def test_configuring_a_plugin_does_not_expose_its_tools():
    """Config and exposure are separate decisions — ``plugins`` owns the wire."""
    reg = _Registry(["cli", "memory"])
    session = _session(reg)
    _configure(session, plugins=["cli"], plugin_configs={"memory": {"x": 1}})

    assert session._tool_plugins == ["cli"]


def test_plugins_none_still_configures_everything_known():
    """``plugins=None`` (no profile) kept applying every config; it still does."""
    reg = _Registry(["cli", "permission"])
    _configure(_session(reg), plugins=None,
               plugin_configs={"cli": {}, "permission": {}})

    assert _configured(reg) == {"cli", "permission"}


def test_agent_name_is_injected_into_an_undeclared_plugins_config():
    """The trace-logging injection applies on this path too, not just the old one."""
    reg = _Registry(["permission"])
    session = _session(reg)
    session.set_agent_context(agent_type="subagent", agent_name="documentalista")
    _configure(session, plugins=[], plugin_configs={"permission": {"policy": {}}})

    (_, cfg), = reg.exposed
    assert cfg["agent_name"] == "documentalista"


def test_a_failing_plugin_does_not_abort_the_rest():
    holder = {}

    def boom(name, config=None):
        if name == "cli":
            raise RuntimeError("bad config")
        holder["reg"].exposed.append((name, config))
        return True

    reg = holder["reg"] = _Registry(["cli", "permission"], expose=boom)
    _configure(_session(reg), plugins=["cli"],
               plugin_configs={"cli": {}, "permission": {}})

    assert _configured(reg) == {"permission"}
