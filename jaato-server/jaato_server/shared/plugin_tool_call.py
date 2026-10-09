"""One plugin tool call with no session: the runner-side half (#1606).

A client that wants a plugin's tool -- jaato-mcp's ``kind: plugin`` tool,
kbwiki offering the ``template`` plugin to its MCP callers -- had two routes,
both wrong for it: a model turn, or importing
``jaato_server.shared.plugins.<x>`` and calling the executor in-process
(jaato-server in the client's environment, internals as API, and the call
outside the profile's confinement).

A tool needs no session.  What it needs is its plugin initialized with the
profile's configuration, the profile's permission policy, and a process
inside the workspace's boundary.  So the runner bootstraps in PLUGIN-HOST
mode (``SessionInitEnvelope.plugin_host``): every envelope step that sets up
the PROCESS runs as for a session (session env, output redaction, private
``/tmp``, the uid drop, AppArmor/SELinux confinement, the tmpdir, the user
tier, plugin discovery and the permission plugin), and the ``JaatoSession``
is never built -- no provider, no model, no history, no system prompt.  In
its place, :class:`PluginToolHost` holds a ``ToolExecutor`` wired the way
``JaatoSession.configure`` wires one, so a call takes the path a model's call
takes: the surface check, the permission gate, argument coercion (#1358),
the result transformers, secret redaction (#1215) and the ``(ok, payload)``
failure contract (#1053).

The surface is the profile's, computed here from the envelope's plugin
specs rather than asked of a session: the enabled set (#1590) is the
``plugins:`` list plus what every session has (``introspection``,
``permission``, enrichment-only plugins), and a ``plugin(tools:[...])``
scope (#1513) leaves the other tools out.  A plugin that reads a
per-session setting (``session_plugin_setting``) finds no current session
and falls back to its instance's value -- which is the profile's, because
plugin discovery initialized it with the profile's ``plugin_configs``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

#: ``reason`` values of a tool that does not exist for the profile.
REASON_UNKNOWN_TOOL = "unknown_tool"
REASON_WRONG_PLUGIN = "wrong_plugin"
REASON_NOT_IN_SURFACE = "not_in_surface"
REASON_HIDDEN = "hidden"


def schema_to_dict(schema: Any) -> Dict[str, Any]:
    """A ``ToolSchema`` as JSON: the fields a caller reads to build a call.

    ``traits`` becomes a sorted list (JSON has no set).  ``editable`` is the
    permission prompt's edit hint and is not part of what a caller passes,
    so it is left out.
    """
    category = getattr(schema, "category", None)
    return {
        "name": str(getattr(schema, "name", "")),
        "description": str(getattr(schema, "description", "") or ""),
        "parameters": dict(getattr(schema, "parameters", {}) or {}),
        "category": str(category) if category is not None else None,
        "discoverability": str(getattr(schema, "discoverability", "") or ""),
        "traits": sorted(str(t) for t in (getattr(schema, "traits", None) or ())),
    }


def _plugin_schema(plugin: Any, tool: str) -> Optional[Any]:
    """The schema ``plugin`` declares for ``tool``, or ``None``."""
    try:
        schemas = plugin.get_tool_schemas() or []
    except Exception:  # noqa: BLE001 -- a broken plugin declares nothing
        return None
    for schema in schemas:
        if getattr(schema, "name", None) == tool:
            return schema
    return None


class PluginToolHost:
    """A plugin registry and the executor that runs its tools, no session.

    Built by the runner's plugin-host bootstrap (see the module docstring).

    Args:
        registry: The runner's ``PluginRegistry``, discovered and
            initialized with the profile's ``plugin_configs``.
        plugins: The profile's ``plugins:`` names, or ``None`` when no
            profile declared a list (every exposed plugin is enabled).
        tool_scopes: ``{plugin: [tool, ...]}`` from ``plugin(tools:[...])``.
        executor: The ``ToolExecutor`` to run calls with; :meth:`wire`
            gives it the surface, the executors and the permission plugin.
    """

    def __init__(
        self,
        registry: Any,
        plugins: Optional[Iterable[str]],
        tool_scopes: Optional[Dict[str, List[str]]],
        executor: Any,
    ) -> None:
        self.registry = registry
        self._plugins: Optional[Set[str]] = set(plugins) if plugins is not None else None
        self._scopes = {k: set(v) for k, v in (tool_scopes or {}).items()}
        # ``_executor`` (with the underscore) is the attribute the runner's
        # //child install reads off a session; the host answers to it too.
        self._executor = executor

    # ------------------------------------------------------------------
    # The profile's surface (#1590 enabled set, #1513 scopes)
    # ------------------------------------------------------------------

    def plugin_enabled(self, plugin_name: str) -> bool:
        """Whether the profile enables ``plugin_name`` (JaatoSession's rule)."""
        if self._plugins is None or plugin_name in self._plugins:
            return True
        from .plugins.registry import PluginRegistry
        if plugin_name in PluginRegistry._ALWAYS_INITIALIZE_PLUGINS:
            return True
        is_enrichment_only = getattr(self.registry, "is_enrichment_only", None)
        try:
            return callable(is_enrichment_only) and is_enrichment_only(plugin_name) is True
        except Exception:  # noqa: BLE001
            return False

    def tool_scope_refusal(self, tool_name: str) -> str:
        """Why ``tool_name`` is not in the profile's surface, or ``""``."""
        try:
            plugin = self.registry.get_plugin_for_tool(tool_name)
        except Exception:  # noqa: BLE001
            plugin = None
        if plugin is None:
            return ""
        name = getattr(plugin, "name", "")
        if not self.plugin_enabled(name):
            return (f"`{tool_name}` is not available: the profile does not "
                    f"enable plugin `{name}`")
        scope = self._scopes.get(name)
        if scope is not None and tool_name not in scope:
            return (f"`{tool_name}` is not available: the profile scopes "
                    f"plugin `{name}` to {sorted(scope)}")
        return ""

    def tool_in_surface(self, tool_name: str) -> bool:
        """Whether ``tool_name`` exists for the profile."""
        return not self.tool_scope_refusal(tool_name)

    def wire(self, runtime: Any, permission_context: Dict[str, Any],
             runtime_limits: Any = None) -> "PluginToolHost":
        """Give the executor what ``JaatoSession.configure`` gives a session's.

        The surface predicate, every enabled plugin's executors, the
        registry (auto-background), the runtime limits (after the
        registry: they are forwarded by walking it, #735) and the
        permission plugin with ``permission_context``.
        """
        executor = self._executor
        executor.set_tool_surface(self.tool_in_surface, self.tool_scope_refusal)
        names = None if self._plugins is None else sorted(self._plugins)
        for name, fn in runtime.get_executors(names).items():
            executor.register(name, fn)
        executor.set_registry(self.registry)
        if runtime_limits is not None:
            executor.set_runtime_limits(None, runtime_limits)
        if getattr(runtime, "permission_plugin", None) is not None:
            executor.set_permission_plugin(runtime.permission_plugin,
                                           context=dict(permission_context))
        return self

    # ------------------------------------------------------------------
    # The two operations
    # ------------------------------------------------------------------

    def describe(self, plugin_name: str, tool: str) -> Dict[str, Any]:
        """``{exists, reason, detail, tool_schema}`` for ``plugin_name``'s ``tool``.

        ``exists`` is True only when the named plugin owns the tool, the
        profile's surface holds it and no visibility predicate hides it;
        ``tool_schema`` is then the schema after every
        ``narrow_tool_schema`` hook, as a session would put it on the wire.
        """
        try:
            owner = self.registry.get_plugin_for_tool(tool)
        except Exception:  # noqa: BLE001
            owner = None
        if owner is None:
            return _absent(REASON_UNKNOWN_TOOL,
                           f"no plugin of this profile provides a tool `{tool}`")
        owner_name = getattr(owner, "name", "")
        if owner_name != plugin_name:
            return _absent(REASON_WRONG_PLUGIN,
                           f"`{tool}` is provided by the plugin `{owner_name}`, "
                           f"not `{plugin_name}`")
        refusal = self.tool_scope_refusal(tool)
        if refusal:
            return _absent(REASON_NOT_IN_SURFACE, refusal)
        schema = _plugin_schema(owner, tool)
        if schema is None:
            return _absent(REASON_UNKNOWN_TOOL,
                           f"the plugin `{plugin_name}` declares no schema for `{tool}`")
        from .tool_visibility import filter_visible_tool_schemas

        visible = filter_visible_tool_schemas(self.registry, [schema])
        if not visible:
            return _absent(REASON_HIDDEN, f"the plugin `{plugin_name}` hides `{tool}`")
        return {"exists": True, "reason": "", "detail": "",
                "tool_schema": schema_to_dict(visible[0])}

    def invoke(self, plugin_name: str, tool: str, args: Dict[str, Any], *,
               call_id: Optional[str] = None,
               cancel_token: Any = None) -> Dict[str, Any]:
        """Run one call of ``tool`` through the executor.

        Returns:
            ``{"ran": False, "reason", "detail"}`` when the tool does not
            exist for the profile (nothing ran), else ``{"ran": True,
            "success", "result"}`` -- ``success`` is the executor's ``ok``
            flag, so a permission denial or a tool failure is
            ``success=False`` with the tool's own payload as ``result``.
        """
        described = self.describe(plugin_name, tool)
        if not described["exists"]:
            return {"ran": False, "reason": described["reason"],
                    "detail": described["detail"]}
        ok, result = _execute(self._executor, tool, dict(args or {}),
                              cancel_token, call_id)
        return {"ran": True, "success": bool(ok), "result": result}


def _absent(reason: str, detail: str) -> Dict[str, Any]:
    return {"exists": False, "reason": reason, "detail": detail,
            "tool_schema": None}


def _execute(executor: Any, tool: str, args: Dict[str, Any],
             cancel_token: Any, call_id: Optional[str]) -> Tuple[bool, Any]:
    kwargs: Dict[str, Any] = {"call_id": call_id}
    if cancel_token is not None:
        kwargs["cancel_token"] = cancel_token
    return executor.execute(tool, args, **kwargs)


def build_plugin_tool_host(
    runtime: Any,
    plugins: Optional[List[str]],
    tool_scopes: Optional[Dict[str, List[str]]],
    *,
    permission_context: Dict[str, Any],
    runtime_limits: Any = None,
    executor_factory: Optional[Callable[[], Any]] = None,
) -> PluginToolHost:
    """A wired :class:`PluginToolHost` over ``runtime``'s registry."""
    if executor_factory is None:
        from .ai_tool_runner import ToolExecutor

        def executor_factory() -> Any:
            return ToolExecutor(ledger=getattr(runtime, "ledger", None))

    host = PluginToolHost(runtime.registry, plugins, tool_scopes, executor_factory())
    return host.wire(runtime, permission_context, runtime_limits)
