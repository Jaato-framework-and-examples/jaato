"""One plugin tool call with no model turn: the runner-side half (#1606).

A client that wants a plugin's tool -- jaato-mcp's ``kind: plugin`` tool,
kbwiki offering the ``template`` plugin to its MCP callers -- used to have
two routes, both wrong for it: a model turn (a model in between), or
importing ``jaato_server.shared.plugins.<x>`` and calling the executor
in-process (jaato-server in the client's environment, internals as API, and
the call outside the profile's confinement).

The daemon now runs the call in a short-lived session of the caller's
workspace (``server/plugin_tool_calls.py``), and the RUNNER answers it here,
against that session.  Two operations, each one function:

:func:`describe_plugin_tool`
    The tool's schema as THIS session exposes it: the owning plugin must be
    the one named, ``tool_in_surface`` must hold (#1513 scopes, #1590
    enabled set), and the schema goes through
    ``filter_visible_tool_schemas`` -- the visibility predicates and every
    ``narrow_tool_schema`` hook (``allow_inline_template: false`` removes
    ``template`` from ``renderTemplateToFile``).  Nothing here re-derives
    any of that; it asks the session the questions the wire asks it.

:func:`invoke_plugin_tool`
    Refuses what :func:`describe_plugin_tool` says does not exist (the model
    of such a session could not call it either), then hands the call to the
    session's own ``ToolExecutor.execute`` -- the path a model's call takes:
    the scope check, the permission gate (an ASK goes to the session's
    channel), argument coercion (#1358), the result transformers, secret
    redaction (#1215) and the ``(ok, payload)`` failure contract (#1053).

Both run with the session set as the current-session ContextVar, as a turn
does, so a plugin that reads a per-session setting
(``session_plugin_setting``) answers for this session.  The ContextVar is
set inside a copied context, so the pool thread that runs this does not keep
the session afterwards.
"""

from __future__ import annotations

import contextvars
from typing import Any, Dict, Optional, Tuple

#: ``reason`` values of a tool that does not exist for the session.
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


def _registry_of(session: Any) -> Any:
    runtime = getattr(session, "_runtime", None)
    return getattr(runtime, "registry", None) if runtime is not None else None


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


def _in_session_context(session: Any, fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Run ``fn`` with ``session`` as the current session, in a copied context."""
    from .session_context import set_current_session

    def _run() -> Any:
        if session is not None:
            set_current_session(session)
        return fn(*args, **kwargs)

    return contextvars.copy_context().run(_run)


def _describe(session: Any, plugin_name: str, tool: str) -> Dict[str, Any]:
    registry = _registry_of(session)
    if registry is None:
        return _absent(REASON_UNKNOWN_TOOL, "the session has no plugin registry")
    try:
        owner = registry.get_plugin_for_tool(tool)
    except Exception:  # noqa: BLE001
        owner = None
    if owner is None:
        named = registry.get_plugin(plugin_name)
        if named is not None and _plugin_schema(named, tool) is not None:
            # The plugin declares it but the registry does not serve it:
            # the plugin is loaded and not exposed for this session.
            return _absent(REASON_NOT_IN_SURFACE,
                           f"`{tool}` is not available in this session: the "
                           f"plugin `{plugin_name}` is not exposed")
        return _absent(REASON_UNKNOWN_TOOL,
                       f"no plugin in this session provides a tool `{tool}`")
    owner_name = getattr(owner, "name", "")
    if owner_name != plugin_name:
        return _absent(REASON_WRONG_PLUGIN,
                       f"`{tool}` is provided by the plugin `{owner_name}`, "
                       f"not `{plugin_name}`")
    if hasattr(session, "tool_in_surface") and not session.tool_in_surface(tool):
        refusal = (session.tool_scope_refusal(tool)
                   if hasattr(session, "tool_scope_refusal") else "")
        return _absent(REASON_NOT_IN_SURFACE,
                       refusal or f"`{tool}` is not in this session's surface")
    schema = _plugin_schema(owner, tool)
    if schema is None:
        return _absent(REASON_UNKNOWN_TOOL,
                       f"the plugin `{plugin_name}` declares no schema for `{tool}`")
    from .tool_visibility import filter_visible_tool_schemas

    visible = filter_visible_tool_schemas(registry, [schema], session=session)
    if not visible:
        return _absent(REASON_HIDDEN,
                       f"the plugin `{plugin_name}` hides `{tool}` in this session")
    return {"exists": True, "reason": "", "detail": "",
            "tool_schema": schema_to_dict(visible[0])}


def _absent(reason: str, detail: str) -> Dict[str, Any]:
    return {"exists": False, "reason": reason, "detail": detail,
            "tool_schema": None}


def describe_plugin_tool(session: Any, plugin_name: str, tool: str) -> Dict[str, Any]:
    """``{exists, reason, detail, tool_schema}`` for ``plugin_name``'s ``tool``.

    ``exists`` is True only when the named plugin owns the tool, the session's
    surface holds it and no visibility predicate hides it; ``tool_schema`` is
    then the schema after every ``narrow_tool_schema`` hook, as the session
    would put it on the wire.  See the module docstring for the reasons.
    """
    return _in_session_context(session, _describe, session, plugin_name, tool)


def invoke_plugin_tool(
    session: Any,
    plugin_name: str,
    tool: str,
    args: Dict[str, Any],
    *,
    cancel_token: Any = None,
    call_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one call of ``tool`` through the session's executor.

    Returns:
        ``{"ran": False, "reason", "detail"}`` when the tool does not exist
        for the session (nothing ran), else ``{"ran": True, "success",
        "result"}`` -- ``success`` is the executor's ``ok`` flag, so a
        permission denial or a tool failure is ``success=False`` with the
        tool's own payload as ``result``.
    """
    described = describe_plugin_tool(session, plugin_name, tool)
    if not described["exists"]:
        return {"ran": False, "reason": described["reason"],
                "detail": described["detail"]}
    executor = getattr(session, "_executor", None)
    if executor is None:
        return {"ran": False, "reason": REASON_UNKNOWN_TOOL,
                "detail": "the session has no tool executor"}
    ok, result = _in_session_context(
        session, _execute, executor, tool, dict(args or {}), cancel_token, call_id)
    return {"ran": True, "success": bool(ok), "result": result}


def _execute(executor: Any, tool: str, args: Dict[str, Any],
             cancel_token: Any, call_id: Optional[str]) -> Tuple[bool, Any]:
    kwargs: Dict[str, Any] = {"call_id": call_id}
    if cancel_token is not None:
        kwargs["cancel_token"] = cancel_token
    return executor.execute(tool, args, **kwargs)
