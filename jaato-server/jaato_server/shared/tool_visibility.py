"""Per-turn tool visibility: the one filter behind ``is_tool_visible``.

A plugin MAY implement ``is_tool_visible(tool_name) -> bool`` to hide its
own tools while they cannot work (``todo``'s plan-required tools with no
active plan, ``telepathy``'s ``share_context`` with no parent, ``lsp``'s
tools with no language server configured).  Two surfaces must honour it,
and they must agree:

- the tool array sent to the provider
  (``JaatoSession._get_tools_for_provider``);
- the deferred-tool catalog the model browses
  (``introspection``'s ``list_tools`` and ``get_tool_schemas``).

Before #1345 only the first asked, so a hidden tool was still listed by
``list_tools`` and activatable by ``get_tool_schemas`` — the model spent
calls discovering tools the wire then withheld.  Both surfaces now call
:func:`filter_visible_tool_schemas`, so they cannot disagree.

A plugin MAY also implement ``narrow_tool_schema(schema) -> schema`` to
return a narrower copy of one of its own schemas for the current session:
a parameter the session's settings refuse is removed rather than offered
and then refused (``renderTemplateToFile``'s inline ``template`` under
``allow_inline_template: false``).  It is asked about every visible
schema and returns a foreign one unchanged, like ``is_tool_visible``.  It
must only narrow: the filter never adds a tool, and a hook that raises
leaves the schema as it was.

The filter also applies the calling session's profile scopes first
(``plugin(tools:[...])``, #1513), through ``JaatoSession.tool_in_surface``.
Before that only the initial wire schema honoured a scope: ``list_tools``
listed the scoped-out tools, ``get_tool_schemas`` returned them and the
executor ran them.  :func:`tool_in_session_surface` is the same question
for a plugin deciding whether to mention one of its tools in a hint.

The predicates answer for "the current session" (#1195), so the caller
must have set the session ContextVar for the session it is serving before
calling this.  ``_get_tools_for_provider`` sets it explicitly; a tool
executor (``list_tools``) runs after ``_execute_single_tool`` has set it.
"""

from typing import Any, Callable, List, Optional


def _session_or_current(session: Any) -> Any:
    """``session`` itself, else the session ContextVar's, else ``None``."""
    if session is not None:
        return session
    try:
        from .session_context import get_current_session
        return get_current_session()
    except LookupError:
        return None
    except Exception:
        return None


def tool_in_session_surface(tool_name: str, session: Any = None) -> bool:
    """Whether ``tool_name`` exists for the calling session under its
    profile's ``plugin(tools:[...])`` scopes (#1513, #1491).

    Delegates to ``JaatoSession.tool_in_surface`` — the one per-session
    predicate — on ``session`` or, when none is passed, the session the
    ContextVar names.  With no session (a plugin built in isolation,
    catalog discovery before ``configure()``) or a session without the
    method, every tool is in the surface: there is no scope to apply.
    A predicate that raises answers ``True`` for the same reason.

    Plugins call this to decide whether to mention one of their tools in
    an instruction or a hint (``references`` pass 1 / 2 / 2b), so they
    never point a session at a tool it cannot call.
    """
    session = _session_or_current(session)
    predicate = getattr(session, 'tool_in_surface', None)
    if predicate is None:
        return True
    try:
        return bool(predicate(tool_name))
    except Exception:
        return True


def _scoped_to_session(schemas: List[Any], session: Any) -> List[Any]:
    """``schemas`` minus the tools ``session``'s scopes leave out."""
    predicate = getattr(session, 'tool_in_surface', None)
    if predicate is None:
        return schemas
    kept = []
    for schema in schemas:
        try:
            if not predicate(schema.name):
                continue
        except Exception:
            pass
        kept.append(schema)
    return kept if len(kept) != len(schemas) else schemas


def filter_visible_tool_schemas(
    registry: Any,
    schemas: List[Any],
    on_error: Optional[Callable[[str, str, Exception], None]] = None,
    session: Any = None,
) -> List[Any]:
    """Drop the schemas the calling session's tool scopes leave out
    (#1513), then those some exposed plugin's ``is_tool_visible`` hides,
    then apply every ``narrow_tool_schema`` hook to what is left.

    The scope step asks ``session`` (default: the session ContextVar)
    through ``JaatoSession.tool_in_surface``, so ``list_tools`` and
    ``get_tool_schemas`` never list or return a tool the profile's
    ``plugin(tools:[...])`` modifier left out — per session, because the
    registry is shared with sibling subagents and the scopes are not.

    Every exposed plugin that implements the predicate is asked about
    every tool name (a predicate returns ``True`` for names it does not
    own).  A predicate that raises is treated as "visible" — a buggy
    predicate must not break the turn — and reported to ``on_error`` as
    ``(tool_name, plugin_name, exc)``.

    Uses only ``registry.list_exposed()`` and ``registry.get_plugin()``.

    Args:
        registry: The session's ``PluginRegistry``.
        schemas: Candidate tool schemas (anything with a ``.name``).
        on_error: Optional callback for a predicate that raised.
        session: The session to scope for; ``None`` reads the ContextVar.

    Returns:
        ``schemas`` itself when nothing is dropped and no plugin implements
        a hook, else a new list holding the visible schemas in their
        original order.
    """
    schemas = _scoped_to_session(schemas, _session_or_current(session))
    try:
        exposed_names = registry.list_exposed()
    except Exception:
        return schemas
    filters, narrowers = [], []
    for name in exposed_names:
        plugin = registry.get_plugin(name)
        if plugin is not None and hasattr(plugin, 'is_tool_visible'):
            filters.append(plugin)
        if plugin is not None and hasattr(plugin, 'narrow_tool_schema'):
            narrowers.append(plugin)
    if not filters and not narrowers:
        return schemas
    visible = [
        schema for schema in schemas
        if _is_visible(schema.name, filters, on_error)
    ]
    if not narrowers:
        return visible
    return [_narrowed(schema, narrowers, on_error) for schema in visible]


def _narrowed(
    schema: Any,
    narrowers: List[Any],
    on_error: Optional[Callable[[str, str, Exception], None]],
) -> Any:
    """``schema`` after every ``narrow_tool_schema`` hook; a hook that
    raises or returns nothing leaves it as it was."""
    for plugin in narrowers:
        try:
            narrowed = plugin.narrow_tool_schema(schema)
        except Exception as exc:
            if on_error is not None:
                on_error(schema.name, getattr(plugin, 'name', '?'), exc)
            continue
        if narrowed is not None:
            schema = narrowed
    return schema


def _is_visible(
    tool_name: str,
    filters: List[Any],
    on_error: Optional[Callable[[str, str, Exception], None]],
) -> bool:
    """``False`` iff some predicate in ``filters`` hides ``tool_name``."""
    for plugin in filters:
        try:
            if not plugin.is_tool_visible(tool_name):
                return False
        except Exception as exc:
            if on_error is not None:
                on_error(tool_name, getattr(plugin, 'name', '?'), exc)
    return True
