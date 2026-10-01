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

The predicates answer for "the current session" (#1195), so the caller
must have set the session ContextVar for the session it is serving before
calling this.  ``_get_tools_for_provider`` sets it explicitly; a tool
executor (``list_tools``) runs after ``_execute_single_tool`` has set it.
"""

from typing import Any, Callable, List, Optional


def filter_visible_tool_schemas(
    registry: Any,
    schemas: List[Any],
    on_error: Optional[Callable[[str, str, Exception], None]] = None,
) -> List[Any]:
    """Drop the schemas some exposed plugin's ``is_tool_visible`` hides,
    then apply every ``narrow_tool_schema`` hook to what is left.

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

    Returns:
        ``schemas`` itself when no plugin implements the predicate, else a
        new list holding the visible schemas in their original order.
    """
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
