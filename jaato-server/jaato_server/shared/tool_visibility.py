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
    """Drop the schemas some exposed plugin's ``is_tool_visible`` hides.

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
    filters = []
    for name in exposed_names:
        plugin = registry.get_plugin(name)
        if plugin is not None and hasattr(plugin, 'is_tool_visible'):
            filters.append(plugin)
    if not filters:
        return schemas
    return [
        schema for schema in schemas
        if _is_visible(schema.name, filters, on_error)
    ]


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
