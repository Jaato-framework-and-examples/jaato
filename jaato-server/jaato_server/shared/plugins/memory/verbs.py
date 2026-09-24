"""The memory rail's verbs (#1232): one definition, served by whichever copy holds the store.

``memory`` is ``PLUGIN_TIER = "runner"``, so on a runner-served session --
the default -- the plugin the model drives lives in the RUNNER process.  The
daemon keeps a copy of its own (its registry discovers every plugin), and
that copy is the one the ``memory`` command's answer used to be filled from
while the command itself ran on the runner: the #1179 defect class, a
daemon-side copy no runner-served session touches.

So the verbs are answered HERE, by :func:`serve_memory_op`, and this function
is called from exactly two places:

* the runner, for the ``session.memory`` RPC (``server/runner/rpc.py``) --
  the default path;
* the daemon, only when there is NO runner at all (the embedded client,
  standalone WS), where the daemon's plugin IS the store.

Both reach the plugin through :func:`resolve_memory_plugin`, so "which copy
answers" has one rule and one spelling.  Every answer is a dict carrying
``ok``; a failure carries ``error`` and a ``category`` from
:data:`FAILURE_CATEGORIES`.  An empty store is ``ok=True`` with an empty
list, and nothing else is: an empty list is never how a failure is spelled,
because it reads as "nothing remembered".
"""

from typing import Any, Dict, List, Optional

#: The four operations the rail drives.  ``update`` covers approve and
#: dismiss (a ``maturity``) as well as the structured edit.
MEMORY_OPS = frozenset({"list", "get", "update", "delete"})

#: Fields a ``memory.update`` may change.  Deliberately not ``scope``,
#: ``confidence`` or ``evidence``: the rail edits what a person reads and
#: decides, and those three are the author's self-assessment.
EDITABLE_FIELDS = ("description", "content", "tags", "maturity")

#: Every ``category`` a failed answer may carry, whichever side produced it.
FAILURE_CATEGORIES = frozenset({
    "no_session",          # the caller is attached to no session
    "no_plugin",           # the session does not enable the memory plugin
    "runner_unreachable",  # the runner holding the store did not answer
    "not_found",           # no tier holds that id
    "invalid",             # the plugin's schema validator refused the edit
    "not_owner",           # the owner gate refused a mutation
    "unknown_op",          # not one of MEMORY_OPS
    "store_error",         # the plugin raised reading or writing the store
})


def failure(category: str, error: str) -> Dict[str, Any]:
    """A failed answer, in the one shape every consumer reads."""
    return {"ok": False, "category": category, "error": error}


def resolve_memory_plugin(registry: Any) -> Optional[Any]:
    """The memory plugin THIS registry holds and exposes, or ``None``.

    Exposed, not merely discovered: a discovered-but-unexposed plugin was
    never ``initialize()``d, so its storage is ``None`` and every read would
    answer as an empty store -- the one answer this module exists never to
    give.  ``None`` becomes ``category="no_plugin"``.
    """
    if registry is None:
        return None
    try:
        if not registry.is_exposed("memory"):
            return None
        plugin = registry.get_plugin("memory")
    except Exception:  # noqa: BLE001 -- a registry that cannot answer has no plugin
        return None
    if plugin is None or not callable(getattr(plugin, "memory_rows", None)):
        return None
    return plugin


def _fields(args: Dict[str, Any]) -> Dict[str, Any]:
    """The editable fields the caller actually supplied (``None`` = leave)."""
    fields = args.get("fields") or {}
    return {k: fields[k] for k in EDITABLE_FIELDS if fields.get(k) is not None}


def _serve_update(plugin: Any, memory_id: str, args: Dict[str, Any]) -> Dict[str, Any]:
    fields = _fields(args)
    if not fields:
        return failure("invalid", "memory.update: no field to change")
    tags = fields.get("tags")
    if tags is not None and not isinstance(tags, list):
        return failure("invalid", "tags must be a list")
    return plugin.edit_memory_structured(
        memory_id, curator=args.get("curator") or None, **fields)


def serve_memory_op(registry: Any, op: str, args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Answer one memory verb; a store that raises is a ``store_error`` answer.

    See :func:`_serve` for the verbs.  The plugin reads and writes files, and
    an exception escaping here would reach ``handle_request`` on the
    no-runner path and cost the caller its answer, so it is reported in the
    same shape every other failure takes.
    """
    try:
        return _serve(registry, op, args)
    except Exception as exc:  # noqa: BLE001 -- reported, never an empty list
        return failure("store_error", f"memory.{op}: {type(exc).__name__}: {exc}")


def _serve(registry: Any, op: str, args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Answer one memory verb from the plugin ``registry`` holds.

    Args:
        registry: The ``PluginRegistry`` of the process that holds the store.
        op: One of :data:`MEMORY_OPS`.
        args: ``{"session_id"}`` for ``list``; ``{"memory_id"}`` for the
            rest; ``update`` also ``{"fields": {...}, "curator": {...}}``.
            The OWNER gate is the daemon's and has already run -- this side
            is the plugin's, and does not second-guess who may curate.

    Returns:
        ``{"ok": True, ...}`` (``memories`` / ``memory`` / ``memory_id``)
        or :func:`failure`.
    """
    args = dict(args or {})
    if op not in MEMORY_OPS:
        return failure("unknown_op", f"unknown memory operation: {op!r}")
    plugin = resolve_memory_plugin(registry)
    if plugin is None:
        return failure("no_plugin", "the memory plugin is not enabled for this session")
    if op == "list":
        rows: List[Dict[str, Any]] = plugin.memory_rows(args.get("session_id") or None)
        return {"ok": True, "memories": rows}
    memory_id = str(args.get("memory_id") or "")
    if not memory_id:
        return failure("invalid", f"memory.{op}: memory_id is required")
    if op == "get":
        record = plugin.memory_record(memory_id)
        if record is None:
            return failure("not_found", f"Memory not found: {memory_id}")
        return {"ok": True, "memory": record}
    if op == "update":
        return _serve_update(plugin, memory_id, args)
    return plugin.remove_memory(memory_id)
