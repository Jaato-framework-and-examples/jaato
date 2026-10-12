"""Hashed tool and category ids, shown by the names a person knows (#1650).

Tool names reach the MODEL as hash-derived ids (``t_a3f2b1c0`` for a tool,
``c_7e4d9f12`` for a category; ``jaato_server/shared/tool_id_map.py``)
because upstreams restrict tool-name characters.  So the arguments of a
discovery call arrive as ids -- ``list_tools(category_id="c_bbc5e661")`` --
which mean nothing to a reader.  The daemon sends the mapping so clients
need not reverse-engineer it, in two places:

- ``ToolIdRegistryEvent`` (``tools.id_registry``): the FULL current set
  each time, never a delta.  Sent after tool configuration and again when
  deferred tools activate.
- ``SessionInfoEvent.tool_id_mappings``: the same map in the attach
  snapshot.

The agreement is that anything user-facing shows the human name.  This
module is that rule, once, so every client does not write its own copy
(the TUI and the web coder each had one).  The rules:

- **The mapping is replaced wholesale** on each ``ToolIdRegistryEvent``,
  and seeded from ``session.info``'s ``tool_id_mappings`` when that is
  non-empty (an empty one from a daemon that has not configured tools yet
  says nothing, and ids are a pure function of the name, so a kept entry
  is never wrong).
- **Resolution walks dicts and lists and replaces only a string that IS an
  id in the map.**  An id the map does not name is left as it is --
  showing the id is honest, inventing a name is not -- and a string that
  merely contains one is untouched.
- **Resolve at render time, not when the event is reduced**: the registry
  can arrive after the call that used an id, so a row rendered from the
  stored arguments picks the name up whenever it lands.

Mirror: ``jaato-sdk-ts/src/toolIds.ts``.  Same rules, same names.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

__all__ = [
    "ToolIdNames",
    "resolve_tool_ids",
    "tool_id_mappings",
]


def tool_id_mappings(raw: Any) -> Optional[Dict[str, str]]:
    """Read a mapping off an event field, keeping only ``str -> str`` entries.

    Returns ``None`` when ``raw`` is not a mapping at all, so a caller can
    tell "no mapping on this event" from "an empty one".
    """
    if not isinstance(raw, Mapping):
        return None
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def resolve_tool_ids(value: Any, names: Mapping[str, str]) -> Any:
    """Replace every string in ``value`` that IS an id in ``names``.

    Walks dicts (values only, keys are argument names) and lists/tuples;
    a string is replaced only on an exact match, so an unknown id and a
    string containing an id pass through unchanged.  Returns a new
    structure; ``value`` is never mutated.
    """
    if isinstance(value, str):
        return names.get(value, value)
    if isinstance(value, list):
        return [resolve_tool_ids(v, names) for v in value]
    if isinstance(value, tuple):
        return tuple(resolve_tool_ids(v, names) for v in value)
    if isinstance(value, dict):
        return {k: resolve_tool_ids(v, names) for k, v in value.items()}
    return value


def _field(event: Any, name: str) -> Any:
    """Read ``name`` off a pydantic event or a raw wire dict."""
    if isinstance(event, Mapping):
        return event.get(name)
    return getattr(event, name, None)


def _type_of(event: Any) -> Optional[str]:
    """The wire ``type`` string of a pydantic event or a raw wire dict."""
    t = _field(event, "type")
    value = getattr(t, "value", t)
    return value if isinstance(value, str) else None


_REGISTRY_TYPE = "tools.id_registry"
_SESSION_INFO_TYPE = "session.info"


class ToolIdNames:
    """The latest id -> name mapping, kept current from the event stream.

    Lifecycle: start empty (or from ``mappings``); every
    :meth:`observe` of a ``ToolIdRegistryEvent`` REPLACES the map, every
    observe of a ``SessionInfoEvent`` with non-empty ``tool_id_mappings``
    replaces it too; anything else is ignored.  :meth:`track` does the
    observing for you by subscribing to an ``IPCClient`` (or anything with
    the same ``subscribe(event_type, handler)`` shape).

    Read side: :meth:`resolve` for argument structures, :meth:`name` for
    one value such as a ``tool_name``.  Call them when RENDERING, since a
    mapping may arrive after the call that used an id.
    """

    def __init__(self, mappings: Optional[Mapping[str, str]] = None) -> None:
        self._names: Dict[str, str] = dict(tool_id_mappings(mappings) or {})

    @property
    def mappings(self) -> Dict[str, str]:
        """A copy of the current map."""
        return dict(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def replace(self, mappings: Mapping[str, str]) -> None:
        """Replace the map wholesale (the daemon always sends the full set)."""
        self._names = dict(tool_id_mappings(mappings) or {})

    def observe(self, event: Any) -> bool:
        """Update from one event; return whether the map changed source.

        Accepts a pydantic event or a raw wire dict.  ``tools.id_registry``
        replaces the map whatever it holds; ``session.info`` replaces it
        only when its ``tool_id_mappings`` is non-empty.
        """
        kind = _type_of(event)
        if kind == _REGISTRY_TYPE:
            names = tool_id_mappings(_field(event, "mappings"))
            if names is None:
                return False
            self._names = names
            return True
        if kind == _SESSION_INFO_TYPE:
            names = tool_id_mappings(_field(event, "tool_id_mappings"))
            if not names:
                return False
            self._names = names
            return True
        return False

    def name(self, value: str) -> str:
        """The name for ``value`` when it is a known id, else ``value``."""
        return self._names.get(value, value)

    def resolve(self, value: Any) -> Any:
        """:func:`resolve_tool_ids` over the current map."""
        return resolve_tool_ids(value, self._names)

    def track(self, client: Any) -> Callable[[], None]:
        """Keep this map current from ``client``'s event stream.

        Subscribes to ``TOOL_ID_REGISTRY`` and ``SESSION_INFO`` and returns
        one unsubscribe for both.  Subscribe before connecting/attaching so
        the attach snapshot is not missed.
        """
        from jaato_sdk.events import EventType

        unsubs = [
            client.subscribe(EventType.TOOL_ID_REGISTRY, self.observe),
            client.subscribe(EventType.SESSION_INFO, self.observe),
        ]

        def _unsubscribe() -> None:
            for u in unsubs:
                u()

        return _unsubscribe
