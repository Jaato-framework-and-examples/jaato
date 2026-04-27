"""Plugin contract for participating in bundle management.

A **bundle entry handler** is a plugin's interface to the shared bundle
subsystem. The bundle commands (``add``, ``eject``, ``remove``,
``reconcile``, ``pack``, ``unpack``) don't know anything about
references, agents, tasks, profiles, or services on their own — they
ask each registered handler for the operation it owns.

This module defines:

* :class:`BundleEntry` — a tier-/bundle-aware view of one item that a
  handler owns (a reference, an agent definition, a task prompt, etc.).
* :class:`BundleEntryHandler` — the protocol every domain plugin
  implements once and registers exactly once with the global
  :class:`BundleEntryRegistry`.
* :class:`BundleEntryRegistry` — process-global dispatcher keyed by
  ``handler.kind`` (``"references"``, ``"agents"``, ``"tasks"``,
  ``"profiles"``, ``"services"``).

The ``kind`` string is the disambiguator for the user-facing
``<kind>:<id>`` syntax of the future top-level ``bundle`` command
(``bundle add references:api-spec --to teammate``,
``bundle add agents:reviewer --to teammate``). Names are deliberately
required to be globally unique within a kind but may collide across
kinds — ``references:weekly`` and ``tasks:weekly`` are distinct entries.

The protocol is intentionally minimal in this commit: the surface area
needed to back the existing references CRUD verbs (``list``, ``find``,
``move``, ``eject``, ``remove``, ``reload``, ``reconcile``) plus
forward-looking pack/unpack hooks (``collect_pack_payloads``,
``rewrite_for_pack``) with default no-op implementations so non-LOCAL
domains don't have to think about payload bundling. Pack/unpack itself
is rewired to call through this registry in a later commit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

from .bundle import Bundle


@dataclass(frozen=True)
class BundleEntry:
    """A tier- and bundle-aware view of one handler-owned item.

    Returned by :meth:`BundleEntryHandler.list_entries` and
    :meth:`BundleEntryHandler.find_entry`. Snapshot semantics —
    fields are valid at read time; mutating commands re-derive
    them rather than reusing a stale ``BundleEntry``.

    Attributes:
        id: Domain-unique identifier — e.g. a reference's ``id``,
            an agent's filename stem, a task name. Together with
            ``kind`` it uniquely addresses an entry across all
            handlers. Two handlers may use the same ``id`` because
            their ``kind`` values differ.
        kind: Owning handler's ``kind`` string. Mirrors
            ``BundleEntryHandler.kind`` so the entry is
            self-describing and can be passed back to the registry
            for downstream dispatch.
        file_path: Absolute path to the on-disk file backing the
            entry. The file is the unit of distribution: pack reads
            from it, unpack writes to it, ``bundle remove`` deletes
            it. ``None`` is not allowed — orphan entries (in the
            catalog but not on disk) are not surfaced as ``BundleEntry``.
        bundle_name: Name of the bundle that currently owns the
            entry. Empty string when the entry is *free* (loaded by
            the catalog but not enclosed by any bundle's manifest).
            Mirrors :attr:`Bundle.name` semantics.
        bundle_tier: Tier of the owning bundle, one of
            :data:`bundle_common.bundle.BUNDLE_TIER_WORKSPACE` or
            :data:`bundle_common.bundle.BUNDLE_TIER_USER`. For free
            entries this is the tier the file was discovered in.
    """

    id: str
    kind: str
    file_path: Path
    bundle_name: str
    bundle_tier: str


@runtime_checkable
class BundleEntryHandler(Protocol):
    """Protocol every bundle-aware plugin implements.

    Implementations are stateless w.r.t. the handler instance — state
    lives on the plugin the handler delegates to. The handler is
    therefore cheap to construct and is registered once per plugin
    activation (typically inside ``initialize()``).

    Implementers should treat the methods as the *only* surface the
    bundle subsystem will use; per-domain CRUD that doesn't fit any
    method below stays inside the plugin itself.
    """

    @property
    def kind(self) -> str:
        """Stable domain identifier — ``"references"``, ``"agents"``,
        ``"tasks"``, ``"profiles"``, ``"services"``. Used as the
        ``<kind>`` prefix in user-facing ``bundle add <kind>:<id>``
        invocations and as the per-kind subdirectory name inside
        packed archives.
        """
        ...

    @property
    def domain_subpath(self) -> Path:
        """Tier-root subdirectory where this domain's bundles live.

        For the references plugin this is ``Path(".jaato/references")``;
        for an agents plugin it would be ``Path(".jaato/agents")``.
        Joined with ``<workspace>`` and ``Path.home()`` by
        :func:`bundle_common.bundle.resolve_bundle_roots`.
        """
        ...

    def list_entries(self) -> List[BundleEntry]:
        """Enumerate every entry this handler currently owns.

        Spans every loaded bundle and any free (unbundled) entries
        the catalog knows about. The bundle subsystem uses this for
        completion candidates and id-collision detection during
        ``bundle add``.
        """
        ...

    def find_entry(self, entry_id: str) -> Optional[BundleEntry]:
        """Look up a single entry by its id, or ``None`` if missing.

        Equivalent to ``next((e for e in self.list_entries() if
        e.id == entry_id), None)`` but handlers should provide an
        index-based fast path when their plugin already maintains one.
        """
        ...

    def move_entry_to_bundle(
        self,
        entry: BundleEntry,
        target_bundle: Bundle,
    ) -> Path:
        """Physically relocate ``entry``'s file into ``target_bundle``'s
        directory.

        Implementations may also move auxiliary sibling files (an
        agent definition could carry related fragments alongside the
        primary file). Returns the new absolute file path on success.
        Raises a domain-specific exception on failure (e.g. a name
        collision in the target dir).
        """
        ...

    def move_entry_to_free(
        self,
        entry: BundleEntry,
        target_tier: str,
    ) -> Path:
        """Move ``entry`` out of its current bundle into the
        tier-root directory for ``target_tier``.

        Mirror of :meth:`move_entry_to_bundle` for the eject case.
        The target tier must be one of
        :data:`bundle_common.bundle.VALID_BUNDLE_TIERS`. Returns the
        new absolute file path.
        """
        ...

    def delete_entry(self, entry: BundleEntry) -> None:
        """Permanently delete ``entry``'s file (and any auxiliary
        files the handler tracks alongside it)."""
        ...

    def reload_catalog(self) -> None:
        """Refresh the plugin's in-memory catalog from disk.

        Called by the bundle subsystem after each membership-changing
        operation so subsequent commands see the new state without
        forcing the operator to ``references reload`` manually.
        """
        ...

    def reconcile_bundle(self, bundle: Bundle) -> Optional[Any]:
        """Sync ``bundle``'s manifest (and sidecar, if any) with the
        live entries the handler reports.

        References use this to re-embed any drifted vectors;
        non-embedded handlers (agents/tasks/profiles/services) just
        update the manifest's ``rows`` to match the on-disk file
        list. Returns a domain-specific result for logging or
        ``None`` when nothing to report.
        """
        ...

    # ------------------------------------------------------------------
    # Optional pack/unpack hooks. Default implementations cover the
    # common case where an entry is fully self-contained (URL/MCP/INLINE
    # references; pure-config kinds like profiles/services). Domains
    # whose entries point at external payloads (LOCAL references,
    # markdown system_instructions for agents) override these.
    # ------------------------------------------------------------------

    def collect_pack_payloads(
        self, entry: BundleEntry
    ) -> List[Tuple[Path, str]]:
        """Return external payloads to be archived alongside the entry.

        Each tuple is ``(absolute_source_path, archive_relative_path)``;
        ``archive_relative_path`` is relative to the per-kind payload
        root inside the archive (e.g. ``payload/<slot>/<basename>``
        for references).

        Default: no external payloads.
        """
        return []

    def rewrite_for_pack(
        self,
        entry: BundleEntry,
        payload_map: Dict[Path, str],
    ) -> None:
        """Mutate ``entry``'s on-disk serialization for inclusion in
        an archive.

        ``payload_map`` maps original source paths to their archive-
        relative destinations (the inverse of
        :meth:`collect_pack_payloads`). References use this to
        rewrite each LOCAL ref's ``path`` field to the archive
        location and drop ``resolved_path``. Domains without external
        payloads can leave the default no-op.

        The method runs against the *staged* copy of the entry's
        file, not the live workspace one — handlers should not
        worry about side-effects on the operator's working tree.
        """
        return None


class BundleEntryRegistry:
    """Process-global registry of :class:`BundleEntryHandler` instances.

    A single instance lives at module level (:data:`registry`); plugins
    register themselves during ``initialize()`` and unregister during
    ``shutdown()``. The bundle command surface dispatches into the
    registered handlers via :meth:`get` (single-kind ops like
    ``add``/``eject``/``remove``) and :meth:`all_handlers` (multi-kind
    ops like ``pack``/``list``).

    Registration is idempotent within a session — re-registering the
    same ``kind`` overwrites the previous handler, matching how plugins
    can be re-initialized when a session is reset. Concurrent multi-
    plugin sessions don't share state because plugin instances aren't
    shared across sessions in jaato.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, BundleEntryHandler] = {}

    def register(self, handler: BundleEntryHandler) -> None:
        """Insert or replace the handler for ``handler.kind``.

        Replacement is intentional: a plugin re-initialized within
        the same process drops its old handler and inserts a fresh
        one bound to the new state.
        """
        kind = handler.kind
        if not isinstance(kind, str) or not kind:
            raise ValueError("BundleEntryHandler.kind must be a non-empty string")
        self._handlers[kind] = handler

    def unregister(self, kind: str) -> None:
        """Remove the handler for ``kind`` if present.

        Idempotent — unregistering an unknown kind is a no-op so that
        plugin shutdown doesn't have to defensively check whether
        registration succeeded.
        """
        self._handlers.pop(kind, None)

    def get(self, kind: str) -> Optional[BundleEntryHandler]:
        """Return the handler for ``kind`` or ``None`` if not registered."""
        return self._handlers.get(kind)

    def kinds(self) -> List[str]:
        """Return the registered kinds in deterministic (sorted) order."""
        return sorted(self._handlers.keys())

    def all_handlers(self) -> List[BundleEntryHandler]:
        """Return every registered handler, sorted by kind for stable output."""
        return [self._handlers[k] for k in self.kinds()]

    def list_all_entries(self) -> List[BundleEntry]:
        """Aggregate :meth:`BundleEntryHandler.list_entries` across kinds.

        Useful for ``bundle list`` and similar multi-domain commands.
        Order is by kind (sorted) then by handler-defined entry order.
        """
        result: List[BundleEntry] = []
        for handler in self.all_handlers():
            result.extend(handler.list_entries())
        return result

    def find_entry(
        self, kind: str, entry_id: str
    ) -> Optional[BundleEntry]:
        """Resolve ``<kind>:<id>`` to a :class:`BundleEntry`.

        ``None`` when the kind isn't registered or the id is unknown
        within that kind. Use this when the user has explicitly
        qualified the entry; for unqualified ids fall back to
        :meth:`find_entry_any_kind`.
        """
        handler = self._handlers.get(kind)
        if handler is None:
            return None
        return handler.find_entry(entry_id)


# Process-global registry. Plugins import and call ``registry.register(self)``
# in ``initialize()`` and ``registry.unregister(self.kind)`` in ``shutdown()``.
# Tests construct their own :class:`BundleEntryRegistry` instances rather
# than mutating this singleton — this keeps test isolation clean while
# preserving the simple ``from … import registry`` ergonomics for plugins.
registry = BundleEntryRegistry()
