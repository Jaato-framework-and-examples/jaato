"""Profiles plugin's :class:`BundleEntryHandler` implementation.

A *profile* is a subagent configuration (model + plugins +
system_instructions + GC strategy) stored as a single JSON or YAML
file. Profiles live in two tier roots:

* ``<workspace>/.jaato/profiles/`` (workspace tier)
* ``~/.jaato/profiles/`` (user tier)

This handler exposes profile management through the domain-agnostic
:class:`bundle_common.handler.BundleEntryHandler` protocol so the
top-level ``bundle`` command can list, add, eject, remove, pack, and
unpack profiles alongside references and any future kinds.

Phase 13 scope — what's implemented:

* :meth:`list_entries` — walks both tier roots and returns one
  :class:`BundleEntry` per discovered profile JSON. Workspace-tier
  profiles shadow user-tier profiles of the same id (matches the
  precedence already used by :func:`subagent.config.discover_profiles`).
* :meth:`find_entry` — id-based lookup with the same shadowing rules.
* :meth:`list_bundles` — returns ``[]``: profile bundle directories
  (sub-folders containing a manifest under ``.jaato/profiles/``)
  aren't supported on this build. The cross-kind ``bundle list``
  output therefore shows a ``profiles:`` section with no rows for
  bundles, only entries — exactly what the operator should see.
* :meth:`reload_catalog` — re-scans both tier roots.
* :meth:`delete_entry` — removes a profile JSON / YAML from disk.

Methods that don't make sense for the current profile-on-disk shape
(``move_entry_to_bundle`` / ``move_entry_to_free`` /
:meth:`reconcile_bundle` / :meth:`create_empty_bundle` /
:meth:`delete_bundle`) raise :class:`NotImplementedError` with a
domain-specific message. The dispatcher already surfaces those
errors as friendly user messages, so the bundle command reports
"profiles kind doesn't support 'bundle add' (yet — see issue #...)"
rather than crashing.

A follow-up commit will lift profile bundling into a real shape:
profile *bundles* would be subdirectories under ``.jaato/profiles/``
containing a manifest plus their member profile files, mirroring the
references-side layout. The existing
:func:`subagent.config.discover_profiles` already supports nested
directory walking; the missing pieces are manifest-aware load logic
and per-bundle membership tracking on the in-memory profile records.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from ..bundle_common.bundle import (
    BUNDLE_TIER_USER,
    BUNDLE_TIER_WORKSPACE,
    Bundle,
)
from ..bundle_common.handler import BundleEntry, BundleEntryHandler

if TYPE_CHECKING:  # pragma: no cover - import only for type hints
    from .plugin import SubagentPlugin


logger = logging.getLogger(__name__)


_PROFILES_KIND = "profiles"
_PROFILES_DOMAIN_SUBPATH = Path(".jaato") / "profiles"
_PROFILE_EXTENSIONS = (".json", ".yaml", ".yml")


class ProfilesEntryHandler(BundleEntryHandler):
    """:class:`BundleEntryHandler` adapter for the subagent / profiles plugin.

    Holds a strong reference to the subagent plugin so the handler
    stays consistent with the plugin's lifecycle: re-initialization
    replaces the registered handler with one bound to the new state.
    """

    def __init__(self, plugin: "SubagentPlugin") -> None:
        self._plugin = plugin

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def kind(self) -> str:
        return _PROFILES_KIND

    @property
    def domain_subpath(self) -> Path:
        return _PROFILES_DOMAIN_SUBPATH

    # ------------------------------------------------------------------
    # Enumeration
    # ------------------------------------------------------------------

    def list_entries(self) -> List[BundleEntry]:
        """Return one :class:`BundleEntry` per discovered profile.

        Walks the workspace tier first, then the user tier; on a name
        collision the workspace copy wins (matches the precedence
        used by :func:`subagent.config.discover_profiles`).
        """
        seen_ids: set = set()
        results: List[BundleEntry] = []

        for tier, tier_root in self._tier_roots():
            if not tier_root.is_dir():
                continue
            for entry_path in sorted(tier_root.iterdir()):
                if not entry_path.is_file():
                    continue
                if entry_path.suffix not in _PROFILE_EXTENSIONS:
                    continue
                # Profile id == filename stem; matches the convention
                # used by ``subagent.config._scan_profiles_dir``.
                entry_id = entry_path.stem
                if entry_id in seen_ids:
                    # Workspace shadows user.
                    continue
                seen_ids.add(entry_id)
                results.append(BundleEntry(
                    id=entry_id,
                    kind=_PROFILES_KIND,
                    file_path=entry_path,
                    bundle_name="",
                    bundle_tier=tier,
                ))
        return results

    def find_entry(self, entry_id: str) -> Optional[BundleEntry]:
        """Optimized id lookup — checks each tier root with the same
        workspace-shadows-user precedence as :meth:`list_entries`."""
        for tier, tier_root in self._tier_roots():
            if not tier_root.is_dir():
                continue
            for ext in _PROFILE_EXTENSIONS:
                candidate = tier_root / f"{entry_id}{ext}"
                if candidate.is_file():
                    return BundleEntry(
                        id=entry_id,
                        kind=_PROFILES_KIND,
                        file_path=candidate,
                        bundle_name="",
                        bundle_tier=tier,
                    )
        return None

    def list_bundles(self) -> List[Bundle]:
        """Profile *bundles* (sub-directories with manifests) aren't
        supported yet. Returns the empty list so cross-kind aggregation
        still surfaces a "profiles:" section in ``bundle list`` — just
        with no bundle rows. Entries themselves still appear.

        See the module docstring for the planned follow-up shape.
        """
        return []

    # ------------------------------------------------------------------
    # Catalog refresh
    # ------------------------------------------------------------------

    def reload_catalog(self) -> None:
        """Re-discover profiles from disk.

        Delegates to the subagent plugin's existing reload path when
        available; falls back to a no-op when the plugin doesn't
        expose one (the next ``list_entries`` call still re-walks the
        directories, so on-disk state is always reflected).
        """
        reload = getattr(self._plugin, "reload_profiles", None)
        if callable(reload):
            try:
                reload()
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("subagent reload_profiles failed: %s", e)

    # ------------------------------------------------------------------
    # Membership mutations — only delete_entry is implemented in
    # Phase 13. The rest raise NotImplementedError so the dispatcher
    # surfaces a friendly "kind doesn't support this yet" message
    # instead of crashing.
    # ------------------------------------------------------------------

    def delete_entry(self, entry: BundleEntry) -> None:
        """Permanently delete a profile's JSON/YAML file from disk."""
        try:
            entry.file_path.unlink()
        except FileNotFoundError:
            logger.debug("delete_entry: %s already gone", entry.file_path)

    def move_entry_to_bundle(self, entry, target_bundle):
        raise NotImplementedError(
            "kind 'profiles' does not yet support 'bundle add' — profile "
            "bundles (sub-directories with manifests) aren't implemented "
            "on this build"
        )

    def move_entry_to_free(self, entry, target_tier):
        raise NotImplementedError(
            "kind 'profiles' does not yet support 'bundle eject' — profiles "
            "are already free entries (not in any bundle)"
        )

    def reconcile_bundle(self, bundle: Bundle):
        # No bundles, no reconcile.
        return None

    def create_empty_bundle(
        self,
        name: str,
        tier: str,
        *,
        workspace_path: Optional[Path] = None,
        user_home: Optional[Path] = None,
    ) -> Bundle:
        raise NotImplementedError(
            "kind 'profiles' does not yet support 'bundle create' — profile "
            "bundles are planned but not implemented on this build"
        )

    def delete_bundle(self, bundle: Bundle, *, force: bool = False) -> None:
        raise NotImplementedError(
            "kind 'profiles' does not yet support 'bundle delete' — there "
            "are no profile bundles to delete on this build"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tier_roots(self) -> List[tuple]:
        """Return the ``(tier, root_dir)`` pairs to scan in precedence order."""
        ws = getattr(self._plugin, "_workspace_path", None)
        roots: List[tuple] = []
        if ws:
            roots.append((BUNDLE_TIER_WORKSPACE, Path(ws) / _PROFILES_DOMAIN_SUBPATH))
        roots.append((BUNDLE_TIER_USER, Path.home() / _PROFILES_DOMAIN_SUBPATH))
        return roots
