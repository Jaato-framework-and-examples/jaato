"""References plugin's :class:`BundleEntryHandler` implementation.

This module wraps the existing :class:`ReferencesPlugin` and exposes
its bundle-relevant operations through the domain-agnostic
:class:`bundle_common.handler.BundleEntryHandler` protocol. The plugin
constructs one of these in :meth:`ReferencesPlugin.initialize` and
registers it with :data:`bundle_common.handler.registry`; nothing
inside the plugin calls the handler — it exists so future bundle CRUD
/ pack / unpack code (Phase 8 onwards) can dispatch through the
registry without knowing the references plugin specifically.

Behavioral contract:

* The handler's state lives entirely on the wrapped
  :class:`ReferencesPlugin` instance. Re-initialization replaces the
  registered handler with a fresh one bound to the new plugin state.
* All file-locating, moving, deleting, and reconciling delegates to
  existing plugin methods (``_locate_ref_file``,
  ``_post_membership_change`` does *not* reload here — that's the
  plugin's job after dispatch). The handler is a thin adapter, not a
  re-implementation.
* Pack/unpack hooks honour the references-specific contract that
  :class:`bundle_common.handler.BundleEntryHandler` documents:
  LOCAL refs contribute their resolved file as a payload and have
  their ``path`` rewritten on pack; URL/MCP/INLINE refs do nothing
  beyond the default.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ..bundle_common.bundle import (
    BUNDLE_TIER_USER,
    BUNDLE_TIER_WORKSPACE,
    Bundle,
)
from ..bundle_common.handler import BundleEntry, BundleEntryHandler
from .models import SourceType

if TYPE_CHECKING:  # pragma: no cover - import only for type hints
    from .plugin import ReferencesPlugin


logger = logging.getLogger(__name__)


_REFERENCES_KIND = "references"
_REFERENCES_DOMAIN_SUBPATH = Path(".jaato") / "references"


class ReferencesEntryHandler(BundleEntryHandler):
    """:class:`BundleEntryHandler` adapter for the references plugin.

    Holds a strong reference to the plugin so the handler stays
    consistent with the plugin's lifecycle: when the plugin is
    re-initialized, a fresh handler is created and re-registered,
    transparently overwriting the stale one.
    """

    def __init__(self, plugin: "ReferencesPlugin") -> None:
        self._plugin = plugin

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def kind(self) -> str:
        return _REFERENCES_KIND

    @property
    def domain_subpath(self) -> Path:
        return _REFERENCES_DOMAIN_SUBPATH

    # ------------------------------------------------------------------
    # Enumeration
    # ------------------------------------------------------------------

    def list_entries(self) -> List[BundleEntry]:
        """Return one :class:`BundleEntry` per loaded reference source.

        Sources whose backing JSON cannot be located on disk (catalog
        and disk diverged, e.g. someone deleted the file under us)
        are omitted — the bundle subsystem only operates on entries
        that have a concrete file to act on. The ``references reload``
        command exists to re-sync after such drift.
        """
        entries: List[BundleEntry] = []
        for source in self._plugin._sources:
            file_path = self._plugin._locate_ref_file(source)
            if file_path is None:
                continue
            entries.append(self._build_entry(source.id, source.bundle_name, file_path))
        return entries

    def list_bundles(self) -> List[Bundle]:
        """Return every references bundle (workspace + user tiers).

        Mirrors ``self._plugin._bundles``, which is already maintained
        in workspace-then-user discovery order by
        :meth:`ReferencesPlugin._discover_and_load_bundles`.
        """
        return list(self._plugin._bundles)

    def find_entry(self, entry_id: str) -> Optional[BundleEntry]:
        """Resolve ``entry_id`` to a :class:`BundleEntry` or ``None``.

        Optimized vs. ``next(... in list_entries())`` because the
        plugin already keeps an id-indexed source list — only one
        ``_locate_ref_file`` call is needed.
        """
        source = next(
            (s for s in self._plugin._sources if s.id == entry_id), None,
        )
        if source is None:
            return None
        file_path = self._plugin._locate_ref_file(source)
        if file_path is None:
            return None
        return self._build_entry(source.id, source.bundle_name, file_path)

    # ------------------------------------------------------------------
    # Membership mutations
    # ------------------------------------------------------------------

    def move_entry_to_bundle(
        self, entry: BundleEntry, target_bundle: Bundle,
    ) -> Path:
        """Relocate ``entry``'s JSON file into ``target_bundle.directory``.

        Raises :class:`FileExistsError` when the target directory
        already contains a file with the same basename — bundle
        commands should surface this as a user-facing error rather
        than silently merging.
        """
        dest_file = target_bundle.directory / entry.file_path.name
        if dest_file.exists():
            raise FileExistsError(
                f"target bundle '{target_bundle.qualified_ref}' already has a "
                f"file named {entry.file_path.name!r}"
            )
        target_bundle.directory.mkdir(parents=True, exist_ok=True)
        shutil.move(str(entry.file_path), str(dest_file))
        return dest_file

    def move_entry_to_free(
        self, entry: BundleEntry, target_tier: str,
    ) -> Path:
        """Move ``entry`` from its current bundle to the tier root.

        ``target_tier`` is typically the same as ``entry.bundle_tier``
        (eject within the same tier), but the API permits cross-tier
        moves for completeness. The plugin currently only loads free
        references at the workspace tier, so ``target_tier == "user"``
        will produce a file the catalog won't pick up until proper
        user-tier free-ref discovery is added.
        """
        tier_root = self._plugin._tier_root(target_tier)
        if tier_root is None:
            raise ValueError(
                f"cannot resolve tier root for {target_tier!r} — "
                f"workspace_path may be unknown"
            )
        dest_file = tier_root / entry.file_path.name
        if dest_file.exists():
            raise FileExistsError(
                f"tier root {tier_root} already has a file named "
                f"{entry.file_path.name!r}"
            )
        tier_root.mkdir(parents=True, exist_ok=True)
        shutil.move(str(entry.file_path), str(dest_file))
        return dest_file

    def delete_entry(self, entry: BundleEntry) -> None:
        """Permanently delete ``entry``'s JSON file from disk."""
        try:
            entry.file_path.unlink()
        except FileNotFoundError:
            # Race against a concurrent reload; treat as already-done.
            logger.debug(
                "delete_entry: %s already gone", entry.file_path,
            )

    # ------------------------------------------------------------------
    # Catalog & sidecar synchronization
    # ------------------------------------------------------------------

    def reload_catalog(self) -> None:
        """Refresh the plugin's master catalog + bundle list.

        Mirrors what ``_post_membership_change`` does internally,
        without the additional reconcile work — the bundle subsystem
        calls :meth:`reconcile_bundle` separately when needed.
        """
        if self._plugin._workspace_path:
            self._plugin._reload_catalog(self._plugin._workspace_path)
        self._plugin._discover_and_load_bundles()

    def reconcile_bundle(self, bundle: Bundle):
        """Run the references-specific reconcile pass for ``bundle``.

        Also re-attaches the bundle's semantic matcher when an
        embedding provider is available so the next semantic query
        sees the freshly written sidecar. Skipping the re-attach
        would leave the matcher pointing at the pre-reconcile sidecar
        and silently degrade match quality.

        Imported lazily because :mod:`shared.plugins.references.reconcile`
        depends on numpy via the embedding provider — keeping the
        import lazy means handler construction stays cheap and tests
        without numpy still load this module.
        """
        from .reconcile import reconcile_bundle as _reconcile_bundle  # local

        result = _reconcile_bundle(
            bundle, self._plugin._sources, self._plugin._embedding_provider,
        )
        if self._plugin._embedding_provider is not None and hasattr(
            self._plugin, "_attach_matcher",
        ):
            bundle.matcher = self._plugin._attach_matcher(
                bundle, self._plugin._embedding_provider.model_name,
            )
            # Keep the legacy ``self._semantic_matcher`` shim in sync
            # only when the workspace-tier root reconciles.
            if (
                bundle.name == ""
                and bundle.tier == BUNDLE_TIER_WORKSPACE
            ):
                self._plugin._semantic_matcher = bundle.matcher
        return result

    def delete_bundle(self, bundle: Bundle, *, force: bool = False) -> None:
        """Remove a references bundle's directory (or root contents).

        Honours the same non-empty-without-force contract documented
        on the protocol. After deletion the in-memory catalog is
        scrubbed of the bundle's sources so the very next handler
        call doesn't return stale entries — the dispatcher's follow-up
        :meth:`reload_catalog` is still safe but lets us minimize the
        window where state is inconsistent.
        """
        from ..bundle_common.bundle import (
            EMBEDDING_CONFIG_FILENAME as _MANIFEST,
            ROOT_BUNDLE_NAME as _ROOT,
        )

        # Non-emptiness check: any rows OR any *.json reference files
        # (excluding the manifest itself).
        has_rows = bool(bundle.embedding_rows)
        has_ref_files = any(
            p.name != _MANIFEST
            for p in bundle.directory.glob("*.json")
        )
        if (has_rows or has_ref_files) and not force:
            raise ValueError(
                f"bundle '{bundle.qualified_ref}' is not empty "
                f"({len(bundle.embedding_rows)} row(s)); pass force=True "
                f"to delete anyway"
            )

        # Drop the bundle's sources from the live catalog before we
        # mutate disk — keeps state consistent if the rmtree raises.
        self._plugin._sources = [
            s for s in self._plugin._sources if s.bundle_name != bundle.name
        ]

        if bundle.name == _ROOT:
            # Root bundle: remove only the bundle artefacts so other
            # sub-bundles living alongside the root manifest survive.
            for p in bundle.directory.glob("*.json"):
                p.unlink()
            for p in bundle.directory.glob("*.npy"):
                p.unlink()
            for p in bundle.directory.glob("*.npy.lock"):
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
            payload = bundle.directory / "payload"
            if payload.is_dir():
                shutil.rmtree(payload)
        else:
            shutil.rmtree(bundle.directory)

    def create_empty_bundle(
        self,
        name: str,
        tier: str,
        *,
        workspace_path: Optional[Path] = None,
        user_home: Optional[Path] = None,
    ) -> Bundle:
        """Write a fresh references manifest at the chosen tier root.

        The bundle's embedding model and dimensions are inherited from
        the active embedding provider; without a provider the call
        fails because a sidecar can't be written meaningfully. The
        wrapped plugin is *not* required to be the workspace-loaded
        instance — ``workspace_path`` resolves the tier root directly,
        making this method usable from any context that has the
        coordinates.
        """
        from ..bundle_common.bundle import (
            BUNDLE_TIER_USER as _USER,
            BUNDLE_TIER_WORKSPACE as _WS,
            EMBEDDING_CONFIG_FILENAME as _MANIFEST,
            ROOT_BUNDLE_NAME as _ROOT,
            VALID_BUNDLE_TIERS as _TIERS,
        )

        if tier not in _TIERS:
            raise ValueError(
                f"unknown tier {tier!r}; expected one of {', '.join(_TIERS)}"
            )
        provider = self._plugin._embedding_provider
        if provider is None:
            raise RuntimeError(
                "references kind requires an embedding provider to create "
                "an empty bundle (the manifest declares model + dimensions)"
            )
        if not provider.available:
            provider.load_model()
        dimensions = getattr(provider, "dimensions", None)
        if not isinstance(dimensions, int) or dimensions <= 0:
            raise RuntimeError(
                "embedding provider did not report a valid dimension; "
                "cannot write a bundle manifest without it"
            )

        # Resolve the destination directory.
        if tier == _WS:
            if workspace_path is None:
                raise ValueError(
                    "tier='workspace' requires workspace_path to be set"
                )
            tier_root = Path(workspace_path) / self.domain_subpath
        else:
            home = user_home if user_home is not None else Path.home()
            tier_root = home / self.domain_subpath
        bundle_dir = tier_root if name == _ROOT else tier_root / name
        bundle_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = bundle_dir / _MANIFEST
        if manifest_path.is_file():
            raise FileExistsError(
                f"manifest already exists at {manifest_path}"
            )
        sidecar_name = "references.embeddings.npy"
        manifest_path.write_text(
            json.dumps({
                "embedding_model": provider.model_name,
                "embedding_dimensions": int(dimensions),
                "embedding_sidecar": sidecar_name,
                "rows": [],
            }, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return Bundle(
            name=name,
            directory=bundle_dir.resolve(),
            embedding_model=provider.model_name,
            embedding_dimensions=int(dimensions),
            embedding_sidecar=sidecar_name,
            embedding_rows=[],
            tier=tier,
        )

    # ------------------------------------------------------------------
    # Pack / unpack hooks
    # ------------------------------------------------------------------

    def collect_pack_payloads(
        self, entry: BundleEntry,
    ) -> List[Tuple[Path, str]]:
        """Return the LOCAL ref's payload, if any.

        URL / MCP / INLINE references and LOCAL refs without a usable
        path return an empty list (caller treats the entry as
        self-contained). LOCAL refs whose path resolves to a missing
        file also return empty — consistent with
        :func:`pack._plan_local_payloads`'s behaviour today.

        The archive-relative path is left unset here; the actual
        slot allocation happens in the packer (which dedupes across
        entries by canonical source path). The handler just produces
        the ``(source, basename)`` candidate. The packer keeps using
        its own slot logic in Phase 8 until pack itself is rewired
        through the registry.
        """
        source = next(
            (s for s in self._plugin._sources if s.id == entry.id), None,
        )
        if source is None or source.type != SourceType.LOCAL:
            return []
        raw_path = source.resolved_path or source.path
        if not raw_path:
            return []
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            return []
        try:
            canonical = candidate.resolve(strict=True)
        except FileNotFoundError:
            return []
        # Archive-relative path placeholder: ``<basename>``. The
        # packer rewrites this once it allocates a slot.
        return [(canonical, canonical.name)]

    def rewrite_for_pack(
        self,
        entry: BundleEntry,
        payload_map: Dict[Path, str],
    ) -> None:
        """Rewrite the LOCAL ref's ``path`` to its archive-relative
        location and drop ``resolved_path``.

        ``payload_map`` is keyed by canonical source path (matches
        what :meth:`collect_pack_payloads` returned); the value is
        the final archive-relative path the packer chose.
        Non-LOCAL refs and entries without a payload entry leave the
        on-disk JSON untouched.
        """
        try:
            data = json.loads(entry.file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if not isinstance(data, dict):
            return
        if data.get("type") != SourceType.LOCAL.value:
            return
        raw_path = data.get("path") or data.get("resolved_path")
        if not raw_path:
            return
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            return
        try:
            canonical = candidate.resolve(strict=True)
        except FileNotFoundError:
            return
        archive_rel = payload_map.get(canonical)
        if archive_rel is None:
            return
        data["path"] = archive_rel
        data.pop("resolved_path", None)
        entry.file_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_entry(
        self,
        entry_id: str,
        bundle_name: str,
        file_path: Path,
    ) -> BundleEntry:
        """Wrap the lookup of ``bundle_tier`` for an entry.

        Bundled entries inherit their tier from the owning
        :class:`Bundle`. Free entries (``bundle_name == ""``) live at
        the workspace tier today — the catalog only loads free refs
        from the workspace ``.jaato/references/`` directory.
        """
        tier = BUNDLE_TIER_WORKSPACE
        if bundle_name:
            for b in self._plugin._bundles:
                if b.name == bundle_name:
                    tier = b.tier
                    break
        return BundleEntry(
            id=entry_id,
            kind=_REFERENCES_KIND,
            file_path=file_path,
            bundle_name=bundle_name,
            bundle_tier=tier,
        )
