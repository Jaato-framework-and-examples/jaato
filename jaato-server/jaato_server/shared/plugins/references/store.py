"""Where the references catalog comes from, and one immutable view of it.

Three parts (docs/design/references-storage-scaling.md, phase 1):

- :class:`ReferenceStore` is the source of references for one
  :class:`CatalogKey`: today the files under ``.jaato/references/``, the
  user tier and an optional ``references.json``.  ``current()`` answers
  with the snapshot that matches the files now.
- :class:`CatalogSnapshot` is the result of one load at one generation.
  It never changes after it is built, so the indexes built on it stay
  right for its lifetime, and every holder in a process can share it.
- :class:`WorkingSet` is what one session can see: the whole snapshot, or
  a narrowed list (inline ``sources``).  It lives in the plugin, per
  session, and is never shared.

Rules the module holds to:

- **The key is everything a load reads.**  Two holders get the same store
  only when a load would read the same files for both: the explicit
  ``config_path``, ``REFERENCES_CONFIG_PATH``, the workspace, the project
  root the paths resolve against, ``config_root``, and the user tier's
  home (a runner dropped to another uid has another one, #1168).
- **The store holds its current snapshot weakly.**  A snapshot is freed
  when its last holder lets go, so there is no eviction policy: in the
  daemon that is when the last session on the catalog unloads, in a
  runner when the plugin is reset or shut down.
- **Shared objects are never edited.**  A ``ReferenceSource`` or
  ``ReferenceBundle`` in a snapshot may be held by several sessions; a
  session that needs a different value works on a copy.

The load itself is the one the references plugin always did (the root
through :func:`load_config`, each bundle through
:func:`discover_references`), moved here unchanged so a load is a function
of the key alone.
"""

from __future__ import annotations

import functools
import logging
import os
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from . import catalog_watch
from .bundle import (
    BUNDLE_TIER_USER,
    BUNDLE_TIER_WORKSPACE,
    ROOT_BUNDLE_NAME,
    ReferenceBundle,
    discover_bundles,
    resolve_bundle_roots,
)
from .config_loader import (
    ReferencesConfig,
    config_path_candidates,
    discover_references,
    load_config,
)
from .links import LinkIndex
from .models import ReferenceSource, SourceType

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class CatalogKey:
    """Everything one catalog load reads, so equal keys load equal catalogs.

    ``workspace`` is the directory ``load_config`` resolves the root
    catalog against; ``bundle_workspace`` the one the workspace bundle
    tier is found under (the plugin's ``workspace_path``, which may be
    unset while ``workspace`` comes from ``base_path``).
    ``config_candidates`` are the ``references.json`` files the load may
    read (:func:`config_path_candidates`: the explicit path, else
    ``REFERENCES_CONFIG_PATH``, else the workspace and user-tier files),
    so a change of that variable or of the user tier is a different key.
    """

    config_path: Optional[str]
    config_candidates: Tuple[str, ...]
    workspace: Optional[str]
    bundle_workspace: Optional[str]
    project_root: Optional[str]
    config_root: Optional[str]
    user_home: str

    @classmethod
    def for_context(
        cls,
        *,
        config_path: Optional[str],
        workspace: Optional[str],
        bundle_workspace: Optional[str],
        project_root: Optional[str],
        config_root: Optional[str],
    ) -> "CatalogKey":
        """The key for a load with these inputs, in this process, now."""
        return cls(
            config_path=config_path,
            config_candidates=tuple(config_path_candidates(config_path, workspace)),
            workspace=workspace,
            bundle_workspace=bundle_workspace,
            project_root=project_root,
            config_root=config_root,
            user_home=str(Path.home()),
        )


def resolve_for_context(source: ReferenceSource, project_root: Optional[str]) -> Optional[str]:
    """The path a LOCAL source resolves to from ``project_root``, or ``None``.

    ``None`` means "leave ``resolved_path`` as it is": not LOCAL, no path,
    already an absolute path that exists, or nothing found. Never edits
    ``source``; the caller decides whether to set the value on a fresh
    object (a load) or on a copy (a session).
    """
    if source.type != SourceType.LOCAL or not source.path:
        return None
    if source.resolved_path:
        resolved = Path(source.resolved_path)
        if resolved.is_absolute() and resolved.exists():
            return None
        if project_root:
            from_root = Path(project_root) / source.resolved_path
            if from_root.exists():
                return str(from_root.resolve())
    if project_root:
        original = Path(source.path)
        if not original.is_absolute():
            from_root = Path(project_root) / source.path
            if from_root.exists():
                return str(from_root.resolve())
    return None


class CatalogSnapshot:
    """One load of the catalog, immutable once built.

    Attributes:
        key: The :class:`CatalogKey` it was loaded for.
        generation: Increases by one per load of the same store.
        config: The parsed ``ReferencesConfig`` (channel settings, the
            root catalog in ``config.sources``).
        sources: Every reference, in load order: the root first, then each
            bundle's (an id already loaded keeps its first copy).
        bundles: The bundles discovery found. Shared: a session that
            attaches state (a matcher) works on its own copies.
        roots: The tier roots bundle discovery walked.
        watch: The settled stamps of every path this load read
            (:func:`catalog_watch.settle`); :meth:`is_stale` compares them.
    """

    def __init__(
        self,
        *,
        key: CatalogKey,
        generation: int,
        config: ReferencesConfig,
        sources: Sequence[ReferenceSource],
        bundles: Sequence[ReferenceBundle],
        roots: Sequence[str],
        watch: Dict[str, Any],
    ) -> None:
        self.key = key
        self.generation = generation
        self.config = config
        self.sources: Tuple[ReferenceSource, ...] = tuple(sources)
        self.bundles: Tuple[ReferenceBundle, ...] = tuple(bundles)
        self.roots: Tuple[str, ...] = tuple(roots)
        self.watch = dict(watch)
        self.by_id: Dict[str, ReferenceSource] = {}
        for source in self.sources:
            self.by_id.setdefault(source.id, source)

    def get(self, ref_id: str) -> Optional[ReferenceSource]:
        """The reference with ``ref_id``, or ``None``."""
        return self.by_id.get(ref_id)

    @functools.cached_property
    def links(self) -> LinkIndex:
        """The declared edges of this catalog, built on first use."""
        return LinkIndex(self.sources)

    @functools.cached_property
    def by_bundle(self) -> Dict[str, Tuple[ReferenceSource, ...]]:
        """References per ``bundle_name`` (``""`` is the root), in load order."""
        out: Dict[str, List[ReferenceSource]] = {}
        for source in self.sources:
            out.setdefault(source.bundle_name, []).append(source)
        return {name: tuple(items) for name, items in out.items()}

    def is_stale(self) -> bool:
        """Whether a path this load read has moved since (one ``stat`` each)."""
        return catalog_watch.has_changed(self.watch)


class ReferenceStore:
    """The catalog for one :class:`CatalogKey`, in one process.

    Obtained through :func:`store_for`, so every holder of a key in a
    process shares one store and, through it, one snapshot.
    """

    def __init__(self, key: CatalogKey) -> None:
        self.key = key
        self._lock = threading.Lock()
        self._current: Optional["weakref.ref[CatalogSnapshot]"] = None
        self._generation = 0

    def current(self, *, refresh: bool = True) -> CatalogSnapshot:
        """The snapshot to read from now.

        Loads when no holder keeps a snapshot alive, and, with
        ``refresh``, when a path the current one read has moved. Callers
        noticing the same change at once share one load (the lock).
        """
        with self._lock:
            snap = self._current() if self._current else None
            if snap is None or (refresh and snap.is_stale()):
                snap = self._load(snap)
            return snap

    def reload(self) -> CatalogSnapshot:
        """Load again whatever the stamps say (``references reload``, a write)."""
        with self._lock:
            snap = self._current() if self._current else None
            return self._load(snap)

    def _load(self, previous: Optional[CatalogSnapshot]) -> CatalogSnapshot:
        pre = catalog_watch.stat_paths(_watch_paths(self.key, previous))
        config = _load_config(self.key)
        project_root = self.key.project_root
        # Freshly parsed, held by nothing else yet: resolved in place, so
        # ``config.sources`` and ``sources`` hold the same objects.
        for source in config.sources:
            resolved = resolve_for_context(source, project_root)
            if resolved is not None:
                source.resolved_path = resolved
        sources = list(config.sources)
        roots = _bundle_roots(self.key, config)
        bundles = discover_bundles(roots)
        _append_bundle_sources(sources, bundles, project_root)
        self._generation += 1
        snap = CatalogSnapshot(
            key=self.key,
            generation=self._generation,
            config=config,
            sources=sources,
            bundles=bundles,
            roots=[str(path) for path, _tier in roots],
            watch={},
        )
        post = catalog_watch.stat_paths(_watch_paths(self.key, snap))
        snap.watch = catalog_watch.settle(pre, post)
        self._current = weakref.ref(snap)
        logger.debug(
            "references store: loaded generation %d (%d sources, %d bundles) for %s",
            snap.generation, len(snap.sources), len(snap.bundles), self.key.workspace,
        )
        return snap


_STORES: "weakref.WeakValueDictionary[CatalogKey, ReferenceStore]" = weakref.WeakValueDictionary()
_STORES_LOCK = threading.Lock()


def store_for(key: CatalogKey) -> ReferenceStore:
    """The one store for ``key`` in this process, created on first use.

    Held weakly here too: a store nobody holds goes with its last holder.
    """
    with _STORES_LOCK:
        store = _STORES.get(key)
        if store is None:
            store = ReferenceStore(key)
            _STORES[key] = store
        return store


# ---------------------------------------------------------------- the load


def _load_config(key: CatalogKey) -> ReferencesConfig:
    try:
        return load_config(key.config_path, workspace_path=key.workspace)
    except FileNotFoundError:
        return ReferencesConfig()


def _workspace_refs_dir(key: CatalogKey, config: ReferencesConfig) -> Optional[Path]:
    """The workspace tier root: ``references_dir`` from the config, placed.

    ``config_root`` replaces the workspace's ``.jaato`` when set, so
    ``.jaato/references`` becomes ``<config_root>/references``.
    """
    cfg_refs = Path(config.references_dir)
    if cfg_refs.is_absolute():
        return cfg_refs
    if key.config_root:
        root = Path(key.config_root).expanduser().resolve()
        parts = cfg_refs.parts
        if parts and parts[0] == ".jaato":
            inner = Path(*parts[1:]) if len(parts) > 1 else Path()
        else:
            inner = cfg_refs
        return root / inner
    if key.bundle_workspace:
        return Path(key.bundle_workspace) / cfg_refs
    if key.project_root:
        return Path(key.project_root) / cfg_refs
    return None


def _bundle_roots(key: CatalogKey, config: ReferencesConfig) -> List[Tuple[Path, str]]:
    roots: List[Tuple[Path, str]] = []
    workspace_dir = _workspace_refs_dir(key, config)
    if workspace_dir is not None:
        roots.append((workspace_dir.resolve(), BUNDLE_TIER_WORKSPACE))
    for path, tier in resolve_bundle_roots(workspace_path=None):
        if tier == BUNDLE_TIER_USER:
            roots.append((path, tier))
            break
    return roots


def _append_bundle_sources(
    sources: List[ReferenceSource],
    bundles: Iterable[ReferenceBundle],
    project_root: Optional[str],
) -> None:
    """Load every bundle's references after the root's, first id wins.

    The workspace tier's root bundle is skipped: ``load_config`` already
    read it.
    """
    existing: Set[str] = {s.id for s in sources}
    for bundle in bundles:
        if bundle.name == ROOT_BUNDLE_NAME and bundle.tier == BUNDLE_TIER_WORKSPACE:
            continue
        for source in discover_references(
            str(bundle.directory),
            base_path=str(bundle.directory.parent),
            project_root=project_root,
        ):
            if source.id in existing:
                logger.debug(
                    "references store: skipping duplicate id '%s' from bundle '%s'",
                    source.id, bundle.qualified_ref,
                )
                continue
            source.bundle_name = bundle.name
            sources.append(source)
            existing.add(source.id)


def _watch_paths(key: CatalogKey, snap: Optional[CatalogSnapshot]) -> List[str]:
    """Every path whose change means a load from ``key`` may be stale.

    The tier roots and every bundle directory the load walked (a new
    sub-bundle is a new directory in a root), the directory
    ``load_config`` auto-discovers from, every ``*.json`` directly in those
    directories (an in-place edit moves no directory mtime, #1437), and
    each ``references.json`` the loader could pick. Before the first load
    there are no roots or bundles yet, so only the last two.
    """
    workspace = key.bundle_workspace or key.workspace or key.project_root
    paths: Set[str] = set()
    refs_dir = Path(".jaato/references")
    if snap is not None:
        paths.update(snap.roots)
        paths.update(str(b.directory) for b in snap.bundles)
        refs_dir = Path(snap.config.references_dir)
    if not refs_dir.is_absolute() and workspace:
        refs_dir = Path(workspace) / refs_dir
    if refs_dir.is_absolute():
        paths.add(str(refs_dir))
    paths.update(catalog_watch.json_files([p for p in paths if os.path.isdir(p)]))
    paths.update(config_path_candidates(key.config_path, workspace))
    return sorted(paths)


# ---------------------------------------------------------------- per session


class WorkingSet:
    """The references one session can see, over one snapshot.

    Either the whole snapshot (the usual case: no copy is made), or a
    narrowed list a session built itself (inline ``sources``, plus what a
    preselection or a transitive walk pulled in). Never shared between
    sessions.
    """

    def __init__(self, snapshot: Optional[CatalogSnapshot],
                 narrowed: Optional[List[ReferenceSource]] = None) -> None:
        if snapshot is None and narrowed is None:
            raise ValueError("a whole working set needs a snapshot")
        self.snapshot = snapshot
        self._own: Optional[List[ReferenceSource]] = None
        self._own_by_id: Dict[str, ReferenceSource] = {}
        if narrowed is not None:
            self._own = []
            for source in narrowed:
                self.add(source)

    @property
    def is_whole(self) -> bool:
        """True when the session sees the whole snapshot."""
        return self._own is None

    @property
    def sources(self) -> Sequence[ReferenceSource]:
        """In order: the snapshot's, or the narrowed list."""
        return self.snapshot.sources if self._own is None else self._own

    def get(self, ref_id: str) -> Optional[ReferenceSource]:
        """The visible reference with ``ref_id``, or ``None``."""
        if self._own is None:
            return self.snapshot.by_id.get(ref_id)
        return self._own_by_id.get(ref_id)

    def __contains__(self, ref_id: object) -> bool:
        return isinstance(ref_id, str) and self.get(ref_id) is not None

    def __iter__(self) -> Iterator[ReferenceSource]:
        return iter(self.sources)

    def __len__(self) -> int:
        return len(self.sources)

    def ids(self) -> Set[str]:
        """Every visible id."""
        if self._own is None:
            return set(self.snapshot.by_id)
        return set(self._own_by_id)

    def in_bundle(self, name: str) -> List[ReferenceSource]:
        """The visible references whose ``bundle_name`` is ``name``."""
        if self._own is None:
            return list(self.snapshot.by_bundle.get(name, ()))
        return [s for s in self._own if s.bundle_name == name]

    def add(self, source: ReferenceSource) -> None:
        """Make ``source`` visible; an id already visible keeps its first copy.

        On a whole working set every catalog id is already visible, so a
        catalog reference is a no-op; anything else is an error, because a
        whole working set must stay equal to its snapshot.
        """
        if self._own is None:
            if source.id not in self.snapshot.by_id:
                raise ValueError(
                    f"reference '{source.id}' is not in the catalog snapshot; "
                    "only a narrowed working set holds references of its own"
                )
            return
        if source.id in self._own_by_id:
            return
        self._own.append(source)
        self._own_by_id[source.id] = source
