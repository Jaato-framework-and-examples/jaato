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

import dataclasses
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
    BUNDLE_MANIFEST_FILENAME,
    BUNDLE_TIER_USER,
    BUNDLE_TIER_WORKSPACE,
    REFERENCE_NON_SOURCE_FILENAMES,
    ROOT_BUNDLE_NAME,
    ReferenceBundle,
    discover_bundles,
    resolve_bundle_roots,
)
from .config_loader import (
    ReferencesConfig,
    config_path_candidates,
    discover_reference_files,
    is_reference_file,
    load_config,
    parse_reference_file,
    reference_project_root,
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


@dataclass(frozen=True)
class SourceUnit:
    """The references one directory contributed to a load.

    Attributes:
        directory: The directory, normalized (``os.path.normpath`` of its
            absolute form), the key a changed path is placed by.
        bundle_name: ``bundle_name`` its references carry (``""`` for the
            root directory ``load_config`` auto-discovers).
        project_root: The root its LOCAL paths are made relative to
            (:func:`reference_project_root`), so a re-parse matches the load.
        resolve: Whether the load resolves its paths with
            :func:`resolve_for_context` against the key's project root (the
            root directory does, bundles do not).
        files: ``(path, source)`` per file that parsed, in file-name
            order; each path normalized like ``directory``.
    """

    directory: str
    bundle_name: str
    project_root: str
    resolve: bool
    files: Tuple[Tuple[str, ReferenceSource], ...]


def _norm(path: str) -> str:
    return os.path.normpath(os.path.abspath(path))


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
        units: The directories the references came from, root first, then
            each bundle in discovery order (:class:`SourceUnit`); the
            store derives the next generation from them.
        layout: Per tier root (normalized), the sub-bundle directories on
            disk when it was read, shadowed ones included; a change to it
            is a full load.
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
        units: Sequence[SourceUnit] = (),
        layout: Optional[Dict[str, frozenset]] = None,
        watch: Dict[str, Any],
    ) -> None:
        self.key = key
        self.generation = generation
        self.config = config
        self.sources: Tuple[ReferenceSource, ...] = tuple(sources)
        self.bundles: Tuple[ReferenceBundle, ...] = tuple(bundles)
        self.roots: Tuple[str, ...] = tuple(roots)
        self.units: Tuple[SourceUnit, ...] = tuple(units)
        self.layout: Dict[str, frozenset] = dict(layout or {})
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

        Loads when no holder keeps a snapshot alive. With ``refresh``,
        when a path the current one read has moved, the next generation is
        derived from it (only what moved is read again). Callers noticing
        the same change at once share one load (the lock).
        """
        with self._lock:
            snap = self._current() if self._current else None
            if snap is None:
                return self._load(None)
            if refresh and snap.is_stale():
                return self._derive(snap)
            return snap

    def reload(self) -> CatalogSnapshot:
        """Load again whatever the stamps say (``references reload``, a write)."""
        with self._lock:
            snap = self._current() if self._current else None
            return self._load(snap)

    def _load(self, previous: Optional[CatalogSnapshot]) -> CatalogSnapshot:
        """Read the whole catalog from disk."""
        pre = catalog_watch.stat_paths(_watch_paths(self.key, previous))
        config = _load_config(self.key)
        project_root = self.key.project_root
        # Freshly parsed, held by nothing else yet: resolved in place, so
        # ``config.sources`` and the root unit hold the same objects.
        for source in config.sources:
            resolved = resolve_for_context(source, project_root)
            if resolved is not None:
                source.resolved_path = resolved
        roots = _bundle_roots(self.key, config)
        layout = {_norm(str(path)): _sub_bundles(_norm(str(path))) for path, _tier in roots}
        bundles = discover_bundles(roots)
        units = _root_units(self.key, config) + [
            _bundle_unit(bundle, project_root) for bundle in _loaded_bundles(bundles)
        ]
        return self._publish(config, _explicit_sources(config), units, bundles,
                             [str(path) for path, _tier in roots], layout, pre)

    def _derive(self, previous: CatalogSnapshot) -> CatalogSnapshot:
        """Generation G+1 from G: re-read what moved, share everything else.

        The paths whose stamp moved are placed by directory: a reference
        file is parsed again (or dropped when gone), a directory whose
        listing changed is listed again. Anything else that moved (a
        ``references.json``, a bundle manifest or embedding config, a
        sub-bundle added or removed, a path no unit holds) changes more than
        a file's own references, so it is a full load. Both paths assemble
        the snapshot in :meth:`_publish`, so a derived snapshot equals the
        one a full load of the same disk would build.
        """
        pre = catalog_watch.stat_paths(_watch_paths(self.key, previous))
        moved = {p for p in set(pre) | set(previous.watch)
                 if pre.get(p) != previous.watch.get(p)}
        units = _rederive_units(previous, moved)
        if units is None:
            return self._load(previous)
        root = units[0] if units and units[0].resolve else None
        explicit = _explicit_sources(previous.config)
        config = dataclasses.replace(
            previous.config,
            sources=_root_sources(explicit, root),
            discovered=list(root.files) if root else [],
        )
        return self._publish(config, explicit, units, previous.bundles,
                             previous.roots, previous.layout, pre)

    def _publish(
        self,
        config: ReferencesConfig,
        explicit: List[ReferenceSource],
        units: List[SourceUnit],
        bundles: Sequence[ReferenceBundle],
        roots: Sequence[str],
        layout: Dict[str, frozenset],
        pre: Dict[str, Any],
    ) -> CatalogSnapshot:
        """Assemble, stamp and hold one generation (the one assembly)."""
        has_root = bool(units) and units[0].resolve
        root = units[0] if has_root else None
        sources = _root_sources(explicit, root)
        existing: Set[str] = {s.id for s in sources}
        for unit in units[1:] if has_root else units:
            for _path, source in unit.files:
                if source.id in existing:
                    logger.debug(
                        "references store: skipping duplicate id '%s' from bundle '%s'",
                        source.id, unit.bundle_name,
                    )
                    continue
                sources.append(source)
                existing.add(source.id)
        self._generation += 1
        snap = CatalogSnapshot(
            key=self.key,
            generation=self._generation,
            config=config,
            sources=sources,
            bundles=bundles,
            roots=roots,
            units=units,
            layout=layout,
            watch={},
        )
        post = catalog_watch.stat_paths(_watch_paths(self.key, snap))
        snap.watch = catalog_watch.settle(pre, post)
        self._current = weakref.ref(snap)
        logger.debug(
            "references store: generation %d (%d sources, %d bundles) for %s",
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


def _loaded_bundles(bundles: Iterable[ReferenceBundle]) -> List[ReferenceBundle]:
    """The bundles whose references a load reads, in discovery order.

    The workspace tier's root bundle is skipped: ``load_config`` already
    read it.
    """
    return [b for b in bundles
            if not (b.name == ROOT_BUNDLE_NAME and b.tier == BUNDLE_TIER_WORKSPACE)]


def _root_dir(key: CatalogKey, config: ReferencesConfig) -> Optional[Path]:
    """The directory ``load_config`` auto-discovered from, or ``None``."""
    if not config.auto_discover_references or key.workspace is None:
        return None
    refs = Path(config.references_dir)
    return refs if refs.is_absolute() else Path(key.workspace) / refs


def _root_units(key: CatalogKey, config: ReferencesConfig) -> List[SourceUnit]:
    """The root directory as a unit, from what ``load_config`` discovered."""
    directory = _root_dir(key, config)
    if directory is None:
        return []
    return [SourceUnit(
        directory=_norm(str(directory)),
        bundle_name=ROOT_BUNDLE_NAME,
        project_root=reference_project_root(directory, None),
        resolve=True,
        files=tuple((_norm(path), source) for path, source in config.discovered),
    )]


def _bundle_unit(bundle: ReferenceBundle, project_root: Optional[str]) -> SourceUnit:
    """One bundle's references, each tagged with the bundle's name."""
    files = discover_reference_files(
        str(bundle.directory),
        base_path=str(bundle.directory.parent),
        project_root=project_root,
    )
    for _path, source in files:
        source.bundle_name = bundle.name
    return SourceUnit(
        directory=_norm(str(bundle.directory)),
        bundle_name=bundle.name,
        project_root=reference_project_root(Path(bundle.directory), project_root),
        resolve=False,
        files=tuple((_norm(path), source) for path, source in files),
    )


def _explicit_sources(config: ReferencesConfig) -> List[ReferenceSource]:
    """The root references that came from ``references.json``, in order."""
    discovered = {id(source) for _path, source in config.discovered}
    return [s for s in config.sources if id(s) not in discovered]


def _root_sources(explicit: List[ReferenceSource],
                  root: Optional[SourceUnit]) -> List[ReferenceSource]:
    """``config.sources`` as ``load_config`` builds it.

    The explicit sources, then every discovered one whose id no explicit
    source has (a duplicate among the discovered stays, as in
    ``load_config``; the snapshot's ``by_id`` keeps the first).
    """
    sources = list(explicit)
    if root is not None:
        explicit_ids = {s.id for s in explicit}
        sources.extend(s for _path, s in root.files if s.id not in explicit_ids)
    return sources


def _rederive_units(previous: CatalogSnapshot,
                    moved: Set[str]) -> Optional[List[SourceUnit]]:
    """The units with ``moved`` applied, or ``None`` when a full load is needed.

    A moved reference file is parsed again with its unit's project root
    (and resolved when the unit is the root directory); a moved unit
    directory is listed again for files added or removed. A moved tier
    root is fine while its sub-bundles are the ones it had. Anything else
    returns ``None``.
    """
    by_dir = {unit.directory: unit for unit in previous.units}
    relist: Set[str] = set()
    reparse: Dict[str, Set[str]] = {}
    for path in moved:
        normal = _norm(path)
        if normal in previous.layout:
            if _sub_bundles(normal) != previous.layout[normal]:
                return None
            if normal in by_dir:
                relist.add(normal)
            continue
        if normal in by_dir:
            relist.add(normal)
            continue
        parent = os.path.dirname(normal)
        if parent in by_dir and _is_reference_name(os.path.basename(normal)):
            reparse.setdefault(parent, set()).add(normal)
            continue
        return None
    return [
        _reread_unit(unit, unit.directory in relist, reparse.get(unit.directory, set()),
                     previous.key.project_root)
        if unit.directory in relist or unit.directory in reparse else unit
        for unit in previous.units
    ]


def _is_reference_name(name: str) -> bool:
    """Whether a file named ``name`` would be read as a reference definition."""
    return name.endswith(".json") and name not in REFERENCE_NON_SOURCE_FILENAMES


def _sub_bundles(root: str) -> frozenset:
    """The directories directly under a tier root that hold a ``bundle.json``."""
    found: Set[str] = set()
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if (entry.is_dir(follow_symlinks=True)
                        and os.path.isfile(os.path.join(entry.path, BUNDLE_MANIFEST_FILENAME))):
                    found.add(_norm(entry.path))
    except OSError:
        pass
    return frozenset(found)


def _reread_unit(unit: SourceUnit, relist: bool, changed: Set[str],
                 resolve_root: Optional[str]) -> SourceUnit:
    """``unit`` with its listing re-read and ``changed`` files parsed again.

    A root-directory reference is resolved against ``resolve_root`` (the
    key's project root), as the full load resolves it.

    Paths are normalized. A file that no longer parses (or is gone) drops
    out; the order is file-name order, as a full load lists them.
    """
    files: Dict[str, ReferenceSource] = dict(unit.files)
    if relist:
        present: Set[str] = set()
        try:
            with os.scandir(unit.directory) as entries:
                for entry in entries:
                    if is_reference_file(Path(entry.path)):
                        present.add(_norm(entry.path))
        except OSError:
            pass
        for gone in set(files) - present:
            del files[gone]
        changed = changed | (present - set(files))
    for path in changed:
        files.pop(path, None)
        file_path = Path(path)
        if not is_reference_file(file_path):
            continue
        source = parse_reference_file(file_path, unit.project_root)
        if source is None:
            continue
        if unit.resolve:
            resolved = resolve_for_context(source, resolve_root)
            if resolved is not None:
                source.resolved_path = resolved
        else:
            source.bundle_name = unit.bundle_name
        files[path] = source
    ordered = sorted(files.items(), key=lambda item: os.path.basename(item[0]))
    return dataclasses.replace(unit, files=tuple(ordered))


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
