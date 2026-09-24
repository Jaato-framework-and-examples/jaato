"""References-specific bundle helpers.

The domain-agnostic parts of bundle management — :class:`Bundle` (name,
directory, tier), the tier constants, ``BundleRef`` /
``parse_bundle_ref`` / ``find_bundle``, root resolution and directory
scanning — live in :mod:`shared.plugins.bundle_common.bundle` so every
domain (agents, tasks, profiles, services) can reuse the same
machinery. This module re-exports that surface for the references
plugin's existing import sites and owns everything that is *about
references*:

* :class:`ReferenceBundle` — a :class:`Bundle` plus this domain's
  vector index (``embedding_model``, ``embedding_dimensions``,
  ``embedding_sidecar``, ``embedding_rows``, ``reconcile_mode``) and
  per-bundle runtime state (``matcher``, ``owned_source_ids``). These
  fields used to sit on the generic dataclass, where they made a
  vectorless bundle unrepresentable and pinned an ``Any``-typed live
  matcher instance onto a domain-agnostic on-disk abstraction.
* :data:`EMBEDDING_CONFIG_FILENAME` and
  :func:`load_reference_bundle` / :func:`write_manifest` — the
  ``embedding_config.json`` reader and writer. That file is a
  references file, name and body alike: the generic layer neither
  reads it nor recognises its name, and it marks nothing (#1130). The
  names this domain keeps beside its definitions are declared once in
  :data:`REFERENCE_NON_SOURCE_FILENAMES` and handed to the generic
  layer through
  :meth:`~shared.plugins.bundle_common.handler.BundleEntryHandler.non_entry_filenames`.
* :func:`discover_bundles` — the generic scan, with each discovered
  directory upgraded to a :class:`ReferenceBundle`.
* :func:`metadata_hash` — fingerprint stored in
  ``ReferenceSource.embedding.source_hash``.
* :class:`DriftReport` + :func:`detect_drift` — compare a bundle
  manifest against the live reference catalog.

**A bundle without an index is a bundle.** ``embedding_model`` and its
siblings default to empty, :attr:`ReferenceBundle.has_index` reports
whether the bundle declares a vector index at all, and the paths that
only make sense with one (:attr:`sidecar_path`, :attr:`lock_path`) are
``None`` without it. Reconcile, merge and matcher attachment each
decline such a bundle by name rather than computing a path from an
empty filename.

The module also wraps
:func:`bundle_common.bundle.resolve_bundle_roots` to bake in the
references-domain subpath (``.jaato/references``) so existing call
sites (and tests) don't have to know about the ``domain_subpath``
argument.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

# Re-export the domain-agnostic surface so existing imports
# (``from shared.plugins.references.bundle import …``) keep working.
from ..bundle_common.bundle import (  # noqa: F401
    BUNDLE_MANIFEST_FILENAME,
    BUNDLE_TIER_USER,
    BUNDLE_TIER_WORKSPACE,
    ROOT_BUNDLE_NAME,
    VALID_BUNDLE_TIERS,
    AmbiguousBundleRefError,
    Bundle,
    BundleRef,
    find_bundle,
    is_bundle_directory,
    parse_bundle_ref,
    write_bundle_manifest,
)
from ..bundle_common.bundle import (
    discover_bundles as _discover_bundles_generic,
)
from ..bundle_common.bundle import (
    resolve_bundle_roots as _resolve_bundle_roots_generic,
)

from .models import ReferenceSource

logger = logging.getLogger(__name__)


# The references plugin's vector-index file. Owned here: this module is
# the only thing in the tree that reads or writes its body, and the
# generic layer knows nothing about it -- it marks NOTHING. A references
# directory is a bundle because it carries ``bundle.json``; whether it
# also carries a vector index is a separate, optional fact about its
# contents. That separation is the whole of #1130.
EMBEDDING_CONFIG_FILENAME = "embedding_config.json"

# Files inside a references bundle that are metadata, not reference
# definitions: the generic manifest that marks the bundle, and this
# plugin's own vector-index descriptor.  #1130 keeps these two SEPARATE
# files on purpose -- a bundle and an index are independent things, and
# one file carrying both is what this issue exists to undo.
REFERENCE_NON_SOURCE_FILENAMES: Tuple[str, ...] = (
    BUNDLE_MANIFEST_FILENAME,
    EMBEDDING_CONFIG_FILENAME,
)

# Valid reconcile modes declared in a bundle's embedding config.
_VALID_RECONCILE_MODES: Set[str] = {"eager", "lazy", "off"}

# Default reconcile mode when the config declares none.
DEFAULT_RECONCILE_MODE = "eager"


@dataclass
class ReferenceBundle(Bundle):
    """A references bundle — a :class:`Bundle` plus its vector index.

    Each bundle owns at most one sidecar matrix and one embedding
    config. The bundle's entries come from JSON files in its own
    directory and nowhere else; cross-bundle overlap is handled at the
    plugin level by namespacing.

    Every index field defaults to "absent", so a directory that
    declares a bundle and no vector index loads cleanly and reports
    :attr:`has_index` ``False``. That is the normal state for a
    references set that is distributed as definitions only.

    Attributes:
        embedding_model: sentence-transformers model used to produce
            the sidecar vectors. Empty when the bundle has no index.
        embedding_dimensions: Vector dimensionality. Must equal
            ``matrix.shape[1]`` when the sidecar is loaded. ``0`` when
            the bundle has no index.
        embedding_sidecar: Filename of the ``.npy`` file, relative to
            ``directory``. Empty when the bundle has no index.
        embedding_rows: Ordered list of entry ids — ``rows[i]`` is the
            id whose vector lives at matrix row ``i``. Authoritative
            mapping from row to id.
        reconcile_mode: ``"eager"`` (reconcile during ``initialize``),
            ``"lazy"`` (reconcile before the first semantic query), or
            ``"off"`` (only reconcile when the operator runs reconcile
            manually).
        owned_source_ids: Cached set of ids the bundle claims in its
            ``rows`` list. Populated on load; the live catalog is the
            source of truth for which ids actually exist.
        matcher: Attached semantic matcher instance; ``None`` when the
            bundle has no compatible matcher (no index, model mismatch,
            missing provider, empty rows, load failure). Not serialized.
    """

    embedding_model: str = ""
    embedding_dimensions: int = 0
    embedding_sidecar: str = ""
    embedding_rows: List[str] = field(default_factory=list)
    reconcile_mode: str = DEFAULT_RECONCILE_MODE
    owned_source_ids: Set[str] = field(default_factory=set)
    matcher: Optional[Any] = None

    @property
    def has_index(self) -> bool:
        """Whether this bundle declares a usable vector index.

        All three of model, dimensions and sidecar are needed before
        anything can be read from or written to the sidecar, so the
        three are reported as one fact rather than checked separately
        at every call site.
        """
        return bool(
            self.embedding_model
            and self.embedding_dimensions
            and self.embedding_sidecar
        )

    @property
    def embedding_config_path(self) -> Path:
        """Absolute path to this bundle's ``embedding_config.json``.

        Where the file belongs, whether or not it exists — an
        index-less bundle has a path here and no file at it.
        """
        return self.directory / EMBEDDING_CONFIG_FILENAME

    @property
    def sidecar_path(self) -> Optional[Path]:
        """Absolute path to this bundle's ``.npy`` sidecar matrix.

        ``None`` when the bundle declares no index — deriving a path
        from an empty filename would yield the bundle DIRECTORY, which
        every caller would then read or overwrite.
        """
        if not self.embedding_sidecar:
            return None
        return self.directory / self.embedding_sidecar

    @property
    def lock_path(self) -> Optional[Path]:
        """Advisory-lock filename used by the reconcile writer.

        A sibling of the sidecar so concurrent daemons targeting the
        same workspace serialize their rewrites. ``None`` when the
        bundle declares no index, for the reason given on
        :attr:`sidecar_path`.
        """
        if not self.embedding_sidecar:
            return None
        return self.directory / (self.embedding_sidecar + ".lock")


def _read_embedding_config(
    directory: Path, *, name: str,
) -> Dict[str, Any]:
    """Return the index fields declared by ``embedding_config.json``.

    Returns an empty dict when the file is absent, unreadable, or does
    not declare a complete index. A malformed file is logged and read
    as "no index": the directory is still a bundle (its manifest is
    what says so), it just has no sidecar anything can use. Refusing
    the whole directory is what used to make a references set with a
    broken config vanish from the catalog without a word.

    Args:
        directory: Bundle directory.
        name: Bundle name, for log messages.

    Returns:
        Keyword arguments for :class:`ReferenceBundle`'s index fields.
    """
    path = directory / EMBEDDING_CONFIG_FILENAME
    if not path.is_file():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(
            "Bundle '%s': failed to read %s: %s; treating as no vector index",
            name or "(root)", path, e,
        )
        return {}

    if not isinstance(raw, dict):
        logger.warning(
            "Bundle '%s': %s must be a JSON object; treating as no vector "
            "index", name or "(root)", path,
        )
        return {}

    model = raw.get("embedding_model")
    dims = raw.get("embedding_dimensions")
    sidecar = raw.get("embedding_sidecar")
    rows = raw.get("rows", [])
    reconcile_mode = raw.get("reconcile", DEFAULT_RECONCILE_MODE)

    if not model or not dims or not sidecar:
        logger.warning(
            "Bundle '%s': %s declares no complete vector index "
            "(embedding_model, embedding_dimensions, embedding_sidecar); "
            "loading the bundle without one",
            name or "(root)", path,
        )
        return {}

    rows = _valid_rows(rows, name=name, path=path)
    reconcile_mode = _valid_reconcile_mode(reconcile_mode, name=name, path=path)

    return {
        "embedding_model": str(model),
        "embedding_dimensions": int(dims),
        "embedding_sidecar": str(sidecar),
        "embedding_rows": list(rows),
        "reconcile_mode": reconcile_mode,
        "owned_source_ids": set(rows),
    }


def _valid_rows(raw: Any, *, name: str, path: Path) -> List[str]:
    """Return the declared row ids, or ``[]`` when they are unusable.

    A ``rows`` list that is not a list of strings is a corrupt index,
    not a reason to lose the bundle: reading it as empty makes every id
    look missing, which is exactly what reconcile is for.
    """
    if isinstance(raw, list) and all(isinstance(r, str) for r in raw):
        return list(raw)
    logger.warning(
        "Bundle '%s': 'rows' must be a list of entry ids: %s; reading it "
        "as empty", name or "(root)", path,
    )
    return []


def _valid_reconcile_mode(raw: Any, *, name: str, path: Path) -> str:
    """Return the declared reconcile mode, or the default when unknown."""
    if raw in _VALID_RECONCILE_MODES:
        return str(raw)
    logger.warning(
        "Bundle '%s': unknown reconcile mode %r in %s, falling back to %r",
        name or "(root)", raw, path, DEFAULT_RECONCILE_MODE,
    )
    return DEFAULT_RECONCILE_MODE


def require_index_paths(bundle: ReferenceBundle) -> Tuple[Path, Path]:
    """Return ``(sidecar_path, lock_path)`` for an indexed bundle.

    The two paths are ``None`` on a bundle that declares no vector
    index, so every writer would otherwise repeat the same assertion.
    Callers gate on :attr:`ReferenceBundle.has_index` first, which makes
    this a contract check rather than a branch they are expected to
    take.

    Raises:
        ValueError: ``bundle`` declares no vector index.
    """
    sidecar, lock = bundle.sidecar_path, bundle.lock_path
    if sidecar is None or lock is None:
        raise ValueError(
            f"bundle '{bundle.qualified_ref}' declares no vector index"
        )
    return sidecar, lock


def load_reference_bundle(
    directory: Path,
    *,
    name: str,
    tier: str = BUNDLE_TIER_WORKSPACE,
) -> Optional[ReferenceBundle]:
    """Load ``directory`` as a :class:`ReferenceBundle`, or ``None``.

    ``None`` means "not a bundle" — no manifest of any kind. A bundle
    whose ``embedding_config.json`` is absent or unusable still loads,
    with :attr:`ReferenceBundle.has_index` ``False``.

    Args:
        directory: Candidate bundle directory.
        name: Bundle name (``""`` for root, subdir name otherwise).
        tier: Tier root this bundle was found under.

    Returns:
        The loaded :class:`ReferenceBundle`, or ``None``.
    """
    from ..bundle_common.bundle import load_bundle as _load_bundle

    generic = _load_bundle(directory, name=name, tier=tier)
    if generic is None:
        return None
    return _upgrade(generic)


def _upgrade(generic: Bundle) -> ReferenceBundle:
    """Attach this domain's index fields to a generic :class:`Bundle`."""
    return ReferenceBundle(
        name=generic.name,
        directory=generic.directory,
        tier=generic.tier,
        **_read_embedding_config(generic.directory, name=generic.name),
    )


def discover_bundles(
    roots: Union[Path, Sequence[Tuple[Path, str]]],
) -> List[ReferenceBundle]:
    """Scan tier roots for references bundles.

    Thin wrapper over
    :func:`shared.plugins.bundle_common.bundle.discover_bundles` — the
    scan, the ordering and the workspace-shadows-user rule are the
    generic ones — that upgrades each result to a
    :class:`ReferenceBundle` by reading its ``embedding_config.json``.

    Args:
        roots: A single ``Path`` (treated as the workspace tier) or an
            ordered sequence of ``(root_dir, tier)`` pairs.

    Returns:
        List of :class:`ReferenceBundle` in discovery order.
    """
    return [_upgrade(b) for b in _discover_bundles_generic(roots)]


def write_manifest(bundle: ReferenceBundle, *, rows: List[str]) -> None:
    """Write the bundle's ``embedding_config.json`` with a new ``rows``.

    Uses an atomic write (``.tmp`` + rename) so a crash mid-write
    cannot corrupt the file.

    Args:
        bundle: Target bundle. Must declare an index — there is no
            sidecar to describe otherwise.
        rows: New ordered list of entry ids. ``len(rows)`` must equal
            the new sidecar matrix's row count.

    Raises:
        ValueError: ``bundle`` declares no vector index.
    """
    if not bundle.has_index:
        raise ValueError(
            f"bundle '{bundle.qualified_ref}' declares no vector index; "
            f"nothing to write to {EMBEDDING_CONFIG_FILENAME}"
        )

    payload: Dict[str, Any] = {
        "embedding_model": bundle.embedding_model,
        "embedding_dimensions": bundle.embedding_dimensions,
        "embedding_sidecar": bundle.embedding_sidecar,
        "rows": rows,
    }
    if bundle.reconcile_mode != DEFAULT_RECONCILE_MODE:
        payload["reconcile"] = bundle.reconcile_mode

    target = bundle.embedding_config_path
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(target)
    bundle.embedding_rows = list(rows)
    bundle.owned_source_ids = set(rows)


# References-domain subpath under each tier root. The references
# plugin's bundles live at ``<workspace>/.jaato/references/`` and
# ``~/.jaato/references/``. Other domains pass their own subpath.
_REFERENCES_SUBPATH = Path(".jaato") / "references"


def resolve_bundle_roots(
    workspace_path: Optional[Union[str, Path]],
    *,
    user_home: Optional[Path] = None,
) -> List[Tuple[Path, str]]:
    """References-domain wrapper over the generic resolver.

    Bakes in :data:`_REFERENCES_SUBPATH` so callers and tests don't
    have to repeat ``.jaato/references`` at every call site. Equivalent
    to::

        bundle_common.bundle.resolve_bundle_roots(
            workspace_path,
            domain_subpath=Path(".jaato/references"),
            user_home=user_home,
        )

    Args:
        workspace_path: Workspace root, or ``None`` if unknown.
        user_home: Override for ``Path.home()`` (test seam).

    Returns:
        Ordered list of ``(absolute_root_dir, tier_name)`` pairs.
    """
    return _resolve_bundle_roots_generic(
        workspace_path,
        domain_subpath=_REFERENCES_SUBPATH,
        user_home=user_home,
    )


def metadata_hash(source: ReferenceSource) -> str:
    """Compute the canonical fingerprint stored in ``embedding.source_hash``.

    We hash *metadata*, not content: the embedding is produced from
    ``name + description + tags + fetchHint`` (the text the
    ``gen-references`` agent feeds to ``compute_embedding``), so the
    right staleness signal is "did any of that metadata drift?"
    Content-hashing would be wrong: a LOCAL reference's content can
    change independently without the vector needing to be regenerated,
    and URL/MCP references have no local content to hash at all.

    The format is ``sha256:<hex>``. Tags are sorted for stability
    across ordering-insensitive edits.

    Args:
        source: The reference source to fingerprint.

    Returns:
        A ``sha256:<hex>`` string.
    """
    text = "\n".join([
        source.name,
        source.description,
        ",".join(sorted(source.tags)),
        source.fetch_hint or "",
    ])
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


@dataclass
class DriftReport:
    """Per-bundle summary of what reconcile would do if run now.

    Attributes:
        missing: Source ids present in the catalog but not in
            ``rows``, or present without a stored ``source_hash``.
            Reconcile would embed these and append rows.
        stale: Source ids whose current metadata hash differs from
            the stored ``embedding.source_hash``. Reconcile would
            re-embed and replace the row.
        orphan: Row ids present in the bundle's ``rows`` list but
            missing from the catalog. Reconcile would drop the row.
    """

    missing: List[str] = field(default_factory=list)
    stale: List[str] = field(default_factory=list)
    orphan: List[str] = field(default_factory=list)

    def is_clean(self) -> bool:
        """True iff the bundle needs no reconcile work."""
        return not (self.missing or self.stale or self.orphan)

    def summary(self) -> str:
        """One-line human summary. ``"up-to-date"`` when clean."""
        if self.is_clean():
            return "up-to-date"
        parts = []
        if self.missing:
            parts.append(f"{len(self.missing)} missing")
        if self.stale:
            parts.append(f"{len(self.stale)} stale")
        if self.orphan:
            parts.append(f"{len(self.orphan)} orphan")
        return ", ".join(parts)


def detect_drift(
    bundle: ReferenceBundle,
    sources: List[ReferenceSource],
) -> DriftReport:
    """Compare the bundle's manifest against the live catalog.

    Only sources whose ``bundle_name`` matches this bundle are
    considered; this lets the plugin hold one flat catalog and still
    get per-bundle drift reports.

    Args:
        bundle: The bundle to inspect.
        sources: Full catalog (across all bundles). Filtered internally.

    Returns:
        A populated :class:`DriftReport`.
    """
    own_sources = {
        s.id: s for s in sources if s.bundle_name == bundle.name
    }
    rows_set = set(bundle.embedding_rows)

    missing: List[str] = []
    stale: List[str] = []
    for source_id, source in own_sources.items():
        if source_id not in rows_set:
            missing.append(source_id)
            continue
        if source.embedding is None:
            missing.append(source_id)
            continue
        if metadata_hash(source) != source.embedding.source_hash:
            stale.append(source_id)

    orphan = [sid for sid in bundle.embedding_rows if sid not in own_sources]

    return DriftReport(missing=missing, stale=stale, orphan=orphan)
