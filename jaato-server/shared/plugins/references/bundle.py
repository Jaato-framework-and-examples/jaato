"""Knowledge bundle abstraction for the references plugin.

A **bundle** is a self-contained unit of reference knowledge: its own
``embedding_config.json`` manifest, its own ``.npy`` sidecar matrix, and
its own set of reference JSON files. The root bundle lives directly under
``.jaato/references/``; additional bundles are immediate subdirectories
that contain their own ``embedding_config.json``.

This module owns:
    * ``Bundle`` — dataclass holding manifest + runtime state for one bundle
    * ``metadata_hash`` — the canonical fingerprint used by ``source_hash``
    * ``DriftReport`` + ``detect_drift`` — compare catalog vs. bundle manifest
    * ``discover_bundles`` — scan the references directory for bundles

Reconcile (writing updated sidecars) lives in :mod:`reconcile`; this module
is deliberately numpy-free so bundle discovery and drift detection work in
environments without the embedding provider installed.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .models import ReferenceSource

logger = logging.getLogger(__name__)


# Well-known filename for the per-bundle manifest. Duplicated from
# ``config_loader`` to keep ``bundle`` importable without the full loader.
EMBEDDING_CONFIG_FILENAME = "embedding_config.json"

# Sentinel name for the root bundle. Displayed as ``(root)`` to users.
ROOT_BUNDLE_NAME = ""

# Valid reconcile modes declared in a bundle manifest.
_VALID_RECONCILE_MODES: Set[str] = {"eager", "lazy", "off"}


@dataclass
class Bundle:
    """One knowledge bundle — a cohesive unit of reference metadata + vectors.

    Each bundle owns exactly one sidecar matrix and one manifest. The
    bundle's sources come from JSON files in its own directory and nowhere
    else; cross-bundle overlap is handled at the plugin level by
    namespacing (``<bundle>/<id>``).

    Lifecycle:
        1. ``discover_bundles`` walks ``.jaato/references/`` and creates
           one ``Bundle`` per directory that has an ``embedding_config.json``.
           At this point ``matcher`` is None and ``owned_source_ids`` is
           populated from the ``rows`` list.
        2. The plugin resolves actual ``ReferenceSource`` instances for
           each bundle (via ``discover_references`` on the bundle's
           directory) and stores the sources in its flat catalog. Each
           source carries ``bundle_name`` so membership is recoverable.
        3. On ``_init_bundle_matchers`` the plugin attaches a fresh
           ``SemanticMatcherProtocol`` instance per bundle whose embedding
           model matches the active provider.
        4. Reconcile (``reconcile.reconcile_bundle``) rewrites the manifest
           + sidecar on disk and the plugin re-attaches the matcher.

    Attributes:
        name: Bundle identifier. The root bundle uses ``ROOT_BUNDLE_NAME``
            (empty string); sub-bundles use their directory name.
        directory: Absolute path to the directory that owns this bundle's
            manifest + sidecar + reference JSON files.
        embedding_model: sentence-transformers model used to produce the
            sidecar vectors. A change invalidates every row.
        embedding_dimensions: Vector dimensionality. Must equal
            ``matrix.shape[1]`` when loaded.
        embedding_sidecar: Filename of the ``.npy`` file, relative to
            ``directory``.
        embedding_rows: Ordered list of reference ids — ``rows[i]`` is the
            id whose vector is at matrix row ``i``. Authoritative mapping
            from row to id; per-reference ``embedding.index`` does not exist.
        reconcile_mode: ``"eager"`` (reconcile during ``initialize``),
            ``"lazy"`` (reconcile before the first semantic query), or
            ``"off"`` (only reconcile when the operator runs
            ``references reconcile`` manually).
        owned_source_ids: Cached set of ids the bundle claims in its
            ``rows`` list. Populated on load; the live catalog is the
            source of truth for which ids actually exist.
        matcher: Attached ``SemanticMatcherProtocol`` instance; ``None``
            when the bundle has no compatible matcher (model mismatch,
            missing provider, empty rows, load failure). Not serialized.
    """

    name: str
    directory: Path
    embedding_model: str
    embedding_dimensions: int
    embedding_sidecar: str
    embedding_rows: List[str] = field(default_factory=list)
    reconcile_mode: str = "eager"
    owned_source_ids: Set[str] = field(default_factory=set)
    matcher: Optional[Any] = None

    @property
    def display_name(self) -> str:
        """Human-facing label for the bundle."""
        return "(root)" if self.name == ROOT_BUNDLE_NAME else self.name

    @property
    def manifest_path(self) -> Path:
        """Absolute path to this bundle's ``embedding_config.json``."""
        return self.directory / EMBEDDING_CONFIG_FILENAME

    @property
    def sidecar_path(self) -> Path:
        """Absolute path to this bundle's ``.npy`` sidecar matrix."""
        return self.directory / self.embedding_sidecar

    @property
    def lock_path(self) -> Path:
        """Advisory-lock filename used by the reconcile writer.

        A sibling of the sidecar so concurrent daemons targeting the same
        workspace serialize their rewrites.
        """
        return self.directory / (self.embedding_sidecar + ".lock")


def metadata_hash(source: ReferenceSource) -> str:
    """Compute the canonical fingerprint stored in ``embedding.source_hash``.

    We hash *metadata*, not content: the embedding is produced from
    ``name + description + tags + fetchHint`` (the text the ``gen-references``
    agent feeds to ``compute_embedding``), so the right staleness signal is
    "did any of that metadata drift?" Content-hashing would be wrong: a
    LOCAL reference's content can change independently without the vector
    needing to be regenerated, and URL/MCP references have no local content
    to hash at all.

    The format is ``sha256:<hex>``. Tags are sorted for stability across
    ordering-insensitive edits.

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
        missing: Source ids present in the catalog but not in ``rows``, or
            present without a stored ``source_hash``. Reconcile would embed
            these and append rows.
        stale: Source ids whose current metadata hash differs from the
            stored ``embedding.source_hash``. Reconcile would re-embed and
            replace the row.
        orphan: Row ids present in the bundle's ``rows`` list but missing
            from the catalog. Reconcile would drop the row.
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
    bundle: Bundle,
    sources: List[ReferenceSource],
) -> DriftReport:
    """Compare the bundle's manifest against the live catalog.

    Only sources whose ``bundle_name`` matches this bundle are considered;
    this lets the plugin hold one flat catalog and still get per-bundle
    drift reports.

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


def _load_bundle_from_manifest(
    manifest_path: Path,
    *,
    name: str,
) -> Optional[Bundle]:
    """Build a :class:`Bundle` from an ``embedding_config.json`` on disk.

    Returns ``None`` when the file is missing, unreadable, or malformed.
    Malformed bundles are logged but do not raise — a corrupt manifest in
    one subdirectory must not prevent the rest of the catalog from loading.

    Args:
        manifest_path: Absolute path to an ``embedding_config.json``.
        name: Bundle name (``""`` for root, subdir name for sub-bundles).

    Returns:
        The loaded :class:`Bundle`, or ``None`` on failure.
    """
    if not manifest_path.is_file():
        return None

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(
            "Bundle '%s': failed to read manifest %s: %s",
            name or "(root)", manifest_path, e,
        )
        return None

    if not isinstance(raw, dict):
        logger.warning(
            "Bundle '%s': manifest must be a JSON object: %s",
            name or "(root)", manifest_path,
        )
        return None

    model = raw.get("embedding_model")
    dims = raw.get("embedding_dimensions")
    sidecar = raw.get("embedding_sidecar")
    rows = raw.get("rows")
    reconcile_mode = raw.get("reconcile", "eager")

    if not model or not dims or not sidecar:
        logger.warning(
            "Bundle '%s': manifest missing required fields "
            "(embedding_model, embedding_dimensions, embedding_sidecar): %s",
            name or "(root)", manifest_path,
        )
        return None

    if not isinstance(rows, list) or not all(isinstance(r, str) for r in rows):
        logger.warning(
            "Bundle '%s': manifest 'rows' must be a list of reference ids: %s",
            name or "(root)", manifest_path,
        )
        return None

    if reconcile_mode not in _VALID_RECONCILE_MODES:
        logger.warning(
            "Bundle '%s': unknown reconcile mode %r, falling back to 'eager'",
            name or "(root)", reconcile_mode,
        )
        reconcile_mode = "eager"

    return Bundle(
        name=name,
        directory=manifest_path.parent.resolve(),
        embedding_model=str(model),
        embedding_dimensions=int(dims),
        embedding_sidecar=str(sidecar),
        embedding_rows=list(rows),
        reconcile_mode=reconcile_mode,
        owned_source_ids=set(rows),
    )


def discover_bundles(
    references_dir: Path,
) -> List[Bundle]:
    """Scan a references directory for knowledge bundles.

    Discovers the root bundle (manifest at the top level) followed by each
    immediate subdirectory that contains its own ``embedding_config.json``.
    Subdirectories without a manifest are ignored entirely — they are not
    merged into the root bundle — so dropping an unrelated directory into
    ``.jaato/references/`` never accidentally pollutes the catalog.

    Args:
        references_dir: Absolute path to the workspace references directory
            (typically ``<workspace>/.jaato/references``).

    Returns:
        List of :class:`Bundle` in deterministic order: root first, then
        sub-bundles sorted by directory name.
    """
    bundles: List[Bundle] = []

    if not references_dir.is_dir():
        return bundles

    root = _load_bundle_from_manifest(
        references_dir / EMBEDDING_CONFIG_FILENAME,
        name=ROOT_BUNDLE_NAME,
    )
    if root is not None:
        bundles.append(root)

    for child in sorted(references_dir.iterdir()):
        if not child.is_dir():
            continue
        manifest = child / EMBEDDING_CONFIG_FILENAME
        if not manifest.is_file():
            continue
        sub = _load_bundle_from_manifest(manifest, name=child.name)
        if sub is not None:
            bundles.append(sub)

    return bundles


def write_manifest(bundle: Bundle, *, rows: List[str]) -> None:
    """Write the bundle's manifest to disk with an updated ``rows`` list.

    Uses an atomic write (``.tmp`` + rename) so a crash mid-write cannot
    corrupt the manifest.

    Args:
        bundle: Target bundle.
        rows: New ordered list of reference ids. ``len(rows)`` must equal
            the new sidecar matrix's row count.
    """
    payload: Dict[str, Any] = {
        "embedding_model": bundle.embedding_model,
        "embedding_dimensions": bundle.embedding_dimensions,
        "embedding_sidecar": bundle.embedding_sidecar,
        "rows": rows,
    }
    if bundle.reconcile_mode != "eager":
        payload["reconcile"] = bundle.reconcile_mode

    tmp = bundle.manifest_path.with_suffix(bundle.manifest_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(bundle.manifest_path)
    bundle.embedding_rows = list(rows)
    bundle.owned_source_ids = set(rows)
