"""Knowledge bundle abstraction for the references plugin.

A **bundle** is a self-contained unit of reference knowledge: its own
``embedding_config.json`` manifest, its own ``.npy`` sidecar matrix, and
its own set of reference JSON files. Bundles live in two tiers:

* **workspace** tier — under ``<workspace>/.jaato/references/`` (per-project
  knowledge that travels with the repository).
* **user** tier — under ``~/.jaato/references/`` (cross-project personal
  knowledge that follows the user across workspaces).

In each tier the root bundle is the manifest at the top level; additional
bundles are immediate subdirectories that contain their own
``embedding_config.json``. Discovery walks the workspace tier first, then
the user tier; when the same bundle name exists in both tiers the workspace
copy **shadows** the user copy entirely (it is hidden from discovery), the
same way ``.jaato/theme.json`` shadows ``~/.jaato/theme.json``.

This module owns:
    * ``Bundle`` — dataclass holding manifest + runtime state for one bundle
    * ``BUNDLE_TIER_*`` — tier identifiers exposed on ``Bundle.tier``
    * ``resolve_bundle_roots`` — ordered (root, tier) list for discovery
    * ``metadata_hash`` — the canonical fingerprint used by ``source_hash``
    * ``DriftReport`` + ``detect_drift`` — compare catalog vs. bundle manifest
    * ``discover_bundles`` — scan one or more references directories for bundles

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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from .models import ReferenceSource

logger = logging.getLogger(__name__)


# Well-known filename for the per-bundle manifest. Duplicated from
# ``config_loader`` to keep ``bundle`` importable without the full loader.
EMBEDDING_CONFIG_FILENAME = "embedding_config.json"

# Sentinel name for the root bundle. Displayed as ``(root)`` to users.
ROOT_BUNDLE_NAME = ""

# Valid reconcile modes declared in a bundle manifest.
_VALID_RECONCILE_MODES: Set[str] = {"eager", "lazy", "off"}

# Tier identifiers. ``BUNDLE_TIER_WORKSPACE`` is per-project; the bundle
# lives under ``<workspace>/.jaato/references/`` and travels with the repo.
# ``BUNDLE_TIER_USER`` is per-user; the bundle lives under
# ``~/.jaato/references/`` and follows the user across workspaces.
BUNDLE_TIER_WORKSPACE = "workspace"
BUNDLE_TIER_USER = "user"
VALID_BUNDLE_TIERS: Tuple[str, ...] = (BUNDLE_TIER_WORKSPACE, BUNDLE_TIER_USER)

# Subpath under each tier root where bundles live.
_REFERENCES_SUBPATH = Path(".jaato") / "references"


@dataclass
class Bundle:
    """One knowledge bundle — a cohesive unit of reference metadata + vectors.

    Each bundle owns exactly one sidecar matrix and one manifest. The
    bundle's sources come from JSON files in its own directory and nowhere
    else; cross-bundle overlap is handled at the plugin level by
    namespacing (``<bundle>/<id>``).

    Lifecycle:
        1. ``discover_bundles`` walks the configured tier roots (workspace
           first, then user) and creates one ``Bundle`` per directory that
           has an ``embedding_config.json``. Each bundle is tagged with the
           tier it was found in (see :attr:`tier`). At this point
           ``matcher`` is None and ``owned_source_ids`` is populated from
           the ``rows`` list.
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
            (empty string); sub-bundles use their directory name. The same
            ``name`` may exist in multiple tiers, but discovery shadows
            the user-tier copy when a workspace-tier copy is present.
        directory: Absolute path to the directory that owns this bundle's
            manifest + sidecar + reference JSON files.
        tier: Which tier root this bundle was discovered under — either
            ``BUNDLE_TIER_WORKSPACE`` (lives in ``<workspace>/.jaato/
            references/``) or ``BUNDLE_TIER_USER`` (lives in
            ``~/.jaato/references/``). Drives presentation (``references
            bundles`` shows a tier column) and the destination of write
            commands like ``reconcile``, ``merge --into``, and ``unpack``.
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
    tier: str = BUNDLE_TIER_WORKSPACE

    @property
    def display_name(self) -> str:
        """Human-facing label for the bundle."""
        return "(root)" if self.name == ROOT_BUNDLE_NAME else self.name

    @property
    def qualified_ref(self) -> str:
        """``scope:name`` string identifying this bundle across tiers.

        Use this when the bundle name alone is ambiguous (e.g., logging,
        error messages, ``references bundles`` rendering). The root
        bundle still renders as ``(root)`` for the name component.
        """
        return f"{self.tier}:{self.display_name}"

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
    tier: str = BUNDLE_TIER_WORKSPACE,
) -> Optional[Bundle]:
    """Build a :class:`Bundle` from an ``embedding_config.json`` on disk.

    Returns ``None`` when the file is missing, unreadable, or malformed.
    Malformed bundles are logged but do not raise — a corrupt manifest in
    one subdirectory must not prevent the rest of the catalog from loading.

    Args:
        manifest_path: Absolute path to an ``embedding_config.json``.
        name: Bundle name (``""`` for root, subdir name for sub-bundles).
        tier: Which tier root this bundle was discovered under. Stored on
            the resulting ``Bundle.tier`` so downstream commands know
            where the bundle physically lives.

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

    if tier not in VALID_BUNDLE_TIERS:
        logger.warning(
            "Bundle '%s': unknown tier %r, falling back to %r",
            name or "(root)", tier, BUNDLE_TIER_WORKSPACE,
        )
        tier = BUNDLE_TIER_WORKSPACE

    return Bundle(
        name=name,
        directory=manifest_path.parent.resolve(),
        embedding_model=str(model),
        embedding_dimensions=int(dims),
        embedding_sidecar=str(sidecar),
        embedding_rows=list(rows),
        reconcile_mode=reconcile_mode,
        owned_source_ids=set(rows),
        tier=tier,
    )


def resolve_bundle_roots(
    workspace_path: Optional[Union[str, Path]],
    *,
    user_home: Optional[Path] = None,
) -> List[Tuple[Path, str]]:
    """Return the ordered list of ``(root_dir, tier)`` pairs to scan.

    Discovery walks workspace first, then user; the order matters because
    workspace bundles **shadow** user bundles of the same name in
    :func:`discover_bundles`.

    Args:
        workspace_path: Workspace root, or ``None`` if unknown. When None,
            the workspace tier is omitted (the user tier still applies).
        user_home: Override for ``Path.home()``. Test seam — production
            code passes ``None`` to use the real home directory.

    Returns:
        Ordered list of ``(absolute_root_dir, tier_name)``. Roots that do
        not exist on disk are still returned; ``discover_bundles`` treats
        a missing root as "no bundles in this tier" without raising.
    """
    roots: List[Tuple[Path, str]] = []
    if workspace_path is not None:
        roots.append((
            Path(workspace_path).resolve() / _REFERENCES_SUBPATH,
            BUNDLE_TIER_WORKSPACE,
        ))
    home = user_home if user_home is not None else Path.home()
    roots.append((home / _REFERENCES_SUBPATH, BUNDLE_TIER_USER))
    return roots


def discover_bundles(
    roots: Union[Path, Sequence[Tuple[Path, str]]],
) -> List[Bundle]:
    """Scan one or more references directories for knowledge bundles.

    Two calling conventions are supported:

    * **Single-root (legacy):** ``discover_bundles(path)`` — scans the
      given directory as the workspace tier. Kept so existing callers and
      tests don't need to know about tiering.
    * **Multi-root (preferred):** ``discover_bundles([(path, tier), ...])``
      — scans each ``(root, tier)`` pair in order. The first tier wins on
      name collisions: if both the workspace and user tiers contain a
      ``teammate`` bundle, the workspace copy is returned and the user
      copy is silently shadowed (a debug log records the shadow).

    Within each root, the root bundle (manifest at the top level) is
    discovered first, followed by each immediate subdirectory that contains
    its own ``embedding_config.json``. Subdirectories without a manifest are
    ignored entirely — they are not merged into the root bundle — so
    dropping an unrelated directory into ``.jaato/references/`` never
    accidentally pollutes the catalog.

    Shadowing keys on bundle ``name`` (the root bundle name is the empty
    string ``ROOT_BUNDLE_NAME``); a workspace root manifest shadows a user
    root manifest, and ``workspace/teammate`` shadows ``user/teammate``.

    Args:
        roots: Either a single ``Path`` (legacy form, treated as the
            workspace tier) or a sequence of ``(root_dir, tier_name)``
            tuples in the order discovery should walk them.

    Returns:
        List of :class:`Bundle` in deterministic order: per root, the root
        bundle first then sub-bundles sorted by directory name; tiers are
        concatenated in input order.
    """
    if isinstance(roots, Path):
        normalized: Sequence[Tuple[Path, str]] = (
            (roots, BUNDLE_TIER_WORKSPACE),
        )
    else:
        normalized = roots

    bundles: List[Bundle] = []
    seen_names: Set[str] = set()

    for references_dir, tier in normalized:
        if not references_dir.is_dir():
            continue

        root = _load_bundle_from_manifest(
            references_dir / EMBEDDING_CONFIG_FILENAME,
            name=ROOT_BUNDLE_NAME,
            tier=tier,
        )
        if root is not None:
            if root.name in seen_names:
                logger.debug(
                    "discover_bundles: shadowing %s root bundle at %s "
                    "(already provided by an earlier tier)",
                    tier, root.directory,
                )
            else:
                bundles.append(root)
                seen_names.add(root.name)

        for child in sorted(references_dir.iterdir()):
            if not child.is_dir():
                continue
            manifest = child / EMBEDDING_CONFIG_FILENAME
            if not manifest.is_file():
                continue
            sub = _load_bundle_from_manifest(manifest, name=child.name, tier=tier)
            if sub is None:
                continue
            if sub.name in seen_names:
                logger.debug(
                    "discover_bundles: shadowing %s bundle '%s' at %s "
                    "(already provided by an earlier tier)",
                    tier, sub.name, sub.directory,
                )
                continue
            bundles.append(sub)
            seen_names.add(sub.name)

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
