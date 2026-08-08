"""References-specific bundle helpers.

The domain-agnostic parts of bundle management — ``Bundle``, the tier
constants, ``BundleRef`` / ``parse_bundle_ref`` / ``find_bundle``,
``discover_bundles``, ``write_manifest`` — live in
:mod:`shared.plugins.bundle_common.bundle` so future plugins (agents,
tasks, profiles, services) can reuse the same machinery. This module
re-exports those symbols for the references plugin's existing import
sites and adds the references-specific pieces that depend on
:class:`ReferenceSource`:

* :func:`metadata_hash` — fingerprint stored in
  ``ReferenceSource.embedding.source_hash``.
* :class:`DriftReport` + :func:`detect_drift` — compare a bundle
  manifest against the live reference catalog.

The shim also wraps :func:`bundle_common.bundle.resolve_bundle_roots`
to bake in the references-domain subpath (``.jaato/references``) so
existing call sites (and tests) don't have to know about the new
``domain_subpath`` argument.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Re-export the domain-agnostic surface so existing imports
# (``from shared.plugins.references.bundle import …``) keep working.
from ..bundle_common.bundle import (  # noqa: F401
    BUNDLE_TIER_USER,
    BUNDLE_TIER_WORKSPACE,
    EMBEDDING_CONFIG_FILENAME,
    ROOT_BUNDLE_NAME,
    VALID_BUNDLE_TIERS,
    AmbiguousBundleRefError,
    Bundle,
    BundleRef,
    _load_bundle_from_manifest,
    discover_bundles,
    find_bundle,
    parse_bundle_ref,
    write_manifest,
)
from ..bundle_common.bundle import (
    resolve_bundle_roots as _resolve_bundle_roots_generic,
)

from .models import ReferenceSource

logger = logging.getLogger(__name__)


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
    bundle: Bundle,
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
