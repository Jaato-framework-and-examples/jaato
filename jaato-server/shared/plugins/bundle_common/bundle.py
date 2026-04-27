"""Domain-agnostic bundle abstraction.

A **bundle** is a self-contained unit of grouped content stored on disk
under one of two tier roots:

* **workspace** tier — under ``<workspace>/.jaato/<domain>/`` (per-project
  content that travels with the repository).
* **user** tier — under ``~/.jaato/<domain>/`` (cross-project personal
  content that follows the user across workspaces).

The exact ``<domain>`` (``references``, ``agents``, ``tasks``, ...) is
chosen by the calling plugin via :func:`resolve_bundle_roots`. Within
each tier root, the *root bundle* is the manifest at the top level;
additional bundles are immediate subdirectories that contain their own
``embedding_config.json``. Discovery walks the workspace tier first,
then the user tier; when the same bundle name exists in both tiers the
workspace copy **shadows** the user copy entirely (it is hidden from
discovery), the same way ``.jaato/theme.json`` shadows
``~/.jaato/theme.json``.

This module owns the pieces that have no domain affinity:

* :class:`Bundle` — dataclass holding the manifest + runtime state
* :data:`BUNDLE_TIER_WORKSPACE` / :data:`BUNDLE_TIER_USER` — tier ids
* :func:`resolve_bundle_roots` — ordered ``(root, tier)`` list for discovery
* :func:`discover_bundles` — scan one or more roots for bundles
* :class:`BundleRef`, :func:`parse_bundle_ref`, :func:`find_bundle`,
  :exc:`AmbiguousBundleRefError` — the ``[<scope>:]<name>`` user
  reference syntax shared by every bundle-aware command
* :func:`write_manifest` — atomic manifest write

The :class:`Bundle` dataclass currently carries embedding-specific
fields (``embedding_model``, ``embedding_dimensions``, ``embedding_sidecar``,
``embedding_rows``). They are populated by the references plugin's
manifests; future generalization will make them optional or factor
them into a sidecar metadata sub-object so non-embedded domains
(profiles, services, agents, tasks) can use the same dataclass without
fabricating dummy values.

References-specific concepts that depend on ``ReferenceSource`` —
``metadata_hash``, ``DriftReport``, ``detect_drift`` — remain in
:mod:`shared.plugins.references.bundle` and are not re-exported here.

This module is deliberately numpy-free so bundle discovery and
reference parsing work in environments without an embedding provider
installed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


# Well-known filename for the per-bundle manifest. Today this is the
# references-style ``embedding_config.json``; non-embedded domains will
# eventually adopt a generic ``bundle.json`` (with the embedding section
# moved into an optional ``embedding`` sub-object). Until that
# generalization lands, the constant keeps the existing on-disk shape
# stable so the references plugin can keep loading every bundle it has
# already produced.
EMBEDDING_CONFIG_FILENAME = "embedding_config.json"

# Sentinel name for the root bundle. Displayed as ``(root)`` to users.
ROOT_BUNDLE_NAME = ""

# Valid reconcile modes declared in a bundle manifest.
_VALID_RECONCILE_MODES: Set[str] = {"eager", "lazy", "off"}

# Tier identifiers. ``BUNDLE_TIER_WORKSPACE`` is per-project; the bundle
# lives under ``<workspace>/.jaato/<domain>/`` and travels with the
# repo. ``BUNDLE_TIER_USER`` is per-user; the bundle lives under
# ``~/.jaato/<domain>/`` and follows the user across workspaces.
BUNDLE_TIER_WORKSPACE = "workspace"
BUNDLE_TIER_USER = "user"
VALID_BUNDLE_TIERS: Tuple[str, ...] = (BUNDLE_TIER_WORKSPACE, BUNDLE_TIER_USER)


@dataclass
class Bundle:
    """One bundle — a cohesive unit of content + (optional) vector index.

    Each bundle owns at most one sidecar matrix and one manifest. The
    bundle's entries come from JSON files in its own directory and
    nowhere else; cross-bundle overlap is handled at the plugin level
    by namespacing.

    The dataclass currently carries embedding-related fields populated
    by the references plugin. Non-embedded domains (agents, tasks,
    profiles, services) will populate them with placeholder / empty
    values until a future commit makes them strictly optional via a
    sidecar sub-object.

    Attributes:
        name: Bundle identifier. The root bundle uses
            :data:`ROOT_BUNDLE_NAME` (empty string); sub-bundles use
            their directory name. The same ``name`` may exist in
            multiple tiers, but discovery shadows the user-tier copy
            when a workspace-tier copy is present.
        directory: Absolute path to the directory that owns this
            bundle's manifest, sidecar, and entry JSON files.
        tier: Which tier root this bundle was discovered under — either
            :data:`BUNDLE_TIER_WORKSPACE` or :data:`BUNDLE_TIER_USER`.
            Drives presentation and the destination of write commands.
        embedding_model: sentence-transformers model used to produce
            the sidecar vectors. References-specific.
        embedding_dimensions: Vector dimensionality. Must equal
            ``matrix.shape[1]`` when the sidecar is loaded.
        embedding_sidecar: Filename of the ``.npy`` file, relative to
            ``directory``.
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
            bundle has no compatible matcher (model mismatch, missing
            provider, empty rows, load failure). Not serialized.
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

        Use this when the bundle name alone is ambiguous (e.g.,
        logging, error messages, ``bundles`` rendering). The root
        bundle still renders as ``(root)`` for the name component.
        """
        return f"{self.tier}:{self.display_name}"

    @property
    def manifest_path(self) -> Path:
        """Absolute path to this bundle's manifest file."""
        return self.directory / EMBEDDING_CONFIG_FILENAME

    @property
    def sidecar_path(self) -> Path:
        """Absolute path to this bundle's ``.npy`` sidecar matrix."""
        return self.directory / self.embedding_sidecar

    @property
    def lock_path(self) -> Path:
        """Advisory-lock filename used by the reconcile writer.

        A sibling of the sidecar so concurrent daemons targeting the
        same workspace serialize their rewrites.
        """
        return self.directory / (self.embedding_sidecar + ".lock")


def _load_bundle_from_manifest(
    manifest_path: Path,
    *,
    name: str,
    tier: str = BUNDLE_TIER_WORKSPACE,
) -> Optional[Bundle]:
    """Build a :class:`Bundle` from a manifest file on disk.

    Returns ``None`` when the file is missing, unreadable, or malformed.
    Malformed bundles are logged but do not raise — a corrupt manifest
    in one subdirectory must not prevent the rest of the catalog from
    loading.

    Args:
        manifest_path: Absolute path to a manifest JSON file.
        name: Bundle name (``""`` for root, subdir name for sub-bundles).
        tier: Which tier root this bundle was discovered under. Stored
            on the resulting ``Bundle.tier`` so downstream commands
            know where the bundle physically lives.

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
            "Bundle '%s': manifest 'rows' must be a list of entry ids: %s",
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
    domain_subpath: Path,
    user_home: Optional[Path] = None,
) -> List[Tuple[Path, str]]:
    """Return the ordered list of ``(root_dir, tier)`` pairs to scan.

    Discovery walks workspace first, then user; the order matters
    because workspace bundles **shadow** user bundles of the same
    name in :func:`discover_bundles`.

    Args:
        workspace_path: Workspace root, or ``None`` if unknown. When
            None, the workspace tier is omitted (the user tier still
            applies).
        domain_subpath: Plugin-relative tier-root suffix, e.g.
            ``Path(".jaato/references")`` for the references plugin or
            ``Path(".jaato/agents")`` for an agent plugin. Joined with
            both the workspace path and ``Path.home()``.
        user_home: Override for ``Path.home()``. Test seam — production
            code passes ``None`` to use the real home directory.

    Returns:
        Ordered list of ``(absolute_root_dir, tier_name)``. Roots that
        do not exist on disk are still returned; ``discover_bundles``
        treats a missing root as "no bundles in this tier" without
        raising.
    """
    roots: List[Tuple[Path, str]] = []
    if workspace_path is not None:
        roots.append((
            Path(workspace_path).resolve() / domain_subpath,
            BUNDLE_TIER_WORKSPACE,
        ))
    home = user_home if user_home is not None else Path.home()
    roots.append((home / domain_subpath, BUNDLE_TIER_USER))
    return roots


def discover_bundles(
    roots: Union[Path, Sequence[Tuple[Path, str]]],
) -> List[Bundle]:
    """Scan one or more bundle directories for loadable bundles.

    Two calling conventions are supported:

    * **Single-root (legacy):** ``discover_bundles(path)`` — scans the
      given directory as the workspace tier. Kept so existing callers
      and tests don't need to know about tiering.
    * **Multi-root (preferred):** ``discover_bundles([(path, tier), ...])``
      — scans each ``(root, tier)`` pair in order. The first tier wins
      on name collisions: if both the workspace and user tiers contain
      a ``teammate`` bundle, the workspace copy is returned and the
      user copy is silently shadowed (a debug log records the shadow).

    Within each root, the root bundle (manifest at the top level) is
    discovered first, followed by each immediate subdirectory that
    contains its own manifest. Subdirectories without a manifest are
    ignored entirely so dropping an unrelated directory into a tier
    root never accidentally pollutes the catalog.

    Shadowing keys on bundle ``name`` (the root bundle name is the
    empty string :data:`ROOT_BUNDLE_NAME`); a workspace root manifest
    shadows a user root manifest, and ``workspace/teammate`` shadows
    ``user/teammate``.

    Args:
        roots: Either a single ``Path`` (legacy form, treated as the
            workspace tier) or a sequence of ``(root_dir, tier_name)``
            tuples in the order discovery should walk them.

    Returns:
        List of :class:`Bundle` in deterministic order: per root, the
        root bundle first then sub-bundles sorted by directory name;
        tiers are concatenated in input order.
    """
    if isinstance(roots, Path):
        normalized: Sequence[Tuple[Path, str]] = (
            (roots, BUNDLE_TIER_WORKSPACE),
        )
    else:
        normalized = roots

    bundles: List[Bundle] = []
    seen_names: Set[str] = set()

    for refs_dir, tier in normalized:
        if not refs_dir.is_dir():
            continue

        root = _load_bundle_from_manifest(
            refs_dir / EMBEDDING_CONFIG_FILENAME,
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

        for child in sorted(refs_dir.iterdir()):
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


@dataclass(frozen=True)
class BundleRef:
    """Parsed user-supplied bundle reference (``[<scope>:]<name>``).

    Two states are distinguished:

    * ``scope`` set — caller wrote ``workspace:teammate`` or
      ``user:teammate``; the tier is unambiguous and resolution will
      reject any bundle in another tier.
    * ``scope`` is ``None`` — caller wrote a bare ``teammate``;
      resolution tries the supplied default tier first and falls
      through to the other tier when no match exists, surfacing an
      "ambiguous" error only when both tiers contain a bundle of that
      name.

    The ``(root)`` and ``root`` aliases both normalize to
    :data:`ROOT_BUNDLE_NAME` so users don't have to remember the
    empty-string sentinel.
    """

    name: str
    scope: Optional[str] = None  # None means "tier not specified"

    @property
    def display(self) -> str:
        """Human-readable form of this ref, suitable for error messages."""
        name = "(root)" if self.name == ROOT_BUNDLE_NAME else self.name
        return f"{self.scope}:{name}" if self.scope else name


def parse_bundle_ref(raw: str) -> BundleRef:
    """Parse ``[<scope>:]<name>`` user input into a :class:`BundleRef`.

    Accepts the following forms (whitespace is stripped):

    * ``"teammate"`` → ``BundleRef(name="teammate", scope=None)``
    * ``"workspace:teammate"`` → ``BundleRef(name="teammate", scope="workspace")``
    * ``"user:teammate"`` → ``BundleRef(name="teammate", scope="user")``
    * ``"root"`` or ``"(root)"`` → ``BundleRef(name=ROOT_BUNDLE_NAME, scope=None)``
    * ``"workspace:root"`` / ``"workspace:(root)"`` → root with scope set

    Raises:
        ValueError: ``raw`` is empty, contains an unknown scope, or
            has an empty name component (e.g. ``"workspace:"``).
    """
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("bundle reference is empty")

    scope: Optional[str] = None
    name = raw
    if ":" in raw:
        scope_part, _, name_part = raw.partition(":")
        scope_part = scope_part.strip()
        name_part = name_part.strip()
        if scope_part not in VALID_BUNDLE_TIERS:
            raise ValueError(
                f"unknown scope {scope_part!r} in {raw!r}; expected one of "
                f"{', '.join(VALID_BUNDLE_TIERS)}"
            )
        if not name_part:
            raise ValueError(
                f"missing bundle name after scope in {raw!r} "
                f"(write '{scope_part}:root' for the root bundle)"
            )
        scope = scope_part
        name = name_part

    if name in ("root", "(root)"):
        name = ROOT_BUNDLE_NAME

    return BundleRef(name=name, scope=scope)


class AmbiguousBundleRefError(ValueError):
    """A bare bundle name matched bundles in more than one tier.

    Raised by :func:`find_bundle` when the caller did not specify a
    scope and the name appears in both the workspace and user tiers.
    The :attr:`candidates` list lets the caller render a helpful
    "did you mean ``workspace:teammate`` or ``user:teammate``?" error.
    """

    def __init__(self, ref: BundleRef, candidates: List[Bundle]) -> None:
        names = ", ".join(sorted(b.qualified_ref for b in candidates))
        super().__init__(
            f"bundle '{ref.display}' is ambiguous — present in multiple "
            f"tiers ({names}); qualify it as 'workspace:{ref.display}' "
            f"or 'user:{ref.display}'"
        )
        self.ref = ref
        self.candidates = list(candidates)


def find_bundle(
    bundles: Iterable[Bundle],
    ref: BundleRef,
    *,
    default_scope: Optional[str] = None,
) -> Optional[Bundle]:
    """Resolve a :class:`BundleRef` against a list of loaded bundles.

    Resolution rules:

    1. If ``ref.scope`` is set, only bundles with the matching tier
       are considered. Returns the unique match, or ``None`` if none
       exists.
    2. Otherwise:
        * If ``default_scope`` is supplied and a bundle with the given
          name exists in that tier, return it (used by write commands
          to resolve bare names against the workspace tier first).
        * Else if exactly one bundle has the name (in any tier),
          return it (the unambiguous case).
        * Else if multiple bundles match across tiers, raise
          :class:`AmbiguousBundleRefError`.
        * Else return ``None``.

    Args:
        bundles: All loaded bundles (typically the plugin's
            ``self._bundles``).
        ref: Parsed user input.
        default_scope: Tier to prefer when the user wrote a bare name.
            Set to :data:`BUNDLE_TIER_WORKSPACE` for write commands
            and ``None`` for read commands that treat tiers
            symmetrically.

    Returns:
        The matching :class:`Bundle`, or ``None`` if no bundle matches.

    Raises:
        AmbiguousBundleRefError: bare-name lookup matched bundles in
            multiple tiers and ``default_scope`` did not break the tie.
    """
    bundle_list = list(bundles)

    if ref.scope is not None:
        for b in bundle_list:
            if b.tier == ref.scope and b.name == ref.name:
                return b
        return None

    matches = [b for b in bundle_list if b.name == ref.name]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    if default_scope is not None:
        for b in matches:
            if b.tier == default_scope:
                return b

    raise AmbiguousBundleRefError(ref, matches)


def write_manifest(bundle: Bundle, *, rows: List[str]) -> None:
    """Write the bundle's manifest to disk with an updated ``rows`` list.

    Uses an atomic write (``.tmp`` + rename) so a crash mid-write
    cannot corrupt the manifest.

    Args:
        bundle: Target bundle.
        rows: New ordered list of entry ids. ``len(rows)`` must equal
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
