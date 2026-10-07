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
additional bundles are immediate subdirectories that carry their own
manifest. Discovery walks the workspace tier first, then the user tier;
when the same bundle name exists in both tiers the workspace copy
**shadows** the user copy entirely (it is hidden from discovery), the
same way ``.jaato/theme.json`` shadows ``~/.jaato/theme.json``.

This module owns the pieces that have no domain affinity:

* :class:`Bundle` — ``name`` + ``directory`` + ``tier``, and nothing else
* :data:`BUNDLE_TIER_WORKSPACE` / :data:`BUNDLE_TIER_USER` — tier ids
* :func:`resolve_bundle_roots` — ordered ``(root, tier)`` list for discovery
* :func:`discover_bundles` — scan one or more roots for bundles
* :class:`BundleRef`, :func:`parse_bundle_ref`, :func:`find_bundle`,
  :exc:`AmbiguousBundleRefError` — the ``[<scope>:]<name>`` user
  reference syntax shared by every bundle-aware command
* :func:`write_bundle_manifest` — atomic write of the generic manifest

**What a bundle is, and what it is not.** A bundle is a *directory a
domain claims*. Everything that describes the CONTENT of that directory
— which entries it holds, whether it carries a vector index, how it is
reconciled — belongs to the domain, not here. The references plugin's
embedding fields (``embedding_model``, ``embedding_dimensions``,
``embedding_sidecar``, ``embedding_rows``, ``reconcile_mode``) and its
runtime state (``matcher``, ``owned_source_ids``) used to sit on
:class:`Bundle`; they live on
:class:`shared.plugins.references.bundle.ReferenceBundle` now. Three
consequences follow, and each was a defect before the move:

* a domain with no vector index (``agents``, ``tasks``, ``profiles``,
  ``services``) can describe a bundle without fabricating dummy
  embedding values;
* a references directory holding definitions and no sidecar is
  discovered rather than silently dropped;
* :func:`shared.plugins.bundle_common.pack.pack_bundle`, already
  parametric over ``BundleEntryHandler``, is reachable for those
  domains — its entry point no longer takes a type that cannot
  express a vectorless bundle.

**The anti-pollution guard is unchanged.** A subdirectory without a
manifest is still ignored entirely, so dropping an unrelated directory
into a tier root never accidentally pollutes the catalog. Only the
manifest's *schema* changed, not the need for one.

This module is deliberately numpy-free so bundle discovery and
reference parsing work in environments without an embedding provider
installed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
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


# Canonical filename for the generic per-bundle manifest. Its schema is
# open: every key is optional and domain-agnostic (``name``,
# ``description``). Its *presence* is what marks a directory as a
# bundle — that is the whole of the anti-pollution guard.
BUNDLE_MANIFEST_FILENAME = "bundle.json"

# Sentinel name for the root bundle. Displayed as ``(root)`` to users.
ROOT_BUNDLE_NAME = ""

# Tier identifiers. ``BUNDLE_TIER_WORKSPACE`` is per-project; the bundle
# lives under ``<workspace>/.jaato/<domain>/`` and travels with the
# repo. ``BUNDLE_TIER_USER`` is per-user; the bundle lives under
# ``~/.jaato/<domain>/`` and follows the user across workspaces.
BUNDLE_TIER_WORKSPACE = "workspace"
BUNDLE_TIER_USER = "user"
VALID_BUNDLE_TIERS: Tuple[str, ...] = (BUNDLE_TIER_WORKSPACE, BUNDLE_TIER_USER)


@dataclass
class Bundle:
    """One bundle — a directory a domain claims, and where it lives.

    Three fields, all domain-agnostic. A domain that needs more about
    its own bundles (a vector index, per-bundle runtime state, a
    reconcile policy) subclasses this in its own package rather than
    widening it here; see
    :class:`shared.plugins.references.bundle.ReferenceBundle`.

    Attributes:
        name: Bundle identifier. The root bundle uses
            :data:`ROOT_BUNDLE_NAME` (empty string); sub-bundles use
            their directory name. The same ``name`` may exist in
            multiple tiers, but discovery shadows the user-tier copy
            when a workspace-tier copy is present.
        directory: Absolute path to the directory that owns this
            bundle's manifest and entry files.
        tier: Which tier root this bundle was discovered under — either
            :data:`BUNDLE_TIER_WORKSPACE` or :data:`BUNDLE_TIER_USER`.
            Drives presentation and the destination of write commands.
    """

    name: str
    directory: Path
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
        """Where the manifest belongs — the path a writer uses.

        Always ``<directory>/bundle.json``, whether or not that file
        exists yet. There is exactly one marker, so asking where it
        belongs and asking which one a bundle carries are the same
        question; :func:`is_bundle_directory` answers whether it is
        there.
        """
        return self.directory / BUNDLE_MANIFEST_FILENAME


def is_bundle_directory(directory: Path) -> bool:
    """Whether *directory* carries a bundle manifest.

    ONE marker: ``bundle.json``.  A directory is a bundle because a
    domain claimed it, never because of what it happens to contain --
    which is the whole of #1130.  In particular the references plugin's
    ``embedding_config.json`` is an index descriptor that sits beside
    the manifest and marks nothing.
    """
    try:
        return (directory / BUNDLE_MANIFEST_FILENAME).is_file()
    except OSError:
        return False


def load_bundle(
    directory: Path,
    *,
    name: str,
    tier: str = BUNDLE_TIER_WORKSPACE,
) -> Optional[Bundle]:
    """Build a :class:`Bundle` for ``directory``, or ``None``.

    Returns ``None`` only when ``directory`` carries no manifest —
    i.e. when it is not a bundle at all. A manifest that is present but
    malformed still identifies a bundle: the file is the marker, and
    refusing to load the directory because a domain could not parse its
    own metadata is what made a definitions-only references directory
    invisible. Malformed content is logged and otherwise ignored here;
    the owning domain decides what a body it cannot read means.

    Args:
        directory: Candidate bundle directory.
        name: Bundle name (``""`` for root, subdir name for sub-bundles).
        tier: Which tier root this bundle was discovered under. Stored
            on the resulting ``Bundle.tier`` so downstream commands
            know where the bundle physically lives.

    Returns:
        The loaded :class:`Bundle`, or ``None`` when ``directory`` is
        not a bundle.
    """
    manifest = directory / BUNDLE_MANIFEST_FILENAME
    try:
        if not manifest.is_file():
            return None
    except OSError:
        return None

    # Read it only to report a corrupt file. Nothing in the manifest is
    # required, so there is nothing to refuse over -- its PRESENCE is
    # the claim, not its contents.
    try:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(
            "Bundle '%s': failed to read manifest %s: %s",
            name or "(root)", manifest, e,
        )
    else:
        if not isinstance(raw, dict):
            logger.warning(
                "Bundle '%s': manifest must be a JSON object: %s",
                name or "(root)", manifest,
            )

    if tier not in VALID_BUNDLE_TIERS:
        logger.warning(
            "Bundle '%s': unknown tier %r, falling back to %r",
            name or "(root)", tier, BUNDLE_TIER_WORKSPACE,
        )
        tier = BUNDLE_TIER_WORKSPACE

    return Bundle(
        name=name,
        directory=directory.resolve(),
        tier=tier,
    )


def resolve_bundle_roots(
    workspace_path: Optional[Union[str, Path]],
    *,
    domain_subpath: Path,
    user_home: Optional[Path] = None,
    config_root: Optional[str] = None,
) -> List[Tuple[Path, str]]:
    """Return the ordered list of ``(root_dir, tier)`` pairs to scan.

    Discovery walks workspace first, then user; the order matters
    because workspace bundles **shadow** user bundles of the same
    name in :func:`discover_bundles`.

    The workspace tier is determined by:

    1. If ``config_root`` is set: ``Path(config_root) / <domain>`` —
       lets a session decouple where it reads config from where its
       agent's filesystem tools point (see
       :func:`shared.config_resolver.resolve_config_search_path`).
    2. Else if ``workspace_path`` is set: ``workspace_path / domain_subpath``
       — today's behavior, unchanged.
    3. Else: workspace tier omitted; only the user tier is scanned.

    The user tier (``~/<domain_subpath>``) is always appended.

    Args:
        workspace_path: Workspace root, or ``None`` if unknown. When
            None, the workspace tier is omitted (the user tier still
            applies).
        domain_subpath: Plugin-relative tier-root suffix, e.g.
            ``Path(".jaato/references")`` for the references plugin or
            ``Path(".jaato/agents")`` for an agent plugin. Joined with
            both the workspace path and ``Path.home()``.  When
            ``config_root`` is set, only the *trailing* segment of
            ``domain_subpath`` is joined onto ``config_root`` (the
            ``.jaato/`` prefix that already lives in ``config_root``
            is stripped) — so callers don't have to choose between two
            shapes.
        user_home: Override for ``Path.home()``. Test seam — production
            code passes ``None`` to use the real home directory.
        config_root: Optional read-only-config root override (see
            :func:`shared.config_resolver.resolve_config_search_path`).
            When ``None`` (the default), behavior is unchanged.

    Returns:
        Ordered list of ``(absolute_root_dir, tier_name)``. Roots that
        do not exist on disk are still returned; ``discover_bundles``
        treats a missing root as "no bundles in this tier" without
        raising.
    """
    # When no explicit ``config_root`` is provided, honor the
    # ``JAATO_CONFIG_ROOT`` env var (exported by
    # ``JaatoServer._in_workspace``).  This lets plugins whose
    # ``initialize()`` runs before the registry's ``set_config_root``
    # broadcast — and therefore can't read the override from
    # ``self._config_root`` — still pick it up on first discovery.
    import os as _os
    from jaato_server.shared.session_context import get_config_root
    effective_config_root = config_root or get_config_root()

    roots: List[Tuple[Path, str]] = []
    if effective_config_root:
        # ``domain_subpath`` ships as ``Path(".jaato/<domain>")`` from
        # most callers; ``config_root`` is already the analog of
        # ``<workspace>/.jaato``, so we want only the inner segments
        # (e.g. ``"references"``).  ``Path.parts`` lets us strip the
        # leading ``.jaato`` if present without touching the rest.
        parts = domain_subpath.parts
        if parts and parts[0] == ".jaato":
            inner = Path(*parts[1:]) if len(parts) > 1 else Path()
        else:
            inner = domain_subpath
        roots.append((
            Path(effective_config_root).expanduser().resolve() / inner,
            BUNDLE_TIER_WORKSPACE,
        ))
    elif workspace_path is not None:
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
    root never accidentally pollutes the catalog. "A manifest" means
    ``bundle.json`` and nothing else, and nothing about its *content*
    is required -- so a directory carrying only a domain's own metadata
    (the references plugin's ``embedding_config.json``) is NOT a
    bundle, while a bundle that declares no vector index is discovered
    exactly like one that does.

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
        try:
            if not refs_dir.is_dir():
                continue
            children = sorted(refs_dir.iterdir())
        except (PermissionError, OSError) as exc:
            # A confined session is CORRECTLY denied this tier — e.g.
            # ~/.jaato/references (the USER tier) under AppArmor: is_dir() /
            # iterdir() RAISE PermissionError for EACCES, which pathlib does NOT
            # ignore.  Not reaching the user tier from a confined runner is BY
            # DESIGN, not a failure — so DEBUG-skip it rather than letting a
            # misleading ``PermissionError: ~/.jaato/references`` propagate and
            # read like a bug to anyone bug-seeking.  Other (readable) tiers
            # still load.
            logger.debug(
                "discover_bundles: tier %s root %s not scannable (%s); "
                "skipping", tier, refs_dir, exc,
            )
            continue

        root = load_bundle(refs_dir, name=ROOT_BUNDLE_NAME, tier=tier)
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

        for child in children:
            if not child.is_dir():
                continue
            sub = load_bundle(child, name=child.name, tier=tier)
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


def write_bundle_manifest(
    directory: Path,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write ``bundle.json`` into ``directory``, creating it if needed.

    This is the minimum a domain has to do to declare a directory a
    bundle. Every field is optional: the file's *presence* is the
    marker, and its keys are informational. A domain with more to say
    about its own bundles writes its own file beside this one (the
    references plugin writes ``embedding_config.json``) rather than
    adding keys here that only it understands.

    Uses an atomic write (``.tmp`` + rename) so a crash mid-write
    cannot leave a half-written manifest behind.

    Args:
        directory: Bundle directory. Created (with parents) if absent.
        name: Informational bundle name. Discovery derives the real
            name from the directory, so this is documentation for a
            human reading the file, never authority. Omitted from the
            payload when empty — :data:`ROOT_BUNDLE_NAME` IS the empty
            string, and a root bundle has no name to record.
        description: One-line description of what the bundle holds.
        extra: Additional top-level keys to record. Domain-specific
            keys do NOT belong here — this exists for envelope-level
            metadata a future generic feature may add.

    Returns:
        Absolute path to the written manifest.
    """
    directory.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {}
    if name:
        payload["name"] = name
    if description is not None:
        payload["description"] = description
    if extra:
        payload.update(extra)

    target = directory / BUNDLE_MANIFEST_FILENAME
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(target)
    return target.resolve()
