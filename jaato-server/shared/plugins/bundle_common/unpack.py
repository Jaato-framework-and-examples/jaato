"""Unpack a v1 or v2 bundle archive across registered handlers.

Companion to :mod:`bundle_common.pack`. Reads the envelope, validates
the format version, and routes each ``kinds/<kind>/`` subtree into the
matching handler's domain root. v1 archives (single-kind, references-
only, ``bundle/`` + ``payload/`` at the top level) are transparently
mapped onto the ``"references"`` kind so previously shipped archives
keep working.

Three collision modes are supported, matching what was already shipped
for the references-only unpack:

* ``ERROR`` (default) — refuse to clobber an existing bundle of the
  same name on the recipient side.
* ``OVERWRITE`` — delete the existing bundle's contents and replace.
* ``MERGE`` — delegate to each handler's reconcile after extracting
  into a temporary location, letting the handler decide how its
  domain handles a conflict.

The unpacker doesn't reconcile sidecars itself — it leaves that to
the caller, which already knows when to call
:meth:`BundleEntryHandler.reconcile_bundle` in light of post-unpack
state (re-attaching matchers, refreshing live catalogs, etc.).
"""

from __future__ import annotations

import json
import logging
import shutil
import tarfile
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .bundle import (
    BUNDLE_TIER_USER,
    BUNDLE_TIER_WORKSPACE,
    EMBEDDING_CONFIG_FILENAME,
    ROOT_BUNDLE_NAME,
    VALID_BUNDLE_TIERS,
)
from .handler import BundleEntryHandler, BundleEntryRegistry
from .pack import (
    ARCHIVE_BUNDLE_DIR,
    ARCHIVE_ENVELOPE_FILENAME,
    ARCHIVE_FORMAT_VERSION,
    ARCHIVE_FORMAT_VERSION_V1,
    ARCHIVE_KINDS_DIR,
    ARCHIVE_PAYLOAD_DIR,
    LEGACY_V1_KIND,
)

logger = logging.getLogger(__name__)


# Format versions this build can read. v2 is the native format;
# v1 is read-only legacy support for archives produced before the
# pack/unpack lift.
_SUPPORTED_FORMATS = (ARCHIVE_FORMAT_VERSION_V1, ARCHIVE_FORMAT_VERSION)


class UnpackMode(str, Enum):
    """Collision policy when an unpack target already exists."""

    ERROR = "error"
    OVERWRITE = "overwrite"
    MERGE = "merge"


class UnpackError(ValueError):
    """Raised when unpack cannot proceed safely.

    Common causes: archive missing the envelope, format_version
    outside the supported range, tar member with a path that escapes
    the staging directory, target collision without an explicit mode,
    archive references a kind the receiver hasn't registered.
    """


@dataclass
class KindUnpackResult:
    """Per-kind installation result inside an :class:`UnpackResult`."""

    kind: str
    target_dir: Path
    target_tier: str
    target_name: str
    entry_count: int


@dataclass
class UnpackResult:
    """Outcome of a successful :func:`unpack_archive` call.

    Attributes:
        archive_path: Path to the archive that was unpacked.
        format_version: Format version detected (1 for legacy
            references-only archives, 2 for multi-kind).
        target_tier: Tier the archive's contents were installed under.
        target_name: Bundle name on the recipient side.
        mode: Collision policy used.
        kinds: One entry per kind that contributed, in the archive's
            sorted-kind order.
    """

    archive_path: Path
    format_version: int
    target_tier: str
    target_name: str
    mode: UnpackMode
    kinds: List[KindUnpackResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def read_envelope(archive_path: Path) -> Dict[str, Any]:
    """Read and validate the ``bundle_archive.json`` envelope.

    Returns the parsed envelope dict. Raises :class:`UnpackError` for
    malformed archives and :class:`FileNotFoundError` if the archive
    itself doesn't exist. Exposed separately so a caller can preview
    an archive (``bundle inspect``-style) without extracting it.

    Args:
        archive_path: Path to a ``.tar.gz`` produced by :mod:`pack`.

    Raises:
        UnpackError: archive missing/malformed envelope, or unknown
            format version.
        FileNotFoundError: archive doesn't exist.
        tarfile.ReadError: archive isn't a valid tar.
    """
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"archive not found: {archive_path}")

    with tarfile.open(archive_path, "r:*") as tf:
        try:
            member = tf.getmember(ARCHIVE_ENVELOPE_FILENAME)
        except KeyError:
            raise UnpackError(
                f"archive {archive_path.name} is missing "
                f"{ARCHIVE_ENVELOPE_FILENAME}; not a packed bundle"
            )
        extracted = tf.extractfile(member)
        if extracted is None:
            raise UnpackError(
                f"{ARCHIVE_ENVELOPE_FILENAME} in {archive_path.name} "
                f"is not a regular file"
            )
        try:
            envelope = json.loads(extracted.read().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise UnpackError(
                f"{ARCHIVE_ENVELOPE_FILENAME} in {archive_path.name} "
                f"is not valid JSON: {e}"
            ) from e

    if not isinstance(envelope, dict):
        raise UnpackError(f"{ARCHIVE_ENVELOPE_FILENAME} must be a JSON object")

    fmt = envelope.get("format_version")
    if fmt not in _SUPPORTED_FORMATS:
        raise UnpackError(
            f"archive format_version {fmt!r} is not supported by this "
            f"jaato build (supported: {_SUPPORTED_FORMATS})"
        )
    if "source_name" not in envelope:
        raise UnpackError("envelope missing 'source_name' field")
    return envelope


def unpack_archive(
    archive_path: Path,
    *,
    registry: BundleEntryRegistry,
    target_tier: str,
    target_name: Optional[str] = None,
    mode: UnpackMode = UnpackMode.ERROR,
    workspace_path: Optional[Path] = None,
    user_home: Optional[Path] = None,
) -> UnpackResult:
    """Unpack ``archive_path`` into the appropriate domain roots.

    Each kind in the archive is routed to the registered handler with
    that kind, which determines the on-disk install directory via
    ``handler.domain_subpath`` (joined with ``workspace_path`` /
    ``user_home``). If the archive declares a kind the registry
    doesn't know, the unpack aborts before any disk writes.

    Args:
        archive_path: ``.tar.gz`` produced by :mod:`pack`.
        registry: Lookup table for handlers. Production code passes
            the global :data:`bundle_common.handler.registry`; tests
            pass an isolated instance.
        target_tier: ``BUNDLE_TIER_WORKSPACE`` or ``BUNDLE_TIER_USER``.
        target_name: Bundle name on the recipient side. ``None``
            means "use the archive's source_name". Pass ``""`` to
            install as the root bundle.
        mode: Collision policy for the per-kind targets.
        workspace_path: Workspace root for resolving the workspace
            tier. Required if ``target_tier`` is workspace.
        user_home: Override for ``Path.home()`` (test seam).

    Returns:
        :class:`UnpackResult` summarizing the per-kind installations.

    Raises:
        UnpackError: Validation failed, traversal attempt, missing
            handler for a declared kind, or target collision in
            ``ERROR`` mode.
        FileNotFoundError: Archive missing.
    """
    archive_path = Path(archive_path)
    if target_tier not in VALID_BUNDLE_TIERS:
        raise UnpackError(
            f"unknown target_tier {target_tier!r}; "
            f"expected one of {', '.join(VALID_BUNDLE_TIERS)}"
        )

    envelope = read_envelope(archive_path)
    fmt = envelope["format_version"]
    name = (
        target_name
        if target_name is not None
        else envelope.get("source_name", "")
    )
    if not isinstance(name, str):
        raise UnpackError("target_name must be a string")

    # Determine which kinds the archive carries and validate that every
    # declared kind has a registered handler. Failing fast keeps a
    # half-installed archive off disk.
    if fmt == ARCHIVE_FORMAT_VERSION_V1:
        archive_kinds = [LEGACY_V1_KIND]
    else:
        archive_kinds = list(envelope.get("kinds", []))
    missing = [k for k in archive_kinds if registry.get(k) is None]
    if missing:
        raise UnpackError(
            f"archive declares kinds {missing!r} but no matching handlers "
            f"are registered (registered: {registry.kinds() or '(none)'})"
        )

    # Stage the entire archive once. Each kind's subtree is then moved
    # into place under its handler's domain root.
    with tempfile.TemporaryDirectory(prefix="jaato-unpack-") as staging_str:
        staging = Path(staging_str)
        _safe_extract(archive_path, staging)

        per_kind_roots = _resolve_staged_kind_roots(staging, fmt, archive_kinds)

        kind_results: List[KindUnpackResult] = []
        for kind in archive_kinds:
            handler = registry.get(kind)
            assert handler is not None  # validated above
            kind_root = per_kind_roots[kind]
            staged_bundle = kind_root / ARCHIVE_BUNDLE_DIR
            staged_payload = kind_root / ARCHIVE_PAYLOAD_DIR
            if not (staged_bundle / EMBEDDING_CONFIG_FILENAME).is_file():
                raise UnpackError(
                    f"archive's kind={kind!r} subtree is missing "
                    f"bundle/{EMBEDDING_CONFIG_FILENAME}"
                )

            target_dir = _resolve_target_dir(
                handler,
                tier=target_tier,
                name=name,
                workspace_path=workspace_path,
                user_home=user_home,
            )
            kind_results.append(_install_kind(
                staged_bundle=staged_bundle,
                staged_payload=staged_payload,
                target_dir=target_dir,
                kind=kind,
                target_tier=target_tier,
                target_name=name,
                mode=mode,
            ))

    return UnpackResult(
        archive_path=archive_path.resolve(),
        format_version=fmt,
        target_tier=target_tier,
        target_name=name,
        mode=mode,
        kinds=kind_results,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_staged_kind_roots(
    staging: Path,
    fmt: int,
    kinds: List[str],
) -> Dict[str, Path]:
    """Map each kind to its staging root directory.

    For v2 the layout is ``staging/kinds/<kind>/{bundle,payload}/``.
    For v1 there is exactly one kind (``"references"``) and its
    bundle / payload dirs live at ``staging/{bundle,payload}/``.
    """
    if fmt == ARCHIVE_FORMAT_VERSION_V1:
        # v1 archives have a single kind; alias the top-level dirs
        # to the legacy kind. We don't move files — the caller reads
        # bundle/ and payload/ from the staging root.
        return {LEGACY_V1_KIND: staging}
    kinds_dir = staging / ARCHIVE_KINDS_DIR
    if not kinds_dir.is_dir():
        raise UnpackError(
            f"v2 archive missing {ARCHIVE_KINDS_DIR}/ directory"
        )
    return {kind: kinds_dir / kind for kind in kinds}


def _resolve_target_dir(
    handler: BundleEntryHandler,
    *,
    tier: str,
    name: str,
    workspace_path: Optional[Path],
    user_home: Optional[Path],
) -> Path:
    """Compute the destination directory for a given kind+tier+name.

    Mirrors the discovery logic: workspace tier joins
    ``handler.domain_subpath`` with ``workspace_path``; user tier joins
    it with ``Path.home()`` (or ``user_home``). The root bundle
    install directory is the tier root itself; named bundles get a
    subdirectory.
    """
    if tier == BUNDLE_TIER_WORKSPACE:
        if workspace_path is None:
            raise UnpackError(
                "target_tier='workspace' but workspace_path is None"
            )
        tier_root = Path(workspace_path) / handler.domain_subpath
    else:
        home = user_home if user_home is not None else Path.home()
        tier_root = home / handler.domain_subpath
    if name == ROOT_BUNDLE_NAME:
        return tier_root
    return tier_root / name


def _install_kind(
    *,
    staged_bundle: Path,
    staged_payload: Path,
    target_dir: Path,
    kind: str,
    target_tier: str,
    target_name: str,
    mode: UnpackMode,
) -> KindUnpackResult:
    """Move a single kind's staged subtree into its destination dir.

    Honours ``mode`` for the per-kind collision check. Each kind is
    installed atomically w.r.t. itself; if one kind fails, prior kinds
    have already been written and won't roll back. That's an
    intentional trade-off — the failure modes (target exists,
    permissions, disk full) are usually surfaced by the first kind, so
    callers should treat partial-install as "fix the cause and rerun
    with --overwrite" rather than a transactional concern.
    """
    target_exists = (
        target_dir.is_dir()
        and (target_dir / EMBEDDING_CONFIG_FILENAME).is_file()
    )
    if target_exists and mode == UnpackMode.ERROR:
        raise UnpackError(
            f"target {target_dir} already contains a bundle (kind={kind!r}); "
            f"pass mode=overwrite to replace or mode=merge to merge"
        )

    if mode == UnpackMode.MERGE:
        # MERGE leaves on-disk merge logic up to the caller — the
        # bundle subsystem typically calls a handler-side reconcile
        # after this returns. For now we layer the staged contents
        # *into* the existing target without removing anything; the
        # caller's reconcile resolves the resulting drift.
        target_dir.mkdir(parents=True, exist_ok=True)
        _layer_in_place(staged_bundle, target_dir)
        if staged_payload.is_dir() and any(staged_payload.iterdir()):
            payload_dst = target_dir / ARCHIVE_PAYLOAD_DIR
            payload_dst.mkdir(exist_ok=True)
            _layer_in_place(staged_payload, payload_dst)
    else:
        if target_exists and mode == UnpackMode.OVERWRITE:
            _safe_replace_bundle(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        # Move staged bundle dir contents into target_dir.
        for item in staged_bundle.iterdir():
            shutil.move(str(item), str(target_dir / item.name))
        # Move payload subtree alongside the bundle so rewritten
        # ``payload/<slot>/<basename>`` paths resolve correctly when
        # the catalog loads.
        if staged_payload.is_dir() and any(staged_payload.iterdir()):
            payload_dst = target_dir / ARCHIVE_PAYLOAD_DIR
            payload_dst.mkdir(exist_ok=True)
            for item in staged_payload.iterdir():
                shutil.move(str(item), str(payload_dst / item.name))

    entry_count = sum(
        1 for p in target_dir.glob("*.json")
        if p.name != EMBEDDING_CONFIG_FILENAME
    )
    return KindUnpackResult(
        kind=kind,
        target_dir=target_dir.resolve(),
        target_tier=target_tier,
        target_name=target_name,
        entry_count=entry_count,
    )


def _layer_in_place(src: Path, dst: Path) -> None:
    """Move every immediate child of ``src`` into ``dst``, overwriting
    existing files of the same name. Used by ``MERGE`` mode where we
    don't want to clobber the destination wholesale but do want
    incoming files to win on per-file conflicts.
    """
    for item in src.iterdir():
        target = dst / item.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(item), str(target))


def _safe_extract(archive_path: Path, dest: Path) -> None:
    """Extract a tar archive into ``dest`` rejecting traversal attempts.

    Every member's normalized destination must remain inside ``dest``.
    Absolute paths, ``..`` segments, and symlinks/hardlinks pointing
    outside ``dest`` are rejected with :class:`UnpackError` *before*
    any bytes are written, so a malformed archive cannot leave stray
    files on disk after a partial extraction failure.
    """
    dest_resolved = dest.resolve()
    with tarfile.open(archive_path, "r:*") as tf:
        for member in tf.getmembers():
            target = (dest_resolved / member.name).resolve()
            try:
                target.relative_to(dest_resolved)
            except ValueError:
                raise UnpackError(
                    f"refusing to extract {member.name!r} — path escapes "
                    f"the unpack directory"
                )
            if member.issym() or member.islnk():
                link_target = member.linkname
                if Path(link_target).is_absolute():
                    raise UnpackError(
                        f"refusing to extract symlink {member.name!r} → "
                        f"absolute path {link_target!r}"
                    )
                resolved_link = (target.parent / link_target).resolve()
                try:
                    resolved_link.relative_to(dest_resolved)
                except ValueError:
                    raise UnpackError(
                        f"refusing to extract symlink {member.name!r} → "
                        f"{link_target!r} (escapes unpack directory)"
                    )
        try:
            tf.extractall(dest, filter="data")  # type: ignore[arg-type]
        except TypeError:
            tf.extractall(dest)


def _safe_replace_bundle(target_dir: Path) -> None:
    """Remove an existing bundle's artefacts in ``OVERWRITE`` mode.

    We delete only files we recognize as bundle artefacts: the
    manifest, any sidecar ``.npy``, ``*.json`` entry files, and the
    ``payload/`` subtree. Stray non-bundle files (notes, READMEs)
    are left in place — operators sometimes drop project-specific
    files next to a bundle and clobbering them would be surprising.
    """
    if not target_dir.is_dir():
        return
    manifest = target_dir / EMBEDDING_CONFIG_FILENAME
    if manifest.is_file():
        manifest.unlink()
    for npy in target_dir.glob("*.npy"):
        npy.unlink()
    for npy_lock in target_dir.glob("*.npy.lock"):
        try:
            npy_lock.unlink()
        except FileNotFoundError:
            pass
    for js in target_dir.glob("*.json"):
        js.unlink()
    payload = target_dir / ARCHIVE_PAYLOAD_DIR
    if payload.is_dir():
        shutil.rmtree(payload)
